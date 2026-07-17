import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "wan_vace_hybrid_subject"


def _load_node_module():
    name = "wan_vace_hybrid_subject.nodes_wan_vace_hybrid_subject"
    spec = importlib.util.spec_from_file_location(name, PACKAGE / "nodes_wan_vace_hybrid_subject.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _common_upscale(value, width, height, mode, _crop):
    interpolation = "nearest-exact" if mode == "nearest-exact" else "bilinear"
    options = {"size": (height, width), "mode": interpolation}
    if interpolation == "bilinear":
        options["align_corners"] = False
    return F.interpolate(value, **options)


def _conditioning_set_values(conditioning, values, append=False):
    output = dict(conditioning)
    for key, value in values.items():
        if append and key in output:
            existing = output[key] if isinstance(output[key], list) else [output[key]]
            output[key] = existing + (value if isinstance(value, list) else [value])
        else:
            output[key] = value
    return output


class _Wan21LatentFormat:
    def process_out(self, value):
        return value


class _FakeVAE:
    def encode(self, video):
        frames = ((video.shape[0] - 1) // 4) + 1
        sampled = video[::4][:frames]
        pooled = F.avg_pool2d(sampled.permute(0, 3, 1, 2), kernel_size=8, stride=8)
        latent = pooled.mean(dim=1, keepdim=True).repeat(1, 16, 1, 1)
        return {"samples": latent.permute(1, 0, 2, 3).unsqueeze(0).contiguous()}


def _dependency_stubs():
    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.common_upscale = _common_upscale
    comfy_management = types.ModuleType("comfy.model_management")
    comfy_management.intermediate_device = lambda: "cpu"
    comfy_latents = types.ModuleType("comfy.latent_formats")
    comfy_latents.Wan21 = _Wan21LatentFormat
    comfy.utils = comfy_utils
    comfy.model_management = comfy_management
    comfy.latent_formats = comfy_latents

    nodes = types.ModuleType("nodes")
    nodes.MAX_RESOLUTION = 16384
    helpers = types.ModuleType("node_helpers")
    helpers.conditioning_set_values = _conditioning_set_values
    package = types.ModuleType("wan_vace_hybrid_subject")
    package.__path__ = [str(PACKAGE)]

    return {
        "comfy": comfy,
        "comfy.utils": comfy_utils,
        "comfy.model_management": comfy_management,
        "comfy.latent_formats": comfy_latents,
        "nodes": nodes,
        "node_helpers": helpers,
        "wan_vace_hybrid_subject": package,
    }


class WanVaceHybridSubjectContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = mock.patch.dict(sys.modules, _dependency_stubs(), clear=False)
        cls.modules.start()
        cls.node_module = _load_node_module()

    @classmethod
    def tearDownClass(cls):
        cls.modules.stop()
        sys.modules.pop("wan_vace_hybrid_subject.nodes_wan_vace_hybrid_subject", None)

    @staticmethod
    def _video(frames):
        values = torch.linspace(0.0, 1.0, frames * 16 * 16 * 3)
        return values.view(frames, 16, 16, 3)

    @staticmethod
    def _mask():
        mask = torch.zeros(9, 16, 16)
        mask[:, 4:12, 4:12] = 1.0
        return mask

    def _encode(self, **overrides):
        arguments = {
            "positive": {},
            "negative": {},
            "vae": _FakeVAE(),
            "width": 16,
            "height": 16,
            "length": 9,
            "batch_size": 1,
            "strength": 1.0,
            "control_strength": -1.0,
            "reference_strength": -1.0,
            "source_luma_strength": 0.0,
            "source_detail_strength": 0.0,
            "background_preserve_strength": 0.0,
            "mask_feather_px": 0.0,
            "control_video": self._video(9),
            "control_masks": self._mask(),
            "reference_image": self._video(1),
        }
        arguments.update(overrides)
        node = self.node_module.WanVaceToVideoHybridSubjectSwap()
        return node.encode(**arguments)

    def test_emits_conditioning_and_latent_with_reference_trim(self):
        positive, negative, latent, trim = self._encode(batch_size=2, strength=1.1)

        self.assertEqual(trim, 1)
        self.assertEqual(positive["vace_strength"], [1.1])
        self.assertEqual(negative["vace_strength"], [1.1])
        self.assertEqual(tuple(positive["vace_frames"][0].shape), (1, 32, 4, 2, 2))
        self.assertEqual(tuple(positive["vace_mask"][0].shape), (1, 64, 4, 2, 2))
        self.assertEqual(tuple(latent["samples"].shape), (2, 16, 4, 2, 2))

    def test_source_video_remains_one_composite_context(self):
        positive, _, latent, trim = self._encode(
            source_video=self._video(9),
            control_strength=0.45,
            reference_strength=1.2,
            source_luma_strength=0.3,
            source_detail_strength=0.15,
            background_preserve_strength=0.9,
        )

        self.assertEqual(trim, 1)
        self.assertEqual(len(positive["vace_frames"]), 1)
        self.assertEqual(positive["vace_strength"], [1.2])
        self.assertEqual(tuple(latent["samples"].shape), (1, 16, 4, 2, 2))

    def test_control_mask_changes_the_public_conditioning_mask(self):
        masked, _, _, _ = self._encode(source_video=self._video(9))
        unmasked, _, _, _ = self._encode(source_video=self._video(9), control_masks=None)

        self.assertFalse(torch.allclose(masked["vace_mask"][0], unmasked["vace_mask"][0]))

    def test_control_and_reference_strengths_are_independent(self):
        reference_heavy, _, _, _ = self._encode(control_strength=0.35, reference_strength=1.6)
        control_heavy, _, _, _ = self._encode(control_strength=1.6, reference_strength=0.35)

        self.assertEqual(reference_heavy["vace_strength"], [1.6])
        self.assertEqual(control_heavy["vace_strength"], [1.6])
        self.assertFalse(torch.allclose(reference_heavy["vace_frames"][0], control_heavy["vace_frames"][0]))


if __name__ == "__main__":
    unittest.main()
