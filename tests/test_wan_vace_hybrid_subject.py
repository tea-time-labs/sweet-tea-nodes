import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - test environment dependent
    torch = None
    F = None


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "wan_vace_hybrid_subject"


def _load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _common_upscale(x, width, height, mode, _crop):
    if F is None:
        raise RuntimeError("torch is required for hybrid subject node tests")
    interp_mode = "nearest-exact" if mode == "nearest-exact" else "bilinear"
    kwargs = {"size": (height, width), "mode": interp_mode}
    if interp_mode == "bilinear":
        kwargs["align_corners"] = False
    return F.interpolate(x, **kwargs)


def _conditioning_set_values(cond, values, append=False):
    out = dict(cond)
    for key, value in values.items():
        if append and key in out:
            existing = out[key]
            existing = existing if isinstance(existing, list) else [existing]
            addition = value if isinstance(value, list) else [value]
            out[key] = existing + addition
        else:
            out[key] = value
    return out


class _Wan21LatentFormat:
    def process_out(self, x):
        return x


class _FakeVAE:
    def encode(self, x_nhwc):
        if F is None:
            raise RuntimeError("torch is required for hybrid subject node tests")
        t, h, w, _ = x_nhwc.shape
        latent_t = ((t - 1) // 4) + 1
        sampled = x_nhwc[::4][:latent_t]
        pooled = F.avg_pool2d(sampled.permute(0, 3, 1, 2), kernel_size=8, stride=8)
        base = pooled.mean(dim=1, keepdim=True)
        samples = base.repeat(1, 16, 1, 1).permute(1, 0, 2, 3).unsqueeze(0)
        return {"samples": samples.contiguous()}


def _build_dependency_stubs():
    comfy_mod = types.ModuleType("comfy")
    comfy_utils_mod = types.ModuleType("comfy.utils")
    comfy_utils_mod.common_upscale = _common_upscale
    comfy_mm_mod = types.ModuleType("comfy.model_management")
    comfy_mm_mod.intermediate_device = lambda: "cpu"
    comfy_latent_mod = types.ModuleType("comfy.latent_formats")
    comfy_latent_mod.Wan21 = _Wan21LatentFormat
    comfy_mod.utils = comfy_utils_mod
    comfy_mod.model_management = comfy_mm_mod
    comfy_mod.latent_formats = comfy_latent_mod

    nodes_mod = types.ModuleType("nodes")
    nodes_mod.MAX_RESOLUTION = 16384

    node_helpers_mod = types.ModuleType("node_helpers")
    node_helpers_mod.conditioning_set_values = _conditioning_set_values

    pkg_mod = types.ModuleType("wan_vace_hybrid_subject")
    pkg_mod.__path__ = [str(PKG_DIR)]

    return {
        "nodes": nodes_mod,
        "node_helpers": node_helpers_mod,
        "comfy": comfy_mod,
        "comfy.utils": comfy_utils_mod,
        "comfy.model_management": comfy_mm_mod,
        "comfy.latent_formats": comfy_latent_mod,
        "wan_vace_hybrid_subject": pkg_mod,
    }


@unittest.skipUnless(torch is not None, "torch not installed in test interpreter")
class WanVaceHybridSubjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._module_patcher = mock.patch.dict(sys.modules, _build_dependency_stubs(), clear=False)
        cls._module_patcher.start()
        sys.modules.pop("wan_vace_hybrid_subject.nodes_wan_vace_hybrid_subject", None)
        cls.node_mod = _load_module(
            "wan_vace_hybrid_subject.nodes_wan_vace_hybrid_subject",
            PKG_DIR / "nodes_wan_vace_hybrid_subject.py",
        )

    @classmethod
    def tearDownClass(cls):
        cls._module_patcher.stop()
        sys.modules.pop("wan_vace_hybrid_subject.nodes_wan_vace_hybrid_subject", None)

    def _make_node(self):
        return self.node_mod.WanVaceToVideoHybridSubjectSwap()

    def _make_video(self, frames, channels=3):
        data = torch.linspace(0.0, 1.0, frames * 16 * 16 * channels, dtype=torch.float32)
        return data.view(frames, 16, 16, channels)

    def _make_mask(self):
        mask = torch.zeros(9, 16, 16, dtype=torch.float32)
        mask[:, 4:12, 4:12] = 1.0
        return mask

    def test_depth_only_reference_mask_emits_one_context_and_trim_latent(self):
        node = self._make_node()
        positive, negative, latent, trim = node.encode(
            positive={},
            negative={},
            vae=_FakeVAE(),
            width=16,
            height=16,
            length=9,
            batch_size=2,
            strength=1.1,
            control_strength=-1.0,
            reference_strength=-1.0,
            source_luma_strength=0.0,
            source_detail_strength=0.0,
            background_preserve_strength=0.0,
            mask_feather_px=0.0,
            control_video=self._make_video(9),
            control_masks=self._make_mask(),
            reference_image=self._make_video(1),
        )

        self.assertEqual(trim, 1)
        self.assertEqual(len(positive["vace_frames"]), 1)
        self.assertEqual(positive["vace_strength"], [1.1])
        self.assertEqual(negative["vace_strength"], [1.1])
        self.assertEqual(tuple(positive["vace_frames"][0].shape), (1, 32, 4, 2, 2))
        self.assertEqual(tuple(positive["vace_mask"][0].shape), (1, 64, 4, 2, 2))
        self.assertEqual(tuple(latent["samples"].shape), (2, 16, 4, 2, 2))

    def test_source_video_modulates_single_native_like_context(self):
        node = self._make_node()
        positive, _, latent, trim = node.encode(
            positive={},
            negative={},
            vae=_FakeVAE(),
            width=16,
            height=16,
            length=9,
            batch_size=1,
            strength=1.0,
            control_strength=0.45,
            reference_strength=1.2,
            source_luma_strength=0.3,
            source_detail_strength=0.15,
            background_preserve_strength=0.9,
            mask_feather_px=0.0,
            control_video=self._make_video(9),
            source_video=self._make_video(9),
            control_masks=self._make_mask(),
            reference_image=self._make_video(1),
        )

        self.assertEqual(trim, 1)
        self.assertEqual(len(positive["vace_frames"]), 1)
        self.assertEqual(positive["vace_strength"], [1.2])
        self.assertEqual(tuple(positive["vace_frames"][0].shape), (1, 32, 4, 2, 2))
        self.assertEqual(tuple(positive["vace_mask"][0].shape), (1, 64, 4, 2, 2))
        self.assertEqual(tuple(latent["samples"].shape), (1, 16, 4, 2, 2))

    def test_zero_strength_hybrid_contexts_are_omitted(self):
        node = self._make_node()
        positive, _, _, _ = node.encode(
            positive={},
            negative={},
            vae=_FakeVAE(),
            width=16,
            height=16,
            length=9,
            batch_size=1,
            strength=0.8,
            control_strength=-1.0,
            reference_strength=0.0,
            source_luma_strength=0.0,
            source_detail_strength=0.0,
            background_preserve_strength=0.0,
            mask_feather_px=0.0,
            control_video=self._make_video(9),
            source_video=self._make_video(9),
            reference_image=None,
        )

        self.assertEqual(len(positive["vace_frames"]), 1)
        self.assertEqual(positive["vace_strength"], [0.8])

    def test_zero_hybrid_sliders_fall_back_to_masked_depth_behavior(self):
        node = self._make_node()
        masked_positive, _, _, _ = node.encode(
            positive={},
            negative={},
            vae=_FakeVAE(),
            width=16,
            height=16,
            length=9,
            batch_size=1,
            strength=1.0,
            control_strength=-1.0,
            reference_strength=-1.0,
            source_luma_strength=0.0,
            source_detail_strength=0.0,
            background_preserve_strength=0.0,
            mask_feather_px=0.0,
            control_video=self._make_video(9),
            source_video=self._make_video(9),
            control_masks=self._make_mask(),
            reference_image=self._make_video(1),
        )
        unmasked_positive, _, _, _ = node.encode(
            positive={},
            negative={},
            vae=_FakeVAE(),
            width=16,
            height=16,
            length=9,
            batch_size=1,
            strength=1.0,
            control_strength=-1.0,
            reference_strength=-1.0,
            source_luma_strength=0.0,
            source_detail_strength=0.0,
            background_preserve_strength=0.0,
            mask_feather_px=0.0,
            control_video=self._make_video(9),
            source_video=self._make_video(9),
            control_masks=None,
            reference_image=self._make_video(1),
        )

        self.assertEqual(len(masked_positive["vace_frames"]), 1)
        self.assertEqual(len(unmasked_positive["vace_frames"]), 1)
        self.assertFalse(torch.allclose(masked_positive["vace_mask"][0], unmasked_positive["vace_mask"][0]))

    def test_subject_mask_changes_single_composite_context(self):
        node = self._make_node()
        masked_positive, _, _, _ = node.encode(
            positive={},
            negative={},
            vae=_FakeVAE(),
            width=16,
            height=16,
            length=9,
            batch_size=1,
            strength=1.0,
            control_strength=-1.0,
            reference_strength=0.0,
            source_luma_strength=0.2,
            source_detail_strength=0.0,
            background_preserve_strength=0.7,
            mask_feather_px=0.0,
            control_video=self._make_video(9),
            source_video=self._make_video(9),
            control_masks=self._make_mask(),
        )
        unmasked_positive, _, _, _ = node.encode(
            positive={},
            negative={},
            vae=_FakeVAE(),
            width=16,
            height=16,
            length=9,
            batch_size=1,
            strength=1.0,
            control_strength=-1.0,
            reference_strength=0.0,
            source_luma_strength=0.2,
            source_detail_strength=0.0,
            background_preserve_strength=0.7,
            mask_feather_px=0.0,
            control_video=self._make_video(9),
            source_video=self._make_video(9),
            control_masks=None,
        )

        self.assertEqual(len(masked_positive["vace_frames"]), 1)
        self.assertEqual(len(unmasked_positive["vace_frames"]), 1)
        masked_hybrid_frames = masked_positive["vace_frames"][0]
        unmasked_hybrid_frames = unmasked_positive["vace_frames"][0]
        masked_hybrid_mask = masked_positive["vace_mask"][0]

        self.assertTrue(torch.any(masked_hybrid_mask > 0.0))
        self.assertFalse(torch.allclose(masked_hybrid_frames, unmasked_hybrid_frames))
        self.assertFalse(torch.allclose(masked_hybrid_mask, unmasked_positive["vace_mask"][0]))

    def test_control_and_reference_strengths_are_independent(self):
        node = self._make_node()
        positive, _, _, trim = node.encode(
            positive={},
            negative={},
            vae=_FakeVAE(),
            width=16,
            height=16,
            length=9,
            batch_size=1,
            strength=1.0,
            control_strength=0.35,
            reference_strength=1.6,
            source_luma_strength=0.0,
            source_detail_strength=0.0,
            background_preserve_strength=0.0,
            mask_feather_px=0.0,
            control_video=self._make_video(9),
            control_masks=self._make_mask(),
            reference_image=self._make_video(1),
        )

        self.assertEqual(trim, 1)
        self.assertEqual(len(positive["vace_frames"]), 1)
        self.assertEqual(positive["vace_strength"], [1.6])

        control_heavy, _, _, _ = node.encode(
            positive={},
            negative={},
            vae=_FakeVAE(),
            width=16,
            height=16,
            length=9,
            batch_size=1,
            strength=1.0,
            control_strength=1.6,
            reference_strength=0.35,
            source_luma_strength=0.0,
            source_detail_strength=0.0,
            background_preserve_strength=0.0,
            mask_feather_px=0.0,
            control_video=self._make_video(9),
            control_masks=self._make_mask(),
            reference_image=self._make_video(1),
        )

        self.assertEqual(control_heavy["vace_strength"], [1.6])
        self.assertFalse(torch.allclose(positive["vace_frames"][0], control_heavy["vace_frames"][0]))
