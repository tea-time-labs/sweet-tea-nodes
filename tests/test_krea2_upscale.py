import copy
import importlib
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F


ROOT = pathlib.Path(__file__).resolve().parents[1]


class Model:
    def __init__(self):
        self.attachments = {}
    def clone(self):
        return copy.copy(self)
    def get_attachment(self, name):
        return self.attachments.get(name)
    def set_attachments(self, name, value):
        self.attachments = {**self.attachments, name: value}


class VAE:
    def encode(self, image):
        return image.movedim(-1, 1)
    def decode(self, samples):
        return samples.movedim(1, -1)


def resize(x, w, h, *_):
    return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


class UpscaleTests(unittest.TestCase):
    def setUp(self):
        self.seen = []
        comfy = types.ModuleType("comfy")
        comfy.utils = types.ModuleType("comfy.utils")
        comfy.utils.common_upscale = resize
        comfy.model_management = types.ModuleType("comfy.model_management")
        comfy.model_management.get_torch_device = lambda: "cpu"
        comfy.samplers = types.ModuleType("comfy.samplers")
        comfy.samplers.KSampler = types.SimpleNamespace(SAMPLERS=["euler"], SCHEDULERS=["simple"])
        nodes = types.ModuleType("nodes")
        nodes.NODE_CLASS_MAPPINGS = {}
        def sample(**kwargs):
            self.seen.append(kwargs)
            return (kwargs["latent"],)
        nodes.common_ksampler = sample
        folders = types.ModuleType("folder_paths")
        folders.get_full_path = lambda *args: "/fixture/weights.safetensors"
        pkg = types.ModuleType("usdu_pose_tiled")
        pkg.__path__ = [str(ROOT / "usdu_pose_tiled")]
        self.modules = patch.dict(sys.modules, {"comfy": comfy, "comfy.utils": comfy.utils,
            "comfy.model_management": comfy.model_management, "comfy.samplers": comfy.samplers,
            "nodes": nodes, "folder_paths": folders, "usdu_pose_tiled": pkg})
        self.modules.start()
        for name in list(sys.modules):
            if name.startswith("usdu_pose_tiled."):
                del sys.modules[name]
        self.mod = importlib.import_module("usdu_pose_tiled.krea2_control")
        self.nodes, self.folders = nodes, folders
        # A coordinate ramp reveals repeating-full-hint and offset errors.
        x = torch.linspace(0, 1, 256).view(1, 1, 256, 1)
        self.image = x.expand(1, 128, 256, 3).clone()
        self.args = dict(image=self.image, model=Model(), positive=[], negative=[], vae=VAE(),
            upscale_by=2., seed=0, steps=1, cfg=1., sampler_name="euler", scheduler="simple",
            denoise=.25, tile_size=128, tile_padding=32, mask_blur=8,
            control_mode="Off", control_strength=.7, depth_lora="depth", canny_lora="canny",
            tile_lora="tile", depth_resolution=512, canny_low=.1, canny_high=.2)

    def tearDown(self):
        for name in list(sys.modules):
            if name.startswith("usdu_pose_tiled."):
                del sys.modules[name]
        self.modules.stop()

    def test_off_and_zero_strength_need_no_control_nodes_or_weights(self):
        self.folders.get_full_path = lambda *a: None
        for mode, strength in [("Off", .7), ("Depth", 0.)]:
            result = self.mod.STKrea2UltimateUpscale().upscale(**{**self.args,
                "control_mode": mode, "control_strength": strength})[0]
            self.assertEqual(tuple(result.shape), (1, 256, 512, 3))
            self.assertLess(float((result - resize(self.image.movedim(-1, 1), 512, 256).movedim(1, -1)).abs().max()), .02)
        self.assertGreater(len(self.seen), 2)

    def test_control_tracks_each_tile_including_padding_and_full_frame(self):
        class Loader:
            def load_lora(self, model, *args): return (model,)
        class Apply:
            def apply(self, model, latent):
                m = model.clone()
                m.hint = latent["samples"]
                return (m,)
        self.nodes.NODE_CLASS_MAPPINGS.update(Krea2ControlLoRALoader=Loader, Krea2ControlApply=Apply)
        for tile_size, padding in [(128, 32), (1024, 0)]:
            self.seen.clear()
            self.mod.STKrea2UltimateUpscale().upscale(**{**self.args, "control_mode": "Tile",
                "tile_size": tile_size, "tile_padding": padding, "mask_blur": 0})
            for call in self.seen:
                hint, actual = call["model"].hint, call["latent"]["samples"]
                self.assertEqual(hint.shape, actual.shape)
                self.assertLess(float((hint - actual).abs().max()), .025)
            if tile_size == 128:
                means = [float(x["model"].hint.mean()) for x in self.seen]
                self.assertGreater(max(means) - min(means), .5)

    def test_canny_uses_per_tile_references(self):
        class Canny:
            def detect_edge(self, image, *a): return (image,)
        class LoRA:
            def load_lora_model_only(self, model, *a): return (model,)
        class Edit:
            @classmethod
            def execute(cls, model, reference):
                m = model.clone()
                m.reference = reference["samples"]
                return (m,)
        self.nodes.LoraLoaderModelOnly = LoRA
        self.nodes.NODE_CLASS_MAPPINGS["NK2EInContextEditNode"] = Edit
        canny = types.ModuleType("comfy_extras.nodes_canny")
        canny.Canny = Canny
        with patch.dict(sys.modules, {"comfy_extras.nodes_canny": canny}):
            self.mod.STKrea2UltimateUpscale().upscale(**{**self.args, "control_mode": "Canny"})
        for call in self.seen:
            self.assertLess(float((call["model"].reference - call["latent"]["samples"]).abs().max()), .025)

    def test_missing_selected_weights_fail_before_sampling(self):
        self.folders.get_full_path = lambda *a: None
        with self.assertRaisesRegex(FileNotFoundError, "approval"):
            self.mod.STKrea2UltimateUpscale().upscale(**{**self.args, "control_mode": "Tile"})
        self.assertEqual(self.seen, [])


if __name__ == "__main__":
    unittest.main()
