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

    def test_explicit_square_grid_runs_exact_count_on_a_rectangular_canvas(self):
        for count in (4, 9, 16, 25):
            with self.subTest(count=count):
                self.seen.clear()
                result, = self.mod.STKrea2UltimateUpscale().upscale(**self.args, tile_count=str(count))
                self.assertEqual(len(self.seen), count)
                self.assertEqual(tuple(result.shape), (1, 256, 512, 3))

        with self.assertRaisesRegex(ValueError, "Choose fewer tiles"):
            self.mod.STKrea2UltimateUpscale().upscale(**{**self.args, "upscale_by": .25}, tile_count="9")

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

    def test_vision_uses_distinct_crops_and_a_shared_latent_per_frame(self):
        from usdu_pose_tiled import krea2_vision

        class Tokenizer:
            llama_template = "system {}"

        class Clip:
            tokenizer = Tokenizer()

            def tokenize(self, text, *, images, llama_template):
                self.crops.append(images[0])
                self.templates.append(llama_template)
                return {"qwen3vl_4b": [[({"type": "image", "data": images[0]}, 1.)]]}

            def encode_from_tokens(self, tokens):
                image = tokens["qwen3vl_4b"][0][0][0]["data"]
                return image.mean().reshape(1, 1, 1)

        class Diffusion:
            patch, txtlayers, txtdim = 2, 1, 1

        class VisionModel(Model):
            def get_model_object(self, name):
                return Diffusion()

            def add_wrapper_with_key(self, key, name, wrapper):
                self.wrapper = wrapper

            def set_model_patch(self, patch, name):
                self.position = patch

        class VisionVAE(VAE):
            def encode_tiled(self, image):
                return F.avg_pool2d(super().encode(image), 8).unsqueeze(2)

            def decode_tiled(self, samples, overlap):
                return super().decode(samples.squeeze(2).repeat_interleave(8, -2).repeat_interleave(8, -1))

        def sample(*, model, positive, latent, **kwargs):
            x = latent["samples"]
            self.assertEqual(x.shape, (1, 3, 1, 16, 32))
            starts = []

            def predict(tile, time, context, mask, refs, options):
                ids = model.position({"img_ids": torch.zeros(1, 1, 3), "transformer_options": options})
                starts.append((tuple(ids["img_ids"][0, 0, 1:].tolist()), context.item()))
                return torch.ones_like(tile) * context.item()

            context = positive[0][0]
            blended = model.wrapper(predict, x, torch.zeros(1), context,
                                    transformer_options={"cond_or_uncond": [0]})
            self.assertEqual([pos for pos, _ in starts], [(0., 0.), (0., 6.), (2., 0.), (2., 6.)])
            self.assertLess(starts[0][1], starts[1][1])
            self.assertTrue(torch.isfinite(blended).all())
            self.assertGreater(blended[..., -1].mean(), blended[..., 0].mean())
            return ({"samples": blended},)

        model = VisionModel()
        clip = Clip()
        clip.crops, clip.templates = [], []
        source = torch.cat((self.image, self.image * .5))
        isolated = {
            "comfy.ldm.krea2.model": types.SimpleNamespace(SingleStreamDiT=Diffusion),
            "comfy.text_encoders.krea2": types.SimpleNamespace(Krea2Tokenizer=Tokenizer),
            "comfy.patcher_extension": types.SimpleNamespace(WrappersMP=types.SimpleNamespace(DIFFUSION_MODEL="model")),
        }
        with patch.dict(sys.modules, isolated), patch.object(krea2_vision, "common_ksampler", sample), \
             patch.object(sys.modules["comfy.model_management"], "throw_exception_if_processing_interrupted",
                          lambda: None, create=True):
            result, = self.mod.STKrea2UltimateUpscale().upscale(**{
                **self.args, "image": source, "upscale_by": 1., "control_mode": "Vision", "tile_count": "4",
                "model": model, "clip": clip, "vae": VisionVAE(), "vision_prompt": "fine detail",
            })

        self.assertEqual(result.shape, source.shape)
        self.assertFalse(hasattr(model, "wrapper"))
        self.assertEqual(len(clip.crops), 8)
        self.assertGreater(clip.crops[0].mean(), clip.crops[4].mean())
        self.assertTrue(all(krea2_vision.VISION_BLOCK in template for template in clip.templates))

    def test_reference_strength_matches_query_scoped_attention_for_two_guides(self):
        from usdu_pose_tiled.krea2_guides import reference_attention

        generator = torch.Generator().manual_seed(48)
        q, k, v = [torch.randn(2, 3, 14, 8, generator=generator) for _ in range(3)]
        strengths = (.2, .6)
        mask = torch.zeros(1, 1, 14, 14)
        mask[..., :7, 7:11] = torch.tensor(strengths[0]).log()
        mask[..., :7, 11:] = torch.tensor(strengths[1]).log()
        opts = {"reference_image_num_tokens": [4, 3], "block_type": "single"}
        result = reference_attention(strengths)(None, q, k, v, 3, transformer_options=opts)
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        self.assertTrue(torch.allclose(result, expected.transpose(1, 2).flatten(2), atol=1e-6))

    def test_padded_tiles_with_identical_origins_receive_distinct_aligned_guides(self):
        from usdu_pose_tiled import krea2_vision as vision
        from usdu_pose_tiled.krea2_guides import TileGuides

        # At large padding two different cores can start at the same coordinate.
        grid = self.mod.square_tile_grid(192, 192, "9")
        tiles = vision._tiles(192, 192, 192, 192, 64, 2, grid)
        self.assertEqual((tiles[0].top, tiles[0].left), (tiles[1].top, tiles[1].left))
        self.assertNotEqual(tiles[0].right, tiles[1].right)
        latent = torch.arange(24*24, dtype=torch.float32).reshape(1, 1, 24, 24)
        guides = TileGuides([(latent, 1.)], latent, torch.ones(3, 4), .5, 2)
        for tile in tiles:
            crop = latent[tile.region]
            refs, options = guides(tile, crop, {})
            self.assertTrue(torch.equal(refs[0], crop))
            depth = F.unfold(crop, 2, stride=2).sum(1).unsqueeze(-1)*.5
            n = depth.shape[1]
            img = torch.zeros(1, 2*n, 3)
            actual = options["patches"]["post_input"][0]({"img": img})["img"]
            self.assertTrue(torch.equal(actual[:, :n], depth.expand(-1, -1, 3)))
            self.assertFalse(actual[:, n:].any())

    def test_vision_rejects_text_only_tokenizer_output(self):
        from usdu_pose_tiled import krea2_vision

        clip = types.SimpleNamespace(tokenize=lambda *a, **kw: {"qwen3vl_4b": [[(123, 1.)]]})
        with self.assertRaisesRegex(ValueError, "image tokens"):
            krea2_vision._encode_tile(clip, "{}", "", self.image, 1)


if __name__ == "__main__":
    unittest.main()
