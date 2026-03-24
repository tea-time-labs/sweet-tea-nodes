import contextlib
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

from PIL import Image


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "usdu_pose_tiled"


def _load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_dependency_stubs():
    torch_mod = types.ModuleType("torch")
    torch_mod.no_grad = lambda: (lambda fn: fn)
    torch_mod.inference_mode = contextlib.nullcontext
    torch_mod.is_tensor = lambda _value: False
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
    )
    torch_mod.float16 = object()
    torch_mod.bfloat16 = object()

    torch_nn_mod = types.ModuleType("torch.nn")
    torch_nn_f_mod = types.ModuleType("torch.nn.functional")
    torch_nn_f_mod.interpolate = lambda *_args, **_kwargs: None
    torch_nn_mod.functional = torch_nn_f_mod
    torch_mod.nn = torch_nn_mod

    numpy_mod = types.ModuleType("numpy")

    class _DummyNdArray:
        pass

    numpy_mod.ndarray = _DummyNdArray

    tqdm_mod = types.ModuleType("tqdm")

    class _DummyProgressBar:
        def update(self, *_args, **_kwargs):
            return None

        def close(self):
            return None

    tqdm_mod.tqdm = lambda *_args, **_kwargs: _DummyProgressBar()

    comfy_mod = types.ModuleType("comfy")
    comfy_mod.model_management = types.SimpleNamespace(
        cleanup_models=lambda: None,
        cleanup_models_gc=lambda: None,
        soft_empty_cache=lambda: None,
        get_torch_device=lambda: "cpu",
    )

    nodes_mod = types.ModuleType("nodes")
    nodes_mod.common_ksampler = lambda **kwargs: {"samples": kwargs.get("latent", {}).get("samples")}

    pkg_mod = types.ModuleType("usdu_pose_tiled")
    pkg_mod.__path__ = [str(PKG_DIR)]

    return {
        "torch": torch_mod,
        "torch.nn": torch_nn_mod,
        "torch.nn.functional": torch_nn_f_mod,
        "numpy": numpy_mod,
        "tqdm": tqdm_mod,
        "comfy": comfy_mod,
        "nodes": nodes_mod,
        "usdu_pose_tiled": pkg_mod,
    }


class _DummyControlNet:
    def __init__(self, cond_hint_original, previous_controlnet=None, extra_concat_orig=()):
        self.cond_hint_original = cond_hint_original
        self.previous_controlnet = previous_controlnet
        self.extra_concat_orig = extra_concat_orig
        self.cond_hint = "cached-cond"
        self.extra_concat = "cached-extra"
        self.timestep_range = (0, 1)

    def copy(self):
        clone = _DummyControlNet(
            cond_hint_original=self.cond_hint_original,
            previous_controlnet=self.previous_controlnet,
            extra_concat_orig=self.extra_concat_orig,
        )
        clone.cond_hint = self.cond_hint
        clone.extra_concat = self.extra_concat
        clone.timestep_range = self.timestep_range
        return clone

    def set_previous_controlnet(self, previous):
        self.previous_controlnet = previous

    def get_models(self):
        return []


class PoseTilingCharacterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._module_patcher = mock.patch.dict(sys.modules, _build_dependency_stubs(), clear=False)
        cls._module_patcher.start()
        sys.modules.pop("usdu_pose_tiled.utils", None)
        sys.modules.pop("usdu_pose_tiled.processing_pose_tiled", None)
        sys.modules.pop("usdu_pose_tiled.usdu_pose_tiled", None)
        cls.utils = _load_module("usdu_pose_tiled.utils", PKG_DIR / "utils.py")
        cls.processing = _load_module(
            "usdu_pose_tiled.processing_pose_tiled",
            PKG_DIR / "processing_pose_tiled.py",
        )
        cls.usdu = _load_module(
            "usdu_pose_tiled.usdu_pose_tiled",
            PKG_DIR / "usdu_pose_tiled.py",
        )

    @classmethod
    def tearDownClass(cls):
        cls._module_patcher.stop()
        sys.modules.pop("usdu_pose_tiled.utils", None)
        sys.modules.pop("usdu_pose_tiled.processing_pose_tiled", None)
        sys.modules.pop("usdu_pose_tiled.usdu_pose_tiled", None)

    def test_transient_retry_retries_index_error_then_succeeds(self):
        attempts = {"count": 0}

        def flaky():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise IndexError("list index out of range")
            return "ok"

        with mock.patch.object(self.processing, "_recover_model_management_state") as recover_mock, mock.patch.object(
            self.processing.time,
            "sleep",
        ) as sleep_mock:
            result = self.processing._run_with_transient_model_retry(flaky, "vae.encode(tile)")

        self.assertEqual(result, "ok")
        self.assertEqual(attempts["count"], 3)
        self.assertEqual(recover_mock.call_count, 2)
        self.assertEqual(
            sleep_mock.call_args_list,
            [
                mock.call(self.processing._TRANSIENT_MODEL_BACKOFF_SECONDS * 1),
                mock.call(self.processing._TRANSIENT_MODEL_BACKOFF_SECONDS * 2),
            ],
        )

    def test_transient_retry_does_not_swallow_non_transient_errors(self):
        with mock.patch.object(self.processing, "_recover_model_management_state") as recover_mock, mock.patch.object(
            self.processing.time,
            "sleep",
        ) as sleep_mock:
            with self.assertRaisesRegex(RuntimeError, "boom"):
                self.processing._run_with_transient_model_retry(
                    lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                    "vae.decode(tile)",
                )

        recover_mock.assert_not_called()
        sleep_mock.assert_not_called()

    def test_crop_control_recursive_clones_controlnet_and_resets_cached_fields(self):
        previous = _DummyControlNet("spatial:previous")
        root = _DummyControlNet(
            "spatial:root",
            previous_controlnet=previous,
            extra_concat_orig=("spatial:concat", "keep"),
        )

        with mock.patch.object(self.utils, "_looks_spatial", side_effect=lambda v: isinstance(v, str) and v.startswith("spatial:")), mock.patch.object(
            self.utils,
            "_crop_spatial_value",
            side_effect=lambda value, *_args: f"cropped:{value}",
        ):
            cropped = self.utils._crop_control_recursive(
                root,
                (0, 0, 1, 1),
                (1, 1),
                (1, 1),
            )

        self.assertIsNot(cropped, root)
        self.assertEqual(cropped.cond_hint_original, "cropped:spatial:root")
        self.assertIsNot(cropped.previous_controlnet, previous)
        self.assertEqual(
            cropped.previous_controlnet.cond_hint_original,
            "cropped:spatial:previous",
        )
        self.assertIsInstance(cropped.extra_concat_orig, tuple)
        self.assertEqual(cropped.extra_concat_orig, ("cropped:spatial:concat", "keep"))
        self.assertIsNone(cropped.cond_hint)
        self.assertIsNone(cropped.extra_concat)
        self.assertIsNone(cropped.timestep_range)
        self.assertEqual(root.cond_hint_original, "spatial:root")
        self.assertEqual(root.cond_hint, "cached-cond")

    def test_crop_control_recursive_preserves_tuple_container_type(self):
        with mock.patch.object(self.utils, "_looks_spatial", side_effect=lambda v: isinstance(v, str) and v.startswith("spatial:")), mock.patch.object(
            self.utils,
            "_crop_spatial_value",
            side_effect=lambda value, *_args: f"cropped:{value}",
        ):
            result = self.utils._crop_control_recursive(
                ("spatial:item", "keep"),
                (0, 0, 1, 1),
                (1, 1),
                (1, 1),
            )

        self.assertIsInstance(result, tuple)
        self.assertEqual(result, ("cropped:spatial:item", "keep"))

    def test_linear_frontier_editable_mask_first_middle_and_last_tiles(self):
        redraw = self.usdu.USDURedraw(
            tile_width=10,
            tile_height=8,
            padding=2,
            mask_blur=0,
            mode=self.usdu.USDUMode.LINEAR.value,
            lock_padding=True,
        )

        first = redraw._build_linear_frontier_editable_mask(30, 24, 0, 0)
        self.assertEqual(first.getpixel((2, 2)), 255)
        self.assertEqual(first.getpixel((28, 20)), 255)

        middle = redraw._build_linear_frontier_editable_mask(30, 24, 1, 1)
        self.assertEqual(middle.getpixel((5, 4)), 0)
        self.assertEqual(middle.getpixel((25, 4)), 0)
        self.assertEqual(middle.getpixel((5, 12)), 0)
        self.assertEqual(middle.getpixel((15, 12)), 255)
        self.assertEqual(middle.getpixel((25, 12)), 255)
        self.assertEqual(middle.getpixel((5, 20)), 255)

        last = redraw._build_linear_frontier_editable_mask(30, 24, 2, 2)
        self.assertEqual(last.getpixel((5, 4)), 0)
        self.assertEqual(last.getpixel((15, 12)), 0)
        self.assertEqual(last.getpixel((25, 20)), 255)

    def test_linear_process_passes_directional_editable_masks(self):
        redraw = self.usdu.USDURedraw(
            tile_width=10,
            tile_height=10,
            padding=4,
            mask_blur=0,
            mode=self.usdu.USDUMode.LINEAR.value,
            lock_padding=True,
        )
        images = [Image.new("RGB", (20, 20), "black")]
        captured_masks = []

        def fake_run_single_masked_tile(processed_images, *_args, **kwargs):
            captured_masks.append(
                (
                    kwargs["editable_mask"].copy(),
                    _args[10].copy(),
                )
            )
            return processed_images

        with mock.patch.object(redraw, "_run_single_masked_tile", side_effect=fake_run_single_masked_tile):
            out = redraw.linear_process(
                model=None,
                positive=None,
                negative=None,
                vae=None,
                seed=0,
                steps=1,
                cfg=1.0,
                sampler_name="sampler",
                scheduler="scheduler",
                denoise=0.2,
                images=images,
                uniform_mode=False,
                tiled_decode=False,
            )

        self.assertEqual(len(out), 1)
        self.assertEqual(len(captured_masks), 4)

        first_editable, first_commit = captured_masks[0]
        self.assertEqual(first_editable.getpixel((15, 15)), 255)
        self.assertEqual(first_commit.getpixel((15, 15)), 0)

        second_editable, second_commit = captured_masks[1]
        self.assertEqual(second_editable.getpixel((5, 5)), 0)
        self.assertEqual(second_editable.getpixel((15, 5)), 255)
        self.assertEqual(second_editable.getpixel((5, 15)), 255)
        self.assertEqual(second_commit.getpixel((5, 5)), 0)
        self.assertEqual(second_commit.getpixel((15, 5)), 255)
        self.assertEqual(second_commit.getpixel((5, 15)), 0)

        last_editable, last_commit = captured_masks[-1]
        self.assertEqual(last_editable.getpixel((5, 5)), 0)
        self.assertEqual(last_editable.getpixel((15, 15)), 255)
        self.assertEqual(last_commit.getpixel((15, 15)), 255)

    def test_chess_process_keeps_symmetric_lock_behavior(self):
        redraw = self.usdu.USDURedraw(
            tile_width=10,
            tile_height=10,
            padding=4,
            mask_blur=0,
            mode=self.usdu.USDUMode.CHESS.value,
            lock_padding=True,
        )
        images = [Image.new("RGB", (20, 20), "black")]
        editable_masks = []

        def fake_run_single_masked_tile(processed_images, *_args, **kwargs):
            editable_masks.append(kwargs["editable_mask"])
            return processed_images

        with mock.patch.object(redraw, "_run_single_masked_tile", side_effect=fake_run_single_masked_tile):
            redraw.chess_process(
                model=None,
                positive=None,
                negative=None,
                vae=None,
                seed=0,
                steps=1,
                cfg=1.0,
                sampler_name="sampler",
                scheduler="scheduler",
                denoise=0.2,
                images=images,
                uniform_mode=False,
                tiled_decode=False,
            )

        self.assertTrue(editable_masks)
        self.assertTrue(all(mask is None for mask in editable_masks))

    def test_normalized_chess_process_keeps_symmetric_lock_behavior(self):
        redraw = self.usdu.USDURedraw(
            tile_width=10,
            tile_height=10,
            padding=4,
            mask_blur=8,
            mode=self.usdu.USDUMode.CHESS.value,
            lock_padding=True,
        )
        images = [Image.new("RGB", (20, 20), "black")]
        editable_masks = []

        def fake_run_single_masked_tile(processed_images, *_args, **kwargs):
            editable_masks.append(kwargs["editable_mask"])
            return [], None

        with mock.patch.object(self.usdu.np, "zeros", side_effect=lambda *_args, **_kwargs: None, create=True), mock.patch.object(
            self.usdu.np,
            "float32",
            new=object(),
            create=True,
        ), mock.patch.object(
            redraw,
            "_run_single_masked_tile",
            side_effect=fake_run_single_masked_tile,
        ):
            redraw.chess_process(
                model=None,
                positive=None,
                negative=None,
                vae=None,
                seed=0,
                steps=1,
                cfg=1.0,
                sampler_name="sampler",
                scheduler="scheduler",
                denoise=0.2,
                images=images,
                uniform_mode=False,
                tiled_decode=False,
            )

        self.assertTrue(editable_masks)
        self.assertTrue(all(mask is None for mask in editable_masks))

    def test_process_images_uses_commit_mask_for_crop_and_editable_mask_for_noise(self):
        commit_mask = Image.new("L", (4, 4), 0)
        commit_mask.putpixel((0, 0), 255)
        editable_mask = Image.new("L", (4, 4), 255)
        init_image = Image.new("RGB", (4, 4), "black")
        crop_masks = []
        noise_masks = []

        class _FakeLatents:
            shape = (1, 4, 1, 1)
            device = "cpu"

        class _FakeNoiseMask:
            shape = (1, 1, 1)

            def to(self, *_args, **_kwargs):
                return self

        fake_latents = _FakeLatents()
        fake_decoded = types.SimpleNamespace(shape=(1,))
        vae = types.SimpleNamespace(
            encode=lambda _bhwc: {"samples": fake_latents},
            decode=lambda _samples: fake_decoded,
        )

        def fake_get_crop_region(mask, _padding):
            crop_masks.append(mask.copy())
            return (0, 0, 2, 2)

        def fake_make_noise_mask(mask, *_args):
            noise_masks.append(mask.copy())
            return _FakeNoiseMask()

        with mock.patch.object(self.processing, "get_crop_region", side_effect=fake_get_crop_region), mock.patch.object(
            self.processing,
            "expand_crop",
            return_value=((0, 0, 2, 2), (2, 2)),
        ), mock.patch.object(
            self.processing,
            "crop_cond",
            side_effect=lambda cond, *_args: cond,
        ), mock.patch.object(
            self.processing,
            "_pil_list_to_bhwc",
            return_value="bhwc",
        ), mock.patch.object(
            self.processing,
            "_make_latent_noise_mask_from_tile_mask",
            side_effect=fake_make_noise_mask,
        ), mock.patch.object(
            self.processing,
            "_run_with_transient_model_retry",
            side_effect=lambda fn, _name: fn(),
        ), mock.patch.object(
            self.processing,
            "common_ksampler",
            return_value={"samples": fake_latents},
        ), mock.patch.object(
            self.processing,
            "tensor_to_pil",
            return_value=Image.new("RGB", (2, 2), "white"),
        ):
            result = self.processing.process_images_pose_tiled(
                model=None,
                positive="pos",
                negative="neg",
                vae=vae,
                seed=0,
                steps=1,
                cfg=1.0,
                sampler_name="sampler",
                scheduler="scheduler",
                init_images=[init_image],
                image_mask=commit_mask,
                tile_size=(2, 2),
                uniform_tile_mode=False,
                mask_blur=0,
                denoise=0.2,
                tile_padding=0,
                lock_padding=True,
                editable_mask=editable_mask,
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(crop_masks[0].getpixel((3, 3)), 0)
        self.assertEqual(noise_masks[0].getpixel((3, 3)), 255)


if __name__ == "__main__":
    unittest.main()
