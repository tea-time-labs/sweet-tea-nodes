import contextlib
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock


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
        cls.utils = _load_module("usdu_pose_tiled.utils", PKG_DIR / "utils.py")
        cls.processing = _load_module(
            "usdu_pose_tiled.processing_pose_tiled",
            PKG_DIR / "processing_pose_tiled.py",
        )

    @classmethod
    def tearDownClass(cls):
        cls._module_patcher.stop()
        sys.modules.pop("usdu_pose_tiled.utils", None)
        sys.modules.pop("usdu_pose_tiled.processing_pose_tiled", None)

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


if __name__ == "__main__":
    unittest.main()
