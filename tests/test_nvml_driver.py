"""Driver-version discovery does not shell out and remains safe on CPU hosts."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("environment_metadata", Path(__file__).parents[1] / "environment_metadata.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NvmlDriverTests(unittest.TestCase):
    def test_nvml_version_and_shutdown(self):
        calls = []
        fake = types.SimpleNamespace(
            nvmlInit=lambda: calls.append("init"),
            nvmlSystemGetDriverVersion=lambda: b"570.172.08",
            nvmlShutdown=lambda: calls.append("shutdown"),
        )
        with patch.dict(sys.modules, {"pynvml": fake}):
            self.assertEqual(module._nvidia_driver_version(), "570.172.08")
        self.assertEqual(calls, ["init", "shutdown"])

    def test_nvml_failure_is_best_effort(self):
        def no_driver():
            raise RuntimeError("No NVIDIA GPU")
        fake = types.SimpleNamespace(nvmlInit=no_driver)
        with patch.dict(sys.modules, {"pynvml": fake}):
            self.assertIsNone(module._nvidia_driver_version())

    def test_query_failure_still_shuts_down(self):
        calls = []
        def query_failure():
            raise RuntimeError("NVML unavailable")
        fake = types.SimpleNamespace(
            nvmlInit=lambda: calls.append("init"),
            nvmlSystemGetDriverVersion=query_failure,
            nvmlShutdown=lambda: calls.append("shutdown"),
        )
        with patch.dict(sys.modules, {"pynvml": fake}):
            self.assertIsNone(module._nvidia_driver_version())
        self.assertEqual(calls, ["init", "shutdown"])


if __name__ == "__main__":
    unittest.main()
