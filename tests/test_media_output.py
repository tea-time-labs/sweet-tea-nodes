from __future__ import annotations

import importlib.util
import io
import sys
import types
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "media_output"
    / "nodes_media_output.py"
)


def _load_module(monkeypatch, temp_dir: Path):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_temp_directory = lambda: str(temp_dir)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    spec = importlib.util.spec_from_file_location(
        f"sweet_tea_media_output_test_{id(temp_dir)}",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Video:
    def __init__(self, source):
        self.source = source

    def get_stream_source(self):
        return self.source


def _descriptor(result: dict) -> dict:
    assert result["ui"]["animated"] == (True,)
    [descriptor] = result["ui"]["images"]
    return descriptor


def test_temp_backed_video_is_exposed_in_place(monkeypatch, tmp_path):
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    source = temp_dir / "civitai-native.mp4"
    source.write_bytes(b"native-video-with-audio")
    module = _load_module(monkeypatch, temp_dir)

    result = module.SweetTeaPreviewVideo().preview(_Video(str(source)))

    assert _descriptor(result) == {
        "filename": source.name,
        "subfolder": "",
        "type": "temp",
    }
    assert list(temp_dir.iterdir()) == [source]
    assert source.read_bytes() == b"native-video-with-audio"


def test_unserved_video_sources_are_byte_preserved_in_temp(monkeypatch, tmp_path):
    temp_dir = tmp_path / "temp"
    source_dir = tmp_path / "provider"
    temp_dir.mkdir()
    source_dir.mkdir()
    source = source_dir / "provider.webm"
    payload = b"provider-container-bytes"
    source.write_bytes(payload)
    module = _load_module(monkeypatch, temp_dir)

    path_result = module.SweetTeaPreviewVideo().preview(_Video(source))
    stream = io.BytesIO(payload)
    stream.name = "provider.mp4"
    stream_result = module.SweetTeaPreviewVideo().preview(_Video(stream))

    for result in (path_result, stream_result):
        descriptor = _descriptor(result)
        assert descriptor["type"] == "temp"
        assert (temp_dir / descriptor["filename"]).read_bytes() == payload
    assert source.read_bytes() == payload


def test_execution_receipt_publishes_only_curated_provider_facts(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path / "temp")

    result = module.SweetTeaExecutionReceipt().publish(
        provider="fal",
        request_id="req-42",
        endpoint_id="fal-ai/flux/dev",
        operation="image-generation",
        estimated_cost_usd=0.025,
    )

    assert result == {
        "ui": {
            "sweet_tea_execution_receipt": [
                {
                    "provider": "fal",
                    "request_id": "req-42",
                    "endpoint_id": "fal-ai/flux/dev",
                    "operation": "image-generation",
                    "estimated_cost_usd": 0.025,
                }
            ]
        }
    }
