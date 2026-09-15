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


class _EncodingVideo:
    def __init__(self):
        self.calls = []

    def save_to(self, path, **kwargs):
        self.calls.append((path, kwargs))
        Path(path).write_bytes(b"encoded-video")


def test_native_video_explicit_format_encodes_once_to_temp(monkeypatch, tmp_path):
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    module = _load_module(monkeypatch, temp_dir)
    video = _EncodingVideo()

    result = module.SweetTeaPreviewVideo().preview(video, format="mkv", codec="h264", crf=19)

    assert len(video.calls) == 1
    path, kwargs = video.calls[0]
    assert Path(path).parent == temp_dir
    assert Path(path).suffix == ".mkv"
    assert kwargs == {"format": "mkv", "codec": "h264", "crf": 19.0}
    assert _descriptor(result)["type"] == "temp"


def test_image_sequence_video_output_preserves_fps_and_encoding_choices(monkeypatch, tmp_path):
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    module = _load_module(monkeypatch, temp_dir)
    created = {}

    class _Components:
        def __init__(self, **kwargs):
            created["components"] = kwargs

    class _VideoFromComponents(_EncodingVideo):
        def __init__(self, components, bit_depth, color_space):
            super().__init__()
            created["video"] = self
            created["bit_depth"] = bit_depth
            created["color_space"] = color_space

    latest = types.ModuleType("comfy_api.latest")
    latest.Types = types.SimpleNamespace(VideoComponents=_Components)
    latest.InputImpl = types.SimpleNamespace(VideoFromComponents=_VideoFromComponents)
    comfy_api = types.ModuleType("comfy_api")
    comfy_api.latest = latest
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", latest)

    images = object()
    audio = object()
    result = module.SweetTeaPreviewVideoFromImages().preview(
        images,
        frame_rate=23.976,
        format="webm",
        codec="av1",
        crf=28,
        bit_depth="10",
        color_space="HDR",
        audio=audio,
    )

    assert created["components"]["images"] is images
    assert created["components"]["audio"] is audio
    assert float(created["components"]["frame_rate"]) == 23.976
    assert created["bit_depth"] == 10
    assert created["color_space"] == "HDR"
    video = created["video"]
    assert len(video.calls) == 1
    path, kwargs = video.calls[0]
    assert Path(path).suffix == ".webm"
    assert kwargs == {"format": "webm", "codec": "av1", "crf": 28.0}
    assert _descriptor(result)["filename"].endswith(".webm")


def test_image_sequence_output_contract_covers_common_workflow_encoders(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path / "temp")
    input_types = module.SweetTeaPreviewVideoFromImages.INPUT_TYPES()
    required = input_types["required"]
    assert required["frame_rate"][0] == "FLOAT"
    assert {"h264", "h265", "nvenc_h264", "av1", "vp9"}.issubset(set(required["codec"][0]))
    assert {"mp4", "mkv", "webm"}.issubset(set(required["format"][0]))
    assert "yuv420p" in required["pixel_format"][0]
    assert "yuv420p10le" in required["pixel_format"][0]
    assert module._resolve_video_container("auto", "h265") == "mp4"
    assert module._resolve_video_container("auto", "vp9") == "webm"
    assert module._VIDEO_ENCODERS["h264"] == "h264"
    assert module._VIDEO_ENCODERS["av1"] == "libsvtav1"
