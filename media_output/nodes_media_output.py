"""No-save output transports for native ComfyUI media values."""

from __future__ import annotations

import io
import math
import os
import re
import shutil
import uuid
from pathlib import Path

import folder_paths


_VIDEO_SUFFIXES = {
    ".avi",
    ".gif",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".webm",
}
_RECEIPT_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")


def _normalized_video_suffix(source: object) -> str:
    name = source if isinstance(source, (str, os.PathLike)) else getattr(source, "name", "")
    suffix = Path(str(name or "")).suffix.lower()
    return suffix if suffix in _VIDEO_SUFFIXES else ".mp4"


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _materialize_temp_source(source: str | os.PathLike[str] | io.BytesIO) -> Path:
    temp_dir = Path(folder_paths.get_temp_directory()).expanduser().resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(source, (str, os.PathLike)):
        source_path = Path(source).expanduser().resolve(strict=True)
        if not source_path.is_file():
            raise ValueError(f"VIDEO stream source is not a file: {source_path}")
        if _is_within(source_path, temp_dir):
            return source_path

        target = temp_dir / f"sweet_tea_preview_{uuid.uuid4().hex}{_normalized_video_suffix(source_path)}"
        shutil.copyfile(source_path, target)
        return target

    if isinstance(source, io.BytesIO):
        target = temp_dir / f"sweet_tea_preview_{uuid.uuid4().hex}{_normalized_video_suffix(source)}"
        original_position = source.tell()
        try:
            source.seek(0)
            with target.open("xb") as handle:
                shutil.copyfileobj(source, handle)
        finally:
            source.seek(original_position)
        return target

    raise TypeError(
        "SweetTeaPreviewVideo requires VIDEO.get_stream_source() to return a file path or BytesIO"
    )


class SweetTeaPreviewVideo:
    """Expose a native VIDEO through Comfy's temp preview contract without saving."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "Sweet Tea/Output"
    DESCRIPTION = (
        "Publishes a native VIDEO as a temporary preview. Existing temp-backed "
        "videos are exposed in place and are never decoded or re-encoded."
    )

    def preview(self, video):
        get_stream_source = getattr(video, "get_stream_source", None)
        if not callable(get_stream_source):
            raise TypeError("SweetTeaPreviewVideo requires a native ComfyUI VIDEO input")

        source_path = _materialize_temp_source(get_stream_source())
        temp_dir = Path(folder_paths.get_temp_directory()).expanduser().resolve()
        relative_path = source_path.relative_to(temp_dir)
        subfolder = "" if relative_path.parent == Path(".") else relative_path.parent.as_posix()

        return {
            "ui": {
                # Comfy's public PreviewVideo UI output currently serializes video
                # descriptors under `images` and marks the result animated.
                "images": [
                    {
                        "filename": relative_path.name,
                        "subfolder": subfolder,
                        "type": "temp",
                    }
                ],
                "animated": (True,),
            }
        }


class SweetTeaExecutionReceipt:
    """Publish curated provider execution facts into Comfy history."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (
                    "STRING",
                    {
                        "default": "fal",
                        "tooltip": "Stable provider id, for example fal.",
                    },
                ),
                "request_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Provider-issued request identifier.",
                    },
                ),
                "endpoint_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Provider endpoint or model identifier.",
                    },
                ),
                "operation": (
                    [
                        "image-generation",
                        "video-generation",
                        "api-execution",
                    ],
                    {"default": "api-execution"},
                ),
            },
            "optional": {
                "estimated_cost_usd": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1_000_000.0,
                        "step": 0.001,
                        "tooltip": "Optional provider estimate; zero means unavailable.",
                    },
                ),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "publish"
    OUTPUT_NODE = True
    CATEGORY = "Sweet Tea/Output"
    DESCRIPTION = (
        "Publishes an allowlisted external API execution receipt for Sweet Tea "
        "Studio metadata. It does not save media."
    )

    def publish(
        self,
        provider: str,
        request_id: str,
        endpoint_id: str,
        operation: str,
        estimated_cost_usd: float = 0.0,
    ):
        provider_id = str(provider or "").strip().lower()
        request = str(request_id or "").strip()
        operation_id = str(operation or "").strip()
        endpoint = str(endpoint_id or "").strip()
        if not _RECEIPT_ID_RE.fullmatch(provider_id):
            raise ValueError("SweetTeaExecutionReceipt received an invalid provider id")
        if not _RECEIPT_ID_RE.fullmatch(request):
            raise ValueError("SweetTeaExecutionReceipt requires a valid provider request id")
        if not _RECEIPT_ID_RE.fullmatch(operation_id):
            raise ValueError("SweetTeaExecutionReceipt received an invalid operation id")
        if not endpoint or len(endpoint) > 512 or any(ord(char) < 32 for char in endpoint):
            raise ValueError("SweetTeaExecutionReceipt requires a valid endpoint id")

        payload = {
            "provider": provider_id,
            "request_id": request,
            "endpoint_id": endpoint,
            "operation": operation_id,
        }
        estimate = float(estimated_cost_usd)
        if math.isfinite(estimate) and estimate > 0:
            payload["estimated_cost_usd"] = estimate
        return {"ui": {"sweet_tea_execution_receipt": [payload]}}
