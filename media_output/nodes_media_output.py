"""No-save output transports for native ComfyUI media values."""

from __future__ import annotations

import io
import math
import os
import re
import shutil
import uuid
from fractions import Fraction
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


class SweetTeaPreviewImage:
    """Stream full-resolution IMAGE values to the API client without writing them to disk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "Sweet Tea/Output"
    DESCRIPTION = (
        "Streams full-resolution IMAGE values over ComfyUI's websocket output "
        "transport so Sweet Tea Studio can persist them directly without a "
        "duplicate Comfy output file."
    )

    def preview(self, images):
        # Match ComfyUI's SaveImageWebsocket transport exactly: ProgressBar emits
        # full-resolution binary image frames to the connected API client and does
        # not materialize an output/temp image file.
        from PIL import Image
        import comfy.utils
        import numpy as np

        total = int(images.shape[0])
        progress = comfy.utils.ProgressBar(total)
        for index, image in enumerate(images):
            array = 255.0 * image.cpu().numpy()
            pil_image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            progress.update_absolute(index, total, ("PNG", pil_image, None))
        return {}

    @classmethod
    def IS_CHANGED(cls, images):
        import time

        return time.time()


def _resolve_video_container(format_name: str, codec_name: str) -> str:
    format_name = str(format_name or "auto").strip().lower()
    codec_name = str(codec_name or "auto").strip().lower()
    if format_name not in {"auto", "mp4", "mkv", "webm"}:
        raise ValueError(f"Unsupported video container: {format_name}")
    if codec_name not in {"auto", "h264", "h265", "nvenc_h264", "av1", "vp9"}:
        raise ValueError(f"Unsupported video codec: {codec_name}")
    if format_name == "auto":
        return "webm" if codec_name in {"av1", "vp9"} else "mp4"
    if format_name == "webm" and codec_name not in {"auto", "av1", "vp9"}:
        raise ValueError("WebM output requires AV1, VP9, or Auto codec")
    return format_name


def _preview_descriptor(source_path: Path) -> dict:
    temp_dir = Path(folder_paths.get_temp_directory()).expanduser().resolve()
    relative_path = source_path.relative_to(temp_dir)
    subfolder = "" if relative_path.parent == Path(".") else relative_path.parent.as_posix()
    return {
        "ui": {
            "images": [{"filename": relative_path.name, "subfolder": subfolder, "type": "temp"}],
            "animated": (True,),
        }
    }


def _encode_native_video_to_temp(video, format_name: str, codec_name: str, crf: float) -> Path:
    save_to = getattr(video, "save_to", None)
    if not callable(save_to):
        raise TypeError("Sweet Tea video output requires a native ComfyUI VIDEO input")
    container = _resolve_video_container(format_name, codec_name)
    temp_dir = Path(folder_paths.get_temp_directory()).expanduser().resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    target = temp_dir / f"sweet_tea_preview_{uuid.uuid4().hex}.{container}"
    save_to(
        str(target),
        format=str(format_name or "auto").strip().lower(),
        codec=str(codec_name or "auto").strip().lower(),
        crf=None if float(crf) < 0 else float(crf),
    )
    return target


class SweetTeaPreviewVideo:
    """Expose a native VIDEO through Comfy's temp preview contract without permanent saving."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
            },
            "optional": {
                "format": (["auto", "mp4", "mkv", "webm"], {"default": "auto"}),
                "codec": (["auto", "h264", "av1"], {"default": "auto"}),
                "crf": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 63.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "Sweet Tea/Output"
    DESCRIPTION = (
        "Publishes a native VIDEO as a temporary preview without a permanent Comfy output file. "
        "Auto/Auto preserves an existing encoded stream; explicit container or codec settings "
        "materialize the requested output in Comfy's temp directory. CRF -1 uses the encoder default."
    )

    def preview(self, video, format="auto", codec="auto", crf=-1.0):
        format_name = str(format or "auto").strip().lower()
        codec_name = str(codec or "auto").strip().lower()
        if format_name == "auto" and codec_name == "auto" and float(crf) < 0:
            get_stream_source = getattr(video, "get_stream_source", None)
            if not callable(get_stream_source):
                raise TypeError("SweetTeaPreviewVideo requires a native ComfyUI VIDEO input")
            source_path = _materialize_temp_source(get_stream_source())
        else:
            source_path = _encode_native_video_to_temp(video, format_name, codec_name, crf)
        return _preview_descriptor(source_path)


_VIDEO_ENCODERS = {
    "h264": "h264",
    "h265": "libx265",
    "nvenc_h264": "h264_nvenc",
    "av1": "libsvtav1",
    "vp9": "libvpx-vp9",
}
_VIDEO_CONTAINER_FORMATS = {"mp4": "mp4", "mkv": "matroska", "webm": "webm"}


def _encode_image_sequence_custom(
    images,
    frame_rate: float,
    format_name: str,
    codec_name: str,
    crf: float,
    bit_depth: int,
    audio=None,
    pixel_format: str = "auto",
    bitrate_mbps: float = 0.0,
) -> Path:
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Sweet Tea video encoding requires ComfyUI's PyAV and NumPy runtime") from exc

    container = _resolve_video_container(format_name, codec_name)
    if codec_name not in _VIDEO_ENCODERS:
        raise ValueError(f"Unsupported custom video codec: {codec_name}")
    if container == "webm" and codec_name not in {"av1", "vp9"}:
        raise ValueError("Custom WebM encoding requires AV1 or VP9")

    temp_dir = Path(folder_paths.get_temp_directory()).expanduser().resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    target = temp_dir / f"sweet_tea_preview_{uuid.uuid4().hex}.{container}"
    fps = Fraction(round(float(frame_rate) * 1000), 1000)
    pix_fmt = str(pixel_format or "auto").strip().lower()
    if pix_fmt == "auto":
        pix_fmt = "yuv420p10le" if int(bit_depth) >= 10 else "yuv420p"

    with av.open(str(target), mode="w", format=_VIDEO_CONTAINER_FORMATS[container]) as output:
        video_stream = output.add_stream(_VIDEO_ENCODERS[codec_name], rate=fps)
        video_stream.width = int(images.shape[2])
        video_stream.height = int(images.shape[1])
        video_stream.pix_fmt = pix_fmt
        options = {}
        if float(crf) >= 0:
            options["crf"] = str(float(crf))
        if options:
            video_stream.options = options
        if float(bitrate_mbps) > 0:
            video_stream.bit_rate = int(float(bitrate_mbps) * 1_000_000)

        for tensor in images:
            if int(bit_depth) >= 10:
                array = (tensor[..., :3].float() * 65535).clamp(0, 65535).cpu().numpy().astype(np.uint16)
                frame = av.VideoFrame.from_ndarray(array, format="rgb48le")
            else:
                array = (tensor[..., :3] * 255).clamp(0, 255).byte().cpu().numpy()
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame = frame.reformat(format=pix_fmt)
            output.mux(video_stream.encode(frame))
        output.mux(video_stream.encode(None))

        if audio:
            sample_rate = int(audio["sample_rate"])
            waveform = audio["waveform"][0, :, : math.ceil((sample_rate / fps) * int(images.shape[0]))]
            layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(int(waveform.shape[0]), "stereo")
            target_rate = 48000 if container == "webm" else sample_rate
            audio_codec = "libopus" if container == "webm" else "aac"
            audio_stream = output.add_stream(audio_codec, rate=target_rate, layout=layout)
            resampler = None
            if target_rate != sample_rate:
                resampler = av.audio.resampler.AudioResampler(format="fltp", layout=layout, rate=target_rate)
            audio_frame = av.AudioFrame.from_ndarray(
                waveform.float().cpu().contiguous().numpy(), format="fltp", layout=layout
            )
            audio_frame.sample_rate = sample_rate
            audio_frame.pts = 0
            frames = [audio_frame] if resampler is None else resampler.resample(audio_frame)
            for frame in frames:
                output.mux(audio_stream.encode(frame))
            if resampler is not None:
                for frame in resampler.resample(None):
                    output.mux(audio_stream.encode(frame))
            output.mux(audio_stream.encode(None))

    return target


class SweetTeaPreviewVideoFromImages:
    """Encode IMAGE frames once into a temporary video and publish it to Sweet Tea Studio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "format": (["auto", "mp4", "mkv", "webm"], {"default": "mp4"}),
                "codec": (["auto", "h264", "h265", "nvenc_h264", "av1", "vp9"], {"default": "auto"}),
                "crf": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100.0, "step": 1.0}),
                "bitrate_mbps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1}),
                "pixel_format": (["auto", "yuv420p", "yuv420p10le", "yuv444p", "yuv444p10le"], {"default": "auto"}),
                "bit_depth": (["auto", "8", "10"], {"default": "auto"}),
                "color_space": (["sRGB", "HDR", "HDR PQ"], {"default": "sRGB"}),
            },
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "Sweet Tea/Output"
    DESCRIPTION = (
        "Encodes IMAGE frames into one temporary video for Sweet Tea Studio without writing a duplicate "
        "Comfy output file. Frame rate is required because IMAGE batches do not carry timing. Container, "
        "codec, quality, bit depth, color space, and optional audio are preserved in the temporary output."
    )

    def preview(
        self,
        images,
        frame_rate=24.0,
        format="mp4",
        codec="auto",
        crf=-1.0,
        bitrate_mbps=0.0,
        pixel_format="auto",
        bit_depth="auto",
        color_space="sRGB",
        audio=None,
    ):
        try:
            from comfy_api.latest import InputImpl, Types
        except ImportError as exc:
            raise RuntimeError(
                "SweetTeaPreviewVideoFromImages requires a ComfyUI build with native VIDEO support"
            ) from exc

        fps = float(frame_rate)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("frame_rate must be a positive finite number")
        depth = (
            10
            if bit_depth == "auto" and color_space in {"HDR", "HDR PQ"}
            else 8
            if bit_depth == "auto"
            else int(bit_depth)
        )
        codec_name = str(codec or "auto").strip().lower()
        format_name = str(format or "mp4").strip().lower()
        pixel_format_name = str(pixel_format or "auto").strip().lower()
        use_native_encoder = codec_name in {"auto", "h264", "av1"} and pixel_format_name == "auto" and float(bitrate_mbps) <= 0
        if use_native_encoder:
            components = Types.VideoComponents(
                images=images,
                audio=audio,
                frame_rate=Fraction(round(fps * 1000), 1000),
            )
            video = InputImpl.VideoFromComponents(components, bit_depth=depth, color_space=color_space)
            source_path = _encode_native_video_to_temp(video, format_name, codec_name, crf)
        else:
            if color_space != "sRGB":
                raise ValueError("Custom H.265/NVENC/VP9 output currently supports sRGB only")
            source_path = _encode_image_sequence_custom(
                images, fps, format_name, codec_name, crf, depth, audio, pixel_format_name, bitrate_mbps
            )
        return _preview_descriptor(source_path)


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
