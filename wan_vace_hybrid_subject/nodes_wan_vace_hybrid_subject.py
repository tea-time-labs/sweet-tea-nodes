from __future__ import annotations

import torch
import torch.nn.functional as F

import comfy
import comfy.latent_formats
import comfy.model_management
import comfy.utils
import node_helpers
import nodes


_NEUTRAL_RGB = 0.5
_LUMA_WEIGHTS = (0.299, 0.587, 0.114)


def _extract_encoded_latent(vae_out, tag: str) -> torch.Tensor:
    if isinstance(vae_out, dict):
        vae_out = vae_out.get("samples", next((v for v in vae_out.values() if torch.is_tensor(v)), None))
    if not torch.is_tensor(vae_out):
        raise ValueError(f"{tag}: VAE.encode returned non-tensor type {type(vae_out)}")
    if vae_out.dim() == 4:
        vae_out = vae_out.unsqueeze(2)
    if vae_out.dim() != 5:
        raise ValueError(f"{tag}: expected rank-5 latent, got rank {vae_out.dim()} shape={tuple(vae_out.shape)}")
    return vae_out.contiguous()


def _vae_encode_5d(vae, x_nhwc: torch.Tensor, tag: str) -> torch.Tensor:
    return _extract_encoded_latent(vae.encode(x_nhwc[..., :3]), tag)


def _ensure_nhwc_frames(x: torch.Tensor, tag: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise ValueError(f"{tag}: expected torch.Tensor, got {type(x)}")
    while x.dim() > 4:
        x = x[0]
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4:
        raise ValueError(f"{tag}: expected rank-4 NHWC tensor, got rank {x.dim()} shape={tuple(x.shape)}")
    return x.to(dtype=torch.float32)


def _ensure_rgb_channels(x_nhwc: torch.Tensor, tag: str) -> torch.Tensor:
    channels = x_nhwc.shape[-1]
    if channels == 3:
        return x_nhwc
    if channels == 1:
        return x_nhwc.repeat(1, 1, 1, 3)
    if channels == 2:
        return torch.cat([x_nhwc, x_nhwc[..., :1]], dim=-1)
    if channels > 3:
        return x_nhwc[..., :3]
    raise ValueError(f"{tag}: expected at least one channel, got shape {tuple(x_nhwc.shape)}")


def _upscale_nhwc(x_nhwc: torch.Tensor, width: int, height: int) -> torch.Tensor:
    return comfy.utils.common_upscale(
        x_nhwc.movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1).to(dtype=torch.float32)


def _prepare_video_nhwc(
    video,
    width: int,
    height: int,
    length: int,
    device,
    *,
    fill_value: float = _NEUTRAL_RGB,
    tag: str,
) -> torch.Tensor:
    if video is None:
        return torch.full((length, height, width, 3), fill_value, device=device, dtype=torch.float32)

    video = _ensure_rgb_channels(_ensure_nhwc_frames(video, tag), tag)
    video = _upscale_nhwc(video[:length], width, height)
    if video.shape[0] < length:
        video = F.pad(
            video,
            (0, 0, 0, 0, 0, 0, 0, length - video.shape[0]),
            value=fill_value,
        )
    return video.to(device=device, dtype=torch.float32).contiguous()


def _prepare_reference_image_nhwc(reference_image, width: int, height: int, device) -> torch.Tensor | None:
    if reference_image is None:
        return None
    reference_image = _ensure_rgb_channels(_ensure_nhwc_frames(reference_image, "reference_image"), "reference_image")[:1]
    reference_image = _upscale_nhwc(reference_image, width, height)
    return reference_image.to(device=device, dtype=torch.float32).contiguous()


def _prepare_mask_nhwc(
    control_masks,
    width: int,
    height: int,
    length: int,
    device,
    *,
    default_value: float,
    feather_px: float,
) -> torch.Tensor:
    if control_masks is None:
        mask = torch.full((length, height, width, 1), default_value, device=device, dtype=torch.float32)
    else:
        mask = control_masks
        while mask.dim() > 4:
            mask = mask[0]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            if mask.shape[-1] == 1:
                mask = mask.movedim(-1, 1)
            else:
                mask = mask.unsqueeze(1)
        elif mask.dim() == 4:
            if mask.shape[-1] == 1 and mask.shape[1] != 1:
                mask = mask.movedim(-1, 1)
            elif mask.shape[1] != 1:
                raise ValueError(f"control_masks: expected a single mask channel, got shape {tuple(mask.shape)}")
        else:
            raise ValueError(f"control_masks: unsupported rank {mask.dim()} with shape {tuple(mask.shape)}")

        mask = comfy.utils.common_upscale(mask[:length], width, height, "bilinear", "center").movedim(1, -1)
        if mask.shape[0] < length:
            mask = F.pad(
                mask,
                (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]),
                value=default_value,
            )
        mask = mask.to(device=device, dtype=torch.float32)

    mask = mask.clamp(0.0, 1.0)
    radius = max(int(round(float(feather_px))), 0)
    if radius > 0:
        kernel = (radius * 2) + 1
        mask = F.avg_pool2d(mask.movedim(-1, 1), kernel_size=kernel, stride=1, padding=radius).movedim(1, -1)
    return mask.clamp(0.0, 1.0).contiguous()


def _mask_region(content_nhwc: torch.Tensor, region_mask_nhwc: torch.Tensor) -> torch.Tensor:
    return _NEUTRAL_RGB + (content_nhwc - _NEUTRAL_RGB) * region_mask_nhwc


def _rgb_to_luma_rgb(x_nhwc: torch.Tensor) -> torch.Tensor:
    weights = x_nhwc.new_tensor(_LUMA_WEIGHTS).view(1, 1, 1, 3)
    luma = (x_nhwc * weights).sum(dim=-1, keepdim=True)
    return luma.repeat(1, 1, 1, 3)


def _rgb_to_detail_rgb(x_nhwc: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    luma = _rgb_to_luma_rgb(x_nhwc)[..., :1]
    blur = F.avg_pool2d(luma.movedim(-1, 1), kernel_size=kernel_size, stride=1, padding=kernel_size // 2).movedim(1, -1)
    detail = luma - blur
    detail = detail / detail.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
    detail = (detail * 0.25) + _NEUTRAL_RGB
    return detail.clamp(0.0, 1.0).repeat(1, 1, 1, 3)


def _scale_around_neutral(x_nhwc: torch.Tensor, weight: float) -> torch.Tensor:
    if weight <= 0.0:
        return torch.full_like(x_nhwc, _NEUTRAL_RGB)
    return (_NEUTRAL_RGB + ((x_nhwc - _NEUTRAL_RGB) * float(weight))).clamp(0.0, 1.0).contiguous()


def _compose_composite_guidance_video(
    structure_video: torch.Tensor,
    source_rgb_video: torch.Tensor | None,
    subject_mask: torch.Tensor,
    background_region: torch.Tensor,
    *,
    control_weight: float,
    source_luma_weight: float,
    source_detail_weight: float,
    background_weight: float,
) -> torch.Tensor:
    composite = _scale_around_neutral(structure_video, control_weight)

    if source_rgb_video is None:
        return composite

    neutral = torch.full_like(source_rgb_video, _NEUTRAL_RGB)
    if source_luma_weight > 0.0:
        composite = composite + ((_rgb_to_luma_rgb(source_rgb_video) - neutral) * source_luma_weight * subject_mask)

    if source_detail_weight > 0.0:
        composite = composite + ((_rgb_to_detail_rgb(source_rgb_video) - neutral) * source_detail_weight * subject_mask)

    if background_weight > 0.0:
        composite = composite + ((source_rgb_video - composite) * background_weight * background_region)

    return composite.clamp(0.0, 1.0).contiguous()


def _build_vace_mask(mask_nhwc: torch.Tensor, latent_length: int) -> torch.Tensor:
    length, height, width, _ = mask_nhwc.shape
    vae_stride = 8
    height_mask = height // vae_stride
    width_mask = width // vae_stride

    mask_latent = mask_nhwc.view(length, height_mask, vae_stride, width_mask, vae_stride)
    mask_latent = mask_latent.permute(2, 4, 0, 1, 3)
    mask_latent = mask_latent.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
    if mask_latent.shape[1] != latent_length:
        mask_latent = F.interpolate(
            mask_latent.unsqueeze(0),
            size=(latent_length, height_mask, width_mask),
            mode="nearest-exact",
        ).squeeze(0)
    return mask_latent.unsqueeze(0).contiguous()


def _encode_vace_context(
    vae,
    *,
    control_video_nhwc: torch.Tensor,
    control_mask_nhwc: torch.Tensor,
    reference_image_nhwc: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    control_video_centered = control_video_nhwc - _NEUTRAL_RGB
    inactive = (control_video_centered * (1.0 - control_mask_nhwc)) + _NEUTRAL_RGB
    reactive = (control_video_centered * control_mask_nhwc) + _NEUTRAL_RGB

    inactive_latent = _vae_encode_5d(vae, inactive, "inactive")
    reactive_latent = _vae_encode_5d(vae, reactive, "reactive")
    if inactive_latent.shape != reactive_latent.shape:
        raise ValueError(
            f"Inactive/reactive latent shapes differ: {tuple(inactive_latent.shape)} vs {tuple(reactive_latent.shape)}"
        )

    frames_2c = torch.cat([inactive_latent, reactive_latent], dim=1)
    mask_latent = _build_vace_mask(control_mask_nhwc, frames_2c.shape[2])
    trim_latent = 0

    if reference_image_nhwc is not None:
        ref_latent = _vae_encode_5d(vae, reference_image_nhwc, "reference_image")
        zero_half = comfy.latent_formats.Wan21().process_out(torch.zeros_like(ref_latent))
        ref_2c = torch.cat([ref_latent, zero_half], dim=1)
        frames_2c = torch.cat([ref_2c, frames_2c], dim=2)
        mask_pad = torch.zeros(
            (mask_latent.shape[0], mask_latent.shape[1], ref_2c.shape[2], mask_latent.shape[3], mask_latent.shape[4]),
            device=mask_latent.device,
            dtype=mask_latent.dtype,
        )
        mask_latent = torch.cat([mask_pad, mask_latent], dim=2)
        trim_latent = ref_2c.shape[2]

    return frames_2c.contiguous(), mask_latent.contiguous(), trim_latent


def _reconcile_context_lengths(
    vace_frames_list: list[torch.Tensor],
    vace_mask_list: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
    max_frames = max(frames.shape[2] for frames in vace_frames_list)
    out_frames = []
    out_masks = []

    for frames, mask in zip(vace_frames_list, vace_mask_list):
        pad_frames = max_frames - frames.shape[2]
        if pad_frames <= 0:
            out_frames.append(frames.contiguous())
            out_masks.append(mask.contiguous())
            continue

        half_shape = list(frames.shape)
        half_shape[1] = frames.shape[1] // 2
        half_shape[2] = pad_frames
        inactive_pad = comfy.latent_formats.Wan21().process_out(
            torch.zeros(half_shape, device=frames.device, dtype=frames.dtype)
        )
        reactive_pad = comfy.latent_formats.Wan21().process_out(
            torch.zeros_like(inactive_pad)
        )
        frame_pad = torch.cat([inactive_pad, reactive_pad], dim=1)

        mask_pad_shape = list(mask.shape)
        mask_pad_shape[2] = pad_frames
        mask_pad = torch.zeros(mask_pad_shape, device=mask.device, dtype=mask.dtype)

        out_frames.append(torch.cat([frame_pad, frames], dim=2).contiguous())
        out_masks.append(torch.cat([mask_pad, mask], dim=2).contiguous())

    return out_frames, out_masks, max_frames


class WanVaceToVideoHybridSubjectSwap:
    DESCRIPTION = (
        "Depth-first WAN VACE hybrid subject-swap node. Uses structural control video, a character reference, "
        "and optional source-RGB hybrid contexts for luma, detail, and background preservation."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "control_strength": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 4.0, "step": 0.01}),
                "reference_strength": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 4.0, "step": 0.01}),
                "source_luma_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 4.0, "step": 0.01}),
                "source_detail_strength": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 4.0, "step": 0.01}),
                "background_preserve_strength": ("FLOAT", {"default": 0.80, "min": 0.0, "max": 4.0, "step": 0.01}),
                "mask_feather_px": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 128.0, "step": 0.5}),
            },
            "optional": {
                "control_video": ("IMAGE",),
                "source_video": ("IMAGE",),
                "control_masks": ("MASK",),
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_latent")
    FUNCTION = "encode"
    CATEGORY = "conditioning/video_models"
    EXPERIMENTAL = True

    def encode(
        self,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        strength,
        control_strength,
        reference_strength,
        source_luma_strength,
        source_detail_strength,
        background_preserve_strength,
        mask_feather_px,
        control_video=None,
        source_video=None,
        control_masks=None,
        reference_image=None,
    ):
        device = comfy.model_management.intermediate_device()
        if control_strength < 0.0:
            control_strength = strength
        if reference_strength < 0.0:
            reference_strength = strength

        structure_video = _prepare_video_nhwc(
            control_video,
            width,
            height,
            length,
            device,
            fill_value=_NEUTRAL_RGB,
            tag="control_video",
        )
        source_rgb_video = _prepare_video_nhwc(
            source_video,
            width,
            height,
            length,
            device,
            fill_value=_NEUTRAL_RGB,
            tag="source_video",
        ) if source_video is not None else None
        reference_image_nhwc = _prepare_reference_image_nhwc(reference_image, width, height, device)

        subject_mask = _prepare_mask_nhwc(
            control_masks,
            width,
            height,
            length,
            device,
            default_value=1.0,
            feather_px=mask_feather_px,
        )
        background_region = torch.ones_like(subject_mask) if control_masks is None else (1.0 - subject_mask).contiguous()

        effective_reference_strength = float(reference_strength) if reference_image_nhwc is not None else 0.0
        effective_source_luma_strength = float(source_luma_strength) if source_rgb_video is not None else 0.0
        effective_source_detail_strength = float(source_detail_strength) if source_rgb_video is not None else 0.0
        effective_background_preserve_strength = float(background_preserve_strength) if source_rgb_video is not None else 0.0

        overall_strength = max(
            float(control_strength),
            effective_reference_strength,
            effective_source_luma_strength,
            effective_source_detail_strength,
            effective_background_preserve_strength,
            0.0,
        )
        weight_divisor = overall_strength if overall_strength > 0.0 else 1.0

        composite_video = _compose_composite_guidance_video(
            structure_video,
            source_rgb_video,
            subject_mask,
            background_region,
            control_weight=float(control_strength) / weight_divisor,
            source_luma_weight=effective_source_luma_strength / weight_divisor,
            source_detail_weight=effective_source_detail_strength / weight_divisor,
            background_weight=effective_background_preserve_strength / weight_divisor,
        )
        if reference_image_nhwc is not None and effective_reference_strength > 0.0:
            reference_image_nhwc = _scale_around_neutral(reference_image_nhwc, effective_reference_strength / weight_divisor)
        else:
            reference_image_nhwc = None

        control_frames, control_mask, trim_latent = _encode_vace_context(
            vae,
            control_video_nhwc=composite_video,
            control_mask_nhwc=subject_mask,
            reference_image_nhwc=reference_image_nhwc,
        )
        vace_frames_list = [control_frames]
        vace_mask_list = [control_mask]
        vace_strength_list = [float(overall_strength)]
        latent_length = control_frames.shape[2]

        positive = node_helpers.conditioning_set_values(
            positive,
            {
                "vace_frames": vace_frames_list,
                "vace_mask": vace_mask_list,
                "vace_strength": vace_strength_list,
            },
            append=True,
        )
        negative = node_helpers.conditioning_set_values(
            negative,
            {
                "vace_frames": vace_frames_list,
                "vace_mask": vace_mask_list,
                "vace_strength": vace_strength_list,
            },
            append=True,
        )

        latent = torch.zeros(
            [batch_size, 16, latent_length, height // 8, width // 8],
            device=device,
            dtype=torch.float32,
        )
        return (positive, negative, {"samples": latent}, trim_latent)


NODE_CLASS_MAPPINGS = {"WanVaceToVideoHybridSubjectSwap": WanVaceToVideoHybridSubjectSwap}
NODE_DISPLAY_NAME_MAPPINGS = {"WanVaceToVideoHybridSubjectSwap": "Wan VACE → Video (hybrid subject swap)"}
