# wan_vace_caps_inpaint.py
# ComfyUI custom node: Wan VACE → Video (caps inpaint)
#
# ------------------------------- WHAT THIS NODE IS -------------------------------
# This node implements **temporal inpainting** for WAN VACE:
#
# You provide ONLY the "cap" control frames (a start segment and an end segment).
# The node constructs a full-length control timeline by:
#   - placing the start cap at frames [0 : start_control_frames]
#   - placing the end cap at frames [length - end_control_frames : length]
#   - forcing the middle frames to a neutral RGB=0.5 filler (so *no* evidence leaks from any user filler)
#
# Critically, it also sets the VACE mask using **inpainting semantics**:
#   - vace_mask = 0  → "KNOWN / PRESERVE / PASSTHROUGH"  (model should *not* denoise this region)
#   - vace_mask = 1  → "UNKNOWN / INPAINT / DENOISE"     (model should *generate* this region)
#
# Therefore:
#   - start & end caps are "known": preserved and crisp
#   - the middle is "unknown": actually denoised/generated (no fog/veil, no passthrough decode)
#
# Why the neutral middle is forced to RGB=0.5:
#   WAN VAE encoding discards alpha and encodes RGB nonlinearly. If the middle contains any
#   hidden RGB (e.g., "transparent" PNG with RGB under alpha), that RGB becomes *evidence*
#   unless removed. For temporal inpainting you want *absence of evidence* in the unknown region.
#   Setting RGB=0.5 in the unknown region makes the provided middle content maximally neutral.
#
# In VACE, the control is represented as two plates (inactive/reactive) plus a mask:
#   inactive_nhwc = (cv_centered * (1 - mask)) + 0.5
#   reactive_nhwc = (cv_centered * (mask)) + 0.5
# With mask=0 in known regions, "inactive" carries the real control cap frames; with mask=1 in
# unknown regions, we force cv to 0.5 so both plates carry no informative signal while the
# vace_mask tells the model to denoise/generate there.
#
# ------------------------------- HOW TO USE -------------------------------
# - Set LENGTH to your desired output frames.
# - Provide control_video containing ONLY the cap frames (start+end). You can also provide a
#   longer clip; this node will only read the first start_control_frames and last end_control_frames
#   from it.
# - Set start_control_frames and end_control_frames.
# - The node will synthesize the middle.
#
# Drop file into: ComfyUI/custom_nodes/wan_vace_caps_inpaint.py
# Node name in UI: "Wan VACE → Video (caps inpaint)"

from __future__ import annotations

import torch
import torch.nn.functional as F

import nodes
import node_helpers
import comfy
import comfy.utils
import comfy.model_management
import comfy.latent_formats


# ------------------------------- helpers ------------------------------------

def _vae_encode_ensure_5d(vae, x_nhwc: torch.Tensor, tag: str) -> torch.Tensor:
    """
    Call vae.encode on NHWC (T,H,W,C) or (1,H,W,C) and ensure a 5D tensor [1,C,T,Hl,Wl].
    """
    out = vae.encode(x_nhwc[..., :3])
    if isinstance(out, dict):
        if "samples" in out:
            out = out["samples"]
        else:
            out = next((v for v in out.values() if torch.is_tensor(v)), None)
    if not torch.is_tensor(out):
        raise ValueError(f"{tag}: VAE.encode returned non-tensor type {type(out)}")

    if out.dim() == 5:
        return out.contiguous()
    if out.dim() == 4:
        return out.unsqueeze(2).contiguous()
    raise ValueError(f"{tag}: unexpected VAE.encode rank {out.dim()} with shape {tuple(out.shape)}")


def _upscale_nhwc(x_nhwc: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """NHWC -> resize -> NHWC"""
    x = comfy.utils.common_upscale(
        x_nhwc.movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)
    return x.to(dtype=torch.float32)


def _extract_caps_into_timeline(
    control_video: torch.Tensor,
    length: int,
    width: int,
    height: int,
    start_frames: int,
    end_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a length-T RGB control timeline:
      - start cap frames copied to the beginning
      - end cap frames copied to the end
      - middle forced to neutral 0.5
    control_video can be any length; we read from it safely without overlap.
    """
    # target timeline initialized to neutral 0.5 everywhere
    cv_tl = torch.ones((length, height, width, 3), device=device, dtype=torch.float32) * 0.5

    if control_video is None:
        return cv_tl

    cv_in = _upscale_nhwc(control_video, width, height).to(device)
    if cv_in.shape[-1] > 3:
        cv_in = cv_in[..., :3]
    T_in = cv_in.shape[0]

    # clamp requested caps to valid sizes
    start_frames = int(max(0, min(start_frames, length)))
    end_frames = int(max(0, min(end_frames, length - start_frames)))

    # pick start cap from the beginning of input
    start_take = min(start_frames, T_in)
    if start_take > 0:
        cv_tl[0:start_take] = cv_in[0:start_take]

    # pick end cap from the end of input, avoiding overlap with what we already consumed
    remaining = max(0, T_in - start_take)
    end_take = min(end_frames, remaining)
    if end_take > 0:
        end_src_start = T_in - end_take
        end_dst_start = length - end_take
        cv_tl[end_dst_start:length] = cv_in[end_src_start:T_in]

    # middle is already 0.5; keep it forced neutral (no evidence leakage)
    if start_frames + end_frames < length:
        mid0 = start_frames
        mid1 = length - end_frames
        cv_tl[mid0:mid1] = 0.5

    return cv_tl.contiguous()


# ------------------------------- node ---------------------------------------

class WanVaceToVideoCapsInpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "width":  ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81,  "min": 1,  "max": nodes.MAX_RESOLUTION, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "start_control_frames": ("INT", {"default": 24, "min": 0, "max": 100000, "step": 1}),
                "end_control_frames":   ("INT", {"default": 24, "min": 0, "max": 100000, "step": 1}),
            },
            "optional": {
                # control_video should contain ONLY the cap frames (start+end). Middle filler is auto-generated.
                "control_video": ("IMAGE", ),

                # reference (optional)
                "reference_image": ("IMAGE", ),
                "ref_strength":   ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1000.0, "step": 0.01}),
                "ctrl_strength":  ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1000.0, "step": 0.01}),
                "ref_noise_std":  ("FLOAT", {"default": 0.0,  "min": 0.0,  "max": 2.0,    "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_latent")
    FUNCTION = "encode"
    CATEGORY = "conditioning/video_models"
    EXPERIMENTAL = True

    def encode(
        self,
        positive, negative, vae,
        width, height, length, batch_size, strength,
        start_control_frames, end_control_frames,
        control_video=None,
        reference_image=None,
        ref_strength=-1.0, ctrl_strength=-1.0, ref_noise_std=0.0,
    ):
        if length < 1:
            raise ValueError("length must be >= 1")

        # clamp caps so they never exceed length
        start_control_frames = int(max(0, min(start_control_frames, length)))
        end_control_frames = int(max(0, min(end_control_frames, length - start_control_frames)))

        # if not set, mirror global strength
        if ref_strength < 0:
            ref_strength = strength
        if ctrl_strength < 0:
            ctrl_strength = strength

        device = comfy.model_management.intermediate_device()

        # Build RGB control timeline with neutral middle (no filler leakage)
        cv = _extract_caps_into_timeline(
            control_video=control_video,
            length=length,
            width=width,
            height=height,
            start_frames=start_control_frames,
            end_frames=end_control_frames,
            device=device,
        )  # [T,H,W,3] float32

        # Build per-frame inpaint mask (VACE semantics):
        #   0 = known/preserve (caps)
        #   1 = unknown/denoise (middle)
        mask = torch.zeros((length, height, width, 1), device=device, dtype=torch.float32)
        mid0 = start_control_frames
        mid1 = length - end_control_frames
        if mid1 > mid0:
            mask[mid0:mid1] = 1.0

        # Split into plates around 0.5 center
        cv_centered = cv - 0.5
        inactive_nhwc = (cv_centered * (1.0 - mask)) + 0.5  # known info lives here
        reactive_nhwc = (cv_centered * mask) + 0.5          # unknown region neutralized by cv=0.5

        # Encode (WAN VAE already applies temporal stride internally)
        inactive_5d = _vae_encode_ensure_5d(vae, inactive_nhwc, "inactive")  # [1,C,T_i,Hl,Wl]
        reactive_5d = _vae_encode_ensure_5d(vae, reactive_nhwc, "reactive")

        if inactive_5d.shape != reactive_5d.shape:
            raise ValueError(f"Inactive/Reactive latent shapes differ: {tuple(inactive_5d.shape)} vs {tuple(reactive_5d.shape)}")

        B, C, T_i, H_l, W_l = inactive_5d.shape
        if B != 1:
            raise ValueError(f"WAN VAE encode batch should be 1, got {B}")

        # Control latents: [1,2C,T_i,Hl,Wl]
        ctrl_2c_5d = torch.cat((inactive_5d, reactive_5d), dim=1).mul_(float(ctrl_strength))

        # ---- optional reference frame ----
        ref_T = 0
        if reference_image is not None:
            ri = reference_image
            while ri.dim() > 4:
                ri = ri[0]
            if ri.dim() == 3:
                ri = ri.unsqueeze(0)
            ri = _upscale_nhwc(ri, width, height).to(device)  # [1,H,W,C]

            ref_5d = _vae_encode_ensure_5d(vae, ri[:, :, :, :3], "reference")  # [1,C,1,Hl,Wl]
            if ref_noise_std > 0.0:
                ref_5d = ref_5d + torch.randn_like(ref_5d) * float(ref_noise_std)

            zero_block = comfy.latent_formats.Wan21().process_out(torch.zeros_like(ref_5d))
            ref_2c_5d = torch.cat((ref_5d, zero_block), dim=1).mul_(float(ref_strength))  # [1,2C,1,Hl,Wl]

            vace_frames_5d = torch.cat((ref_2c_5d, ctrl_2c_5d), dim=2)
            ref_T = 1
        else:
            vace_frames_5d = ctrl_2c_5d

        T_total = vace_frames_5d.shape[2]

        # ---- build vace_mask (64-channel spatial mask, temporally resampled to T_i) ----
        # Note: mask==1 marks frames to denoise/generate.
        vae_stride = 8
        Hm = height // vae_stride
        Wm = width // vae_stride

        # spatial pack into 64 channels (stock VACE layout)
        m = mask.view(length, Hm, vae_stride, Wm, vae_stride) \
                .permute(2, 4, 0, 1, 3) \
                .reshape(vae_stride * vae_stride, length, Hm, Wm)  # [64, length, Hm, Wm]

        # temporal resample mask to latent time length T_i (nearest-exact for crisp boundaries)
        if m.shape[1] != T_i:
            m = F.interpolate(m.unsqueeze(0), size=(T_i, Hm, Wm), mode="nearest-exact").squeeze(0)  # [64, T_i, Hm, Wm]

        # pad for reference frame if present
        if ref_T > 0:
            pad = torch.zeros((m.shape[0], ref_T, Hm, Wm), device=device, dtype=m.dtype)
            m = torch.cat((pad, m), dim=1)  # [64, T_total, Hm, Wm]

        vace_mask_5d = m.unsqueeze(0).contiguous()  # [1,64,T_total,Hm,Wm]

        # ---- conditioning payload ----
        positive = node_helpers.conditioning_set_values(
            positive,
            {"vace_frames": [vace_frames_5d], "vace_mask": [vace_mask_5d], "vace_strength": [float(strength)]},
            append=True,
        )
        negative = node_helpers.conditioning_set_values(
            negative,
            {"vace_frames": [vace_frames_5d], "vace_mask": [vace_mask_5d], "vace_strength": [float(strength)]},
            append=True,
        )

        # ---- latent stub (16-channel WAN latent) ----
        latent_stub = torch.zeros([batch_size, 16, T_total, Hm, Wm], device=device, dtype=torch.float32)
        out_latent = {"samples": latent_stub}

        trim_latent = ref_T
        return (positive, negative, out_latent, trim_latent)


# Register node
NODE_CLASS_MAPPINGS = {"WanVaceToVideoCapsInpaint": WanVaceToVideoCapsInpaint}
NODE_DISPLAY_NAME_MAPPINGS = {"WanVaceToVideoCapsInpaint": "Wan VACE → Video (caps inpaint)"}
