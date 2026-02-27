# wan_vace_dual_strength.py
# HARDCODED TEST: 24 known (passthrough), 33 unknown (denoise), 24 known
# - Known regions (mask=0): respect control frames (passthrough)
# - Unknown region (mask=1): generate/denoise, with filler neutralized
#
# Assumes length=81.
from __future__ import annotations

import torch
import torch.nn.functional as F

import nodes
import node_helpers
import comfy
import comfy.utils
import comfy.model_management
import comfy.latent_formats


def _vae_encode_ensure_5d(vae, x_nhwc: torch.Tensor, tag: str) -> torch.Tensor:
    out = vae.encode(x_nhwc[..., :3])
    if isinstance(out, dict):
        out = out.get("samples", next((v for v in out.values() if torch.is_tensor(v)), None))
    if not torch.is_tensor(out):
        raise ValueError(f"{tag}: VAE.encode returned non-tensor type {type(out)}")
    if out.dim() == 5:
        return out.contiguous()
    if out.dim() == 4:
        return out.unsqueeze(2).contiguous()
    raise ValueError(f"{tag}: unexpected VAE.encode rank {out.dim()} with shape {tuple(out.shape)}")


def _upscale_nhwc(x_nhwc: torch.Tensor, width: int, height: int) -> torch.Tensor:
    x = comfy.utils.common_upscale(x_nhwc.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
    return x.to(dtype=torch.float32)


class WanVaceToVideoPlus:
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
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            },
            "optional": {
                "control_video": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "ref_strength": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1000.0, "step": 0.01}),
                "ctrl_strength": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1000.0, "step": 0.01}),
                "ref_noise_std": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
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
        control_video=None,
        reference_image=None,
        ref_strength=-1.0, ctrl_strength=-1.0, ref_noise_std=0.0,
    ):
        if length != 81:
            raise ValueError(f"Hardcoded test expects length=81 (24+33+24). Got {length}.")
        if control_video is None:
            raise ValueError("This test requires control_video.")

        if ref_strength < 0:
            ref_strength = strength
        if ctrl_strength < 0:
            ctrl_strength = strength

        device = comfy.model_management.intermediate_device()

        # segments
        seg_a = 24
        seg_mid = 33
        mid0 = seg_a
        mid1 = seg_a + seg_mid  # 57 exclusive; mid frames are 24..56

        # control video frames
        cv = _upscale_nhwc(control_video[:length], width, height).to(device)[:, :, :, :3]

        # neutralize filler evidence in middle no matter what user provided
        cv = cv.clone()
        cv[mid0:mid1] = 0.5

        # IMPORTANT: VACE mask semantics for inpainting:
        # mask=1 => region to be denoised/generated (unknown)
        # mask=0 => region to be preserved/passthrough (known)
        mask = torch.zeros((length, height, width, 1), device=device, dtype=torch.float32)
        mask[mid0:mid1] = 1.0  # ONLY middle is "unknown"

        # split plates (standard VACE construction)
        cv_centered = cv - 0.5
        inactive_nhwc = (cv_centered * (1.0 - mask)) + 0.5  # known area retains cv
        reactive_nhwc = (cv_centered * mask) + 0.5          # unknown area gets cv (but we forced it to 0.5)

        # encode (WAN VAE already applies temporal stride internally: T_i ~ 21)
        inactive_5d = _vae_encode_ensure_5d(vae, inactive_nhwc, "inactive")
        reactive_5d = _vae_encode_ensure_5d(vae, reactive_nhwc, "reactive")

        if inactive_5d.shape != reactive_5d.shape:
            raise ValueError(f"Inactive/Reactive latent shapes differ: {tuple(inactive_5d.shape)} vs {tuple(reactive_5d.shape)}")

        B, C, T_i, H_l, W_l = inactive_5d.shape
        if B != 1:
            raise ValueError(f"Expected WAN VAE batch=1, got {B}")

        # concat and scale
        ctrl_2c_5d = torch.cat((inactive_5d, reactive_5d), dim=1).mul_(float(ctrl_strength))  # [1,2C,T_i,Hl,Wl]

        # reference (optional)
        ref_T = 0
        if reference_image is not None:
            ri = reference_image
            while ri.dim() > 4:
                ri = ri[0]
            if ri.dim() == 3:
                ri = ri.unsqueeze(0)
            ri = _upscale_nhwc(ri, width, height).to(device)

            ref_5d = _vae_encode_ensure_5d(vae, ri[:, :, :, :3], "reference")  # [1,C,1,Hl,Wl]
            if ref_noise_std > 0.0:
                ref_5d = ref_5d + torch.randn_like(ref_5d) * float(ref_noise_std)

            zero_block = comfy.latent_formats.Wan21().process_out(torch.zeros_like(ref_5d))
            ref_2c_5d = torch.cat((ref_5d, zero_block), dim=1).mul_(float(ref_strength))

            vace_frames_5d = torch.cat((ref_2c_5d, ctrl_2c_5d), dim=2)
            ref_T = 1
        else:
            vace_frames_5d = ctrl_2c_5d

        T_total = vace_frames_5d.shape[2]

        # build vace_mask in latent time (aligned to temporal stride)
        # middle frames 24..56 map to latent indices 6..14 (inclusive)
        lat_mid0 = mid0 // 4
        lat_mid1 = ((mid1 - 1) // 4) + 1
        if not (0 <= lat_mid0 < lat_mid1 <= T_i):
            raise ValueError(f"Computed latent mid slice [{lat_mid0}:{lat_mid1}] out of bounds for T_i={T_i}")

        vae_stride = 8
        Hm = height // vae_stride
        Wm = width // vae_stride

        m_lat = torch.zeros((64, T_i, Hm, Wm), device=device, dtype=torch.float32)
        m_lat[:, lat_mid0:lat_mid1] = 1.0  # ONLY middle is denoised/generated

        if ref_T > 0:
            pad = torch.zeros((64, ref_T, Hm, Wm), device=device, dtype=torch.float32)
            m_lat = torch.cat((pad, m_lat), dim=1)

        vace_mask_5d = m_lat.unsqueeze(0).contiguous()  # [1,64,T_total,Hm,Wm]

        positive = node_helpers.conditioning_set_values(
            positive, {"vace_frames": [vace_frames_5d], "vace_mask": [vace_mask_5d], "vace_strength": [float(strength)]}, append=True
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"vace_frames": [vace_frames_5d], "vace_mask": [vace_mask_5d], "vace_strength": [float(strength)]}, append=True
        )

        latent_stub = torch.zeros([batch_size, 16, T_total, Hm, Wm], device=device, dtype=torch.float32)
        out_latent = {"samples": latent_stub}
        return (positive, negative, out_latent, ref_T)


NODE_CLASS_MAPPINGS = {"WanVaceToVideoPlus": WanVaceToVideoPlus}
NODE_DISPLAY_NAME_MAPPINGS = {"WanVaceToVideoPlus": "Wan VACE → Video (dual strength)"}  # keep your existing name
