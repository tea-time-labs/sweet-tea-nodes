import torch
import gc
import comfy

from .usdu_pose_tiled import USDUpscaler, USDUMode, USDUSFMode
from .utils import tensor_to_pil

MAX_RESOLUTION = 8192

MODES = {
    "Linear": USDUMode.LINEAR.value,
    "Chess": USDUMode.CHESS.value,
    "None": USDUMode.NONE.value,
}

SEAM_FIX_MODES = {
    "None": USDUSFMode.NONE.value,
    "Band Pass": USDUSFMode.BAND_PASS.value,
    "Half Tile": USDUSFMode.HALF_TILE.value,
    "Half Tile + Intersections": USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS.value,
}

def USDU_base_inputs():
    required = [
        ("image", ("IMAGE",)),
        ("model", ("MODEL",)),
        ("positive", ("CONDITIONING",)),
        ("negative", ("CONDITIONING",)),
        ("vae", ("VAE",)),
        ("upscale_by", ("FLOAT", {"default": 2.0, "min": 0.05, "max": 4.0, "step": 0.05})),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})),
        ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1})),
        ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0})),
        ("sampler_name", (comfy.samplers.KSampler.SAMPLERS,)),
        ("scheduler", (comfy.samplers.KSampler.SCHEDULERS,)),
        ("denoise", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01})),
        ("upscale_model", ("UPSCALE_MODEL",)),
        ("mode_type", ([k for k in MODES.keys()], {"default": "Linear"})),
        ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8})),
        ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8})),
        ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        ("seam_fix_mode", ([k for k in SEAM_FIX_MODES.keys()], {"default": "None"})),
        ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})),
        ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        ("force_uniform_tiles", ("BOOLEAN", {"default": True})),
        ("tiled_decode", ("BOOLEAN", {"default": False})),
    ]
    optional = [
        ("custom_sampler", ("SAMPLER",)),
        ("custom_sigmas", ("SIGMAS",)),
    ]
    return required, optional

def USDU_locked_inputs():
    # Start from the base inputs and append the lock controls.
    required, optional = USDU_base_inputs()
    required = list(required)
    optional += [
        ("lock_padding", ("BOOLEAN", {"default": True})),
        # 0 = use mask_blur, otherwise explicit blur radius (pixels) for the sampler noise mask
        ("noise_mask_blur", ("INT", {"default": 0, "min": -1, "max": 256, "step": 1})),
        #  - sequential: classic alpha-composite (can leak base/original into feather bands)
        #  - normalized: tile-only weighted blend (eliminates base/original bleed-through)
        ("blend_mode", (["sequential", "normalized"], {"default": "normalized"})),    
    ]
    return required, optional

class UltimateSDUpscalePoseTiled:
    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_locked_inputs()
        return {"required": dict(required), "optional": dict(optional)}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(
        self,
        image,
        model,
        positive,
        negative,
        vae,
        upscale_by,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        upscale_model,
        mode_type,
        tile_width,
        tile_height,
        mask_blur,
        tile_padding,
        seam_fix_mode,
        seam_fix_denoise,
        seam_fix_width,
        seam_fix_mask_blur,
        seam_fix_padding,
        force_uniform_tiles,
        tiled_decode,
        lock_padding=True,
        noise_mask_blur=0,
        blend_mode="normalized",
        custom_sampler=None,
        custom_sigmas=None,
    ):
        # Normalize batch to [B,H,W,C]
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Convert to PIL list
        pil_images = [tensor_to_pil(image, i) for i in range(image.shape[0])]

        # Build pipeline (locked-padding sampler)
        upscaler = USDUpscaler(
            upscale_by,
            upscale_model,
            MODES[mode_type],
            tile_width,
            tile_height,
            mask_blur,
            tile_padding,
            SEAM_FIX_MODES[seam_fix_mode],
            seam_fix_denoise,
            seam_fix_width,
            seam_fix_mask_blur,
            seam_fix_padding,
            force_uniform_tiles,
            tiled_decode,
            lock_padding=lock_padding,
            noise_mask_blur=noise_mask_blur,
            blend_mode=blend_mode,
        )

        # One-time cleanup before heavy passes (not per tile)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Run pipeline -> list of [C,H,W] tensors
        c_hw_tensors = upscaler.process(
            model, positive, negative, vae, seed, steps, cfg,
            sampler_name, scheduler,
            denoise, pil_images
        )

        # Convert each [C,H,W] -> [H,W,C] and stack back to [B,H,W,C]
        h_w_c_tensors = [t.permute(1, 2, 0) for t in c_hw_tensors]
        out = torch.stack(h_w_c_tensors, dim=0)

        return (out,)

NODE_CLASS_MAPPINGS = {
    "UltimateSDUpscalePoseTiled": UltimateSDUpscalePoseTiled
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSDUpscalePoseTiled": "Ultimate SD Upscale (Pose Tiled ControlNet)"
    }
