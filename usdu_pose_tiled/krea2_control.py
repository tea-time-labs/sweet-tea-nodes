"""Krea2 controls are model patches, so encode the matching hint for every tile."""
import math
import torch
import comfy.utils
import nodes

from .nodes_pose_tiled import UltimateSDUpscalePoseTiled

TILE_HOOK = "sweet_tea_tile_model"
TILE_COUNTS = ["Auto", *(str(axis * axis) for axis in range(2, 11))]


@torch.inference_mode()
def prepare_tile_model(model, vae, crop_region, canvas_size, sample_size):
    get_attachment = getattr(model, "get_attachment", None)
    hook = get_attachment(TILE_HOOK) if get_attachment else None
    return hook(model, vae, crop_region, canvas_size, sample_size) if hook else model


def crop_hint(image, crop_region, canvas_size, sample_size):
    """Map the canvas crop into the hint without allocating an upscaled full hint."""
    from PIL import Image
    import numpy as np
    cw, ch = canvas_size
    ih, iw = image.shape[1:3]
    x1, y1, x2, y2 = crop_region
    box = (x1 * iw / cw, y1 * ih / ch, x2 * iw / cw, y2 * ih / ch)
    # PIL's float source box avoids rounding different tile edges inconsistently.
    result = []
    for frame in image:
        pixels = (frame.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(pixels).resize(sample_size, Image.Resampling.BICUBIC, box=box)
        result.append(torch.from_numpy(np.array(pil).astype(np.float32) / 255))
    return torch.stack(result)


def dimensions(width, height, scale, tile_size):
    w = max(32, round(width * scale / 32) * 32)
    h = max(32, round(height * scale / 32) * 32)
    cols, rows = math.ceil(w / tile_size), math.ceil(h / tile_size)
    tw = math.ceil(w / cols / 32) * 32
    th = math.ceil(h / rows / 32) * 32
    return w, h, tw, th


def square_tile_grid(width, height, tile_count):
    tiles_per_axis = math.isqrt(int(tile_count))
    if min(width, height) < tiles_per_axis * 32:
        raise ValueError(
            f"{tile_count} tiles require at least {tiles_per_axis * 32}px on the shorter output "
            "side. Choose fewer tiles or increase the upscale factor."
        )

    def edges(length):
        blocks = length // 32
        return tuple(index * blocks // tiles_per_axis * 32 for index in range(tiles_per_axis + 1))

    return edges(width), edges(height)


def required_node(name, repo):
    cls = nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(f"{name} is required for this control mode. Install {repo} and restart ComfyUI.")
    return cls


class STKrea2UltimateUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {"required": {
            "image": ("IMAGE",), "model": ("MODEL",),
            "positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "vae": ("VAE",),
            "upscale_by": ("FLOAT", {"default": 2., "min": .1, "max": 8., "step": .05}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 8, "min": 1, "max": 1000}),
            "cfg": ("FLOAT", {"default": 1., "min": 0., "max": 100.}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            "denoise": ("FLOAT", {"default": .25, "min": 0., "max": 1., "step": .01}),
            "tile_size": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 32}),
            "tile_padding": ("INT", {"default": 64, "min": 0, "max": 512, "step": 32}),
            "mask_blur": ("INT", {"default": 16, "min": 0, "max": 64}),
            "control_mode": (["Off", "Depth", "Canny", "Tile", "Vision"], {"default": "Off"}),
            "control_strength": ("FLOAT", {"default": .7, "min": 0., "max": 2., "step": .05}),
            "depth_lora": ("STRING", {"default": "depth-control-lora.safetensors"}),
            "canny_lora": ("STRING", {"default": "NK2E-canny-v0.1.safetensors"}),
            "tile_lora": ("STRING", {"default": "krea2_tile_controllora_step001250.safetensors"}),
            "depth_resolution": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
            "canny_low": ("FLOAT", {"default": .1, "min": 0., "max": 1., "step": .01}),
            "canny_high": ("FLOAT", {"default": .2, "min": 0., "max": 1., "step": .01}),
        }, "optional": {
            "tile_count": (TILE_COUNTS, {"default": "Auto", "tooltip":
                "Choose the total number of tiles in a square grid. Auto uses Tile size. "
                "Fewer tiles on a large output use more GPU memory."}),
            "upscale_model": ("UPSCALE_MODEL",),
            "clip": ("CLIP", {"tooltip": "Krea2 Qwen3-VL-4B encoder for Vision mode."}),
            "vision_prompt": ("STRING", {"default": "", "multiline": True,
                                         "tooltip": "Text paired with the source crop for each tile in Vision mode."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "Sweet Tea/Krea2"
    DESCRIPTION = ("Krea2 tiled img2img using Sweet Tea Ultimate SD Upscale. Off requires no control "
                   "weights; Depth uses Depth Anything V2 Large and facok; Canny uses experimental NK2E; "
                   "Tile requires approved compatible weights. Vision uses Qwen3-VL source crops on a shared latent. "
                   "One control at a time. Optional reference "
                   "editing belongs in a separate first pass, before this node.")

    def upscale(self, image, model, positive, negative, vae, upscale_by, seed, steps, cfg,
                sampler_name, scheduler, denoise, tile_size, tile_padding, mask_blur,
                control_mode, control_strength, depth_lora, canny_lora, tile_lora,
                depth_resolution, canny_low, canny_high, upscale_model=None, clip=None,
                vision_prompt="", tile_count="Auto"):
        w, h, tw, th = dimensions(image.shape[2], image.shape[1], upscale_by, tile_size)
        tile_count = str(tile_count)
        if tile_count not in TILE_COUNTS:
            raise ValueError(f"Invalid Krea2 tile count: {tile_count}. Choose Auto or a listed square count.")
        tile_grid = None
        if tile_count != "Auto":
            tile_grid = square_tile_grid(w, h, tile_count)
            tw, th = (max(end - start for start, end in zip(edges, edges[1:])) for edges in tile_grid)
        if control_mode in ("Depth", "Canny", "Tile") and control_strength > 0:
            if any(c.get("reference_latents") or c.get("nk2e_refs") for _, c in positive):
                raise ValueError("Apply reference editing before tiled control, and use text-only conditioning here.")
            import folder_paths
            filename = {"Depth": depth_lora, "Canny": canny_lora, "Tile": tile_lora}[control_mode]
            if not folder_paths.get_full_path("loras", filename):
                extra = " The AdwolfCzar Tile weights require approval from the author." if control_mode == "Tile" else ""
                raise FileNotFoundError(f"Missing {control_mode} control weights: {filename}.{extra}")
            hint = image
            if control_mode == "Depth":
                pre = required_node("DepthAnythingV2Preprocessor", "Fannovel16/comfyui_controlnet_aux")()
                hint = pre.execute(image, ckpt_name="depth_anything_v2_vitl.pth", resolution=depth_resolution)[0]
            elif control_mode == "Canny":
                if canny_low >= canny_high:
                    raise ValueError("Canny low threshold must be lower than high threshold.")
                from comfy_extras.nodes_canny import Canny
                hint = Canny().detect_edge(image, canny_low, canny_high)[0]

            if control_mode in ("Depth", "Tile"):
                loader = required_node("Krea2ControlLoRALoader", "facok/comfyui-krea2-controlnet")()
                apply = required_node("Krea2ControlApply", "facok/comfyui-krea2-controlnet")()
                model = loader.load_lora(model, filename, control_strength)[0]
                def hook(base, tile_vae, crop, canvas, sample):
                    encoded = tile_vae.encode(crop_hint(hint, crop, canvas, sample).to(image.device))
                    return apply.apply(base, {"samples": encoded})[0]
            else:
                edit = required_node("NK2EInContextEditNode", "Nynxz/ComfyUI-NK2E")
                model = nodes.LoraLoaderModelOnly().load_lora_model_only(model, filename, control_strength)[0]
                def hook(base, tile_vae, crop, canvas, sample):
                    encoded = tile_vae.encode(crop_hint(hint, crop, canvas, sample).to(image.device))
                    return edit.execute(base, {"samples": encoded})[0]
            model = model.clone()
            model.set_attachments(TILE_HOOK, hook)

        # Resize once to the requested canvas, preserving aspect to within the 32px lattice.
        source = image
        if upscale_model is not None:
            from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
            image = ImageUpscaleWithModel().upscale(upscale_model, image)[0]
        image = comfy.utils.common_upscale(image.movedim(-1, 1), w, h, "lanczos", "disabled").movedim(1, -1)
        if control_mode == "Vision":
            from .krea2_vision import upscale_with_vision
            return upscale_with_vision(source, image, model, clip, vision_prompt, negative, vae,
                                       tw, th, tile_padding, seed, steps, cfg, sampler_name,
                                       scheduler, denoise, tile_grid=tile_grid)

        return UltimateSDUpscalePoseTiled().upscale(
            image=image, model=model, positive=positive, negative=negative, vae=vae,
            upscale_by=1., seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name,
            scheduler=scheduler, denoise=denoise, upscale_model=None, mode_type="Linear",
            tile_width=tw, tile_height=th, mask_blur=mask_blur, tile_padding=tile_padding,
            seam_fix_mode="None", seam_fix_denoise=denoise, seam_fix_width=64,
            seam_fix_mask_blur=mask_blur, seam_fix_padding=32, force_uniform_tiles=False,
            tiled_decode=False, lock_padding=True, noise_mask_blur=0, blend_mode="normalized",
            tile_grid=tile_grid)


NODE_CLASS_MAPPINGS = {"STKrea2UltimateUpscale": STKrea2UltimateUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {"STKrea2UltimateUpscale": "Krea2 Ultimate Upscale (Sweet Tea)"}
