"""Krea 2 tiled upscale with independently selectable guides."""

import math
import re

import torch
import torch.nn.functional as F

from . import krea2_vision as vision
from .krea2_control import TILE_COUNTS, required_node, square_tile_grid
from .krea2_guides import TileGuides, load_depth


def resized(image, size, work, method):
    import comfy.utils

    pixels = comfy.utils.common_upscale(image.movedim(-1, 1), *size, method, "disabled")
    pixels = F.pad(pixels, (0, work[0] - size[0], 0, work[1] - size[1]), mode="replicate")
    return pixels.movedim(1, -1).contiguous()


def vision_contexts(source, size, grid, tiles, clip, prompt, features):
    from comfy.model_management import throw_exception_if_processing_interrupted
    import numpy as np
    from PIL import Image, ImageDraw

    height, width = source.shape[1:3]
    overview = Image.fromarray(
        (source[0].detach().cpu().float().clamp(0, 1).numpy() * 255).round().astype("uint8")
    )
    overview.thumbnail((640, 640), Image.Resampling.LANCZOS)
    instruction = (
        "Image 1 is the full scene. Yellow outlines the padded crop and cyan outlines its core. "
        "Image 2 shows the padded crop. Image 3 shows just its core. Describe only the visible "
        "parts in image 3 in one short sentence. Do not complete partial people or objects. "
        "Use the original creative prompt to clarify what is visible. Original prompt:\n" + prompt
    )
    template = clip.tokenizer.llama_template.replace("{}", vision.VISION_BLOCK + "\n{}", 1)

    def box(x1, y1, x2, y2):
        return (
            x1 * width // size[0],
            y1 * height // size[1],
            min(width, math.ceil(x2 * width / size[0])),
            min(height, math.ceil(y2 * height / size[1])),
        )

    xs, ys = grid
    contexts = []
    for i, tile in enumerate(tiles):
        throw_exception_if_processing_interrupted()
        row, col = divmod(i, len(xs) - 1)
        core = box(xs[col], ys[row], xs[col + 1], ys[row + 1])
        padded = box(tile.left * 8, tile.top * 8, tile.right * 8, tile.bottom * 8)
        marked = overview.copy()
        draw = ImageDraw.Draw(marked)
        images = []
        for (l, t, r, b), colour in ((padded, "#f7cb29"), (core, "#24faff")):
            left, top = l * marked.width // width, t * marked.height // height
            right, bottom = math.ceil(r * marked.width / width) - 1, math.ceil(b * marked.height / height) - 1
            draw.rectangle((left, top, right, bottom), outline=colour, width=3)
            images.append(source[:, t:b, l:r])
        marked_tensor = torch.from_numpy(np.array(marked)).unsqueeze(0).float() / 255
        tokens = clip.tokenize(instruction, images=[marked_tensor, *images], min_length=1, thinking=False)
        raw = clip.decode(clip.generate(tokens, do_sample=False, max_length=96))
        caption = re.sub(r"^caption\s*:\s*", "", raw.rsplit("</think>", 1)[-1].strip(), flags=re.I).strip(
            '" '
        )
        if not caption or "<think>" in caption:
            raise RuntimeError(f"Qwen did not return a caption for tile {i + 1}.")
        crop = vision._source_crop(source, tile, *size)
        contexts.append(vision._encode_tile(clip, template, caption, crop, features))
    return contexts


class STKrea2GuidedUpscale:
    CATEGORY = "Sweet Tea/Krea2"
    FUNCTION = "upscale"
    RETURN_TYPES = ("IMAGE",)
    DESCRIPTION = (
        "Upscale one shared latent canvas. Vision captions each tile from the original prompt "
        "and its source region. Canny, Depth and a colour reference can be used together."
    )

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        strength = ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01})
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "upscale_by": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 8.0, "step": 0.05}),
                "tile_count": (TILE_COUNTS, {"default": "9"}),
                "tile_size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 128,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "Maximum core size in Auto mode.",
                    },
                ),
                "tile_padding": ("INT", {"default": 256, "min": 0, "max": 512, "step": 32}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vision_enabled": ("BOOLEAN", {"default": True}),
                "canny_enabled": ("BOOLEAN", {"default": False}),
                "canny_strength": strength,
                "depth_enabled": ("BOOLEAN", {"default": False}),
                "depth_strength": strength,
                "reference_enabled": ("BOOLEAN", {"default": False}),
                "reference_strength": strength,
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
                "tile_positions": (
                    ["Canvas", "Tile-local"],
                    {
                        "default": "Canvas",
                        "tooltip": "Canvas gives each patch its full-image position. Tile-local starts each padded tile at its own origin.",
                    },
                ),
                "negative": ("CONDITIONING",),
                "canny_lora": ("STRING", {"default": "NK2E-canny-v0.1.safetensors"}),
                "depth_lora": ("STRING", {"default": "depth-control-lora.safetensors"}),
                "reference_lora": ("STRING", {"default": "NK2E-v0.3.safetensors"}),
            },
        }

    @torch.inference_mode(False)
    @torch.no_grad()
    def upscale(
        self,
        image,
        model,
        clip,
        vae,
        prompt,
        upscale_by,
        tile_count,
        tile_size,
        tile_padding,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        vision_enabled,
        canny_enabled,
        canny_strength,
        depth_enabled,
        depth_strength,
        reference_enabled,
        reference_strength,
        upscale_model=None,
        negative=None,
        canny_lora="NK2E-canny-v0.1.safetensors",
        depth_lora="depth-control-lora.safetensors",
        reference_lora="NK2E-v0.3.safetensors",
        tile_positions="Canvas",
    ):
        import comfy.model_management as mm
        import nodes
        from comfy.ldm.krea2.model import SingleStreamDiT
        from comfy.patcher_extension import WrappersMP
        from comfy.text_encoders.krea2 import Krea2Tokenizer

        height, width = image.shape[1:3]
        size = (max(1, round(width * upscale_by)), max(1, round(height * upscale_by)))
        work = tuple(math.ceil(n / 32) * 32 for n in size)
        if tile_count == "Auto":
            grid = tuple((*range(0, n, tile_size), n) for n in work)
        elif str(tile_count) in TILE_COUNTS:
            grid = square_tile_grid(*work, tile_count)
        else:
            raise ValueError(f"Unknown tile count {tile_count}.")
        canny = canny_enabled and canny_strength > 0
        depth = depth_enabled and depth_strength > 0
        reference = reference_enabled and reference_strength > 0
        source = image[..., :3]
        initial = source
        if upscale_model is not None:
            from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

            (initial,) = ImageUpscaleWithModel().upscale(upscale_model, source)
        initial = resized(initial, size, work, "lanczos")
        if denoise == 0:
            return (initial[:, : size[1], : size[0]].contiguous(),)

        diffusion = model.get_model_object("diffusion_model")
        if not isinstance(diffusion, SingleStreamDiT) or not isinstance(clip.tokenizer, Krea2Tokenizer):
            raise ValueError("Use the native Krea 2 model and Qwen3-VL CLIP loaded with type krea2.")
        if negative and any(
            meta.get("reference_latents") or meta.get("control") or "area" in meta or "mask" in meta
            for _, meta in negative
        ):
            raise ValueError("Use plain negative conditioning with the shared Krea 2 tile canvas.")
        patch = diffusion.patch
        tiles = vision._tiles(*work, *work, tile_padding, patch, grid)
        base = model.clone()
        depth_weight = None
        if depth:
            base, depth_weight = load_depth(base, depth_lora, depth_strength)
        for enabled, filename in ((canny, canny_lora), (reference, reference_lora)):
            if enabled:
                (base,) = nodes.LoraLoaderModelOnly().load_lora_model_only(base, filename, 0.7)
        if tile_positions == "Canvas":
            base.set_model_patch(vision._position_on_canvas, "post_input")
        elif tile_positions != "Tile-local":
            raise ValueError(f"Unknown tile positions {tile_positions!r}.")
        text = None
        if not vision_enabled:
            text = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        if negative is None:
            negative = clip.encode_from_tokens_scheduled(clip.tokenize("")) if cfg != 1 else []
        results = []
        for original, frame in zip(source.split(1), initial.split(1)):
            contexts = [None] * len(tiles)
            positive = text
            if vision_enabled:
                contexts = vision_contexts(
                    original, size, grid, tiles, clip, prompt, diffusion.txtlayers * diffusion.txtdim
                )
                positive = [[contexts[0], {}]]
            latent = vae.encode_tiled(frame)
            if latent.shape[-2:] != (work[1] // 8, work[0] // 8):
                raise ValueError("Use the Krea 2 compatible Qwen Image VAE with 8x spatial compression.")

            def encode_guide(pixels, method):
                return vae.encode_tiled(resized(pixels, size, work, method))

            refs = []
            if reference:
                clean = latent if upscale_model is None else encode_guide(original, "lanczos")
                refs.append((base.model.process_latent_in(clean).cpu(), reference_strength))
            if canny:
                from comfy_extras.nodes_canny import Canny

                (edges,) = Canny().detect_edge(original, 0.1, 0.4)
                refs.append(
                    (base.model.process_latent_in(encode_guide(edges, "bicubic")).cpu(), canny_strength)
                )
            depth_latent = None
            if depth:
                pre = required_node("DepthAnythingV2Preprocessor", "Fannovel16/comfyui_controlnet_aux")()
                (guide,) = pre.execute(original, ckpt_name="depth_anything_v2_vitl.pth", resolution=1024)
                depth_latent = base.model.process_latent_in(encode_guide(guide, "bicubic")).cpu()
            guide = (
                TileGuides(refs, depth_latent, depth_weight, depth_strength, patch) if refs or depth else None
            )
            tiled = base.clone()
            wrapper = vision.TiledVision(tiles, contexts, patch, *latent.shape[-2:], guide=guide)
            tiled.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, vision.CANVAS_ORIGIN, wrapper)
            (sampled,) = nodes.common_ksampler(
                tiled,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                {"samples": latent},
                denoise=denoise,
            )
            mm.unload_all_models()
            with torch.inference_mode():
                decoded = vae.decode_tiled(sampled["samples"], overlap=16)
            results.append(decoded.reshape(-1, *decoded.shape[-3:])[:, : size[1], : size[0]])
        return (torch.cat(results),)
