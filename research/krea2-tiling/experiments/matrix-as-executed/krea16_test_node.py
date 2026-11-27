"""Unreleased ComfyUI test node for the Krea 2 Turbo upscale matrix."""

import re
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

import comfy.model_management as mm
from comfy.patcher_extension import WrappersMP
from comfy.samplers import KSampler
import comfy.utils
from nodes import common_ksampler
from usdu_pose_tiled import krea2_vision as vision
from usdu_pose_tiled.krea2_control import square_tile_grid


VARIANTS = {
    "euler-simple": ("euler", "simple"),
    "euler-sgm_uniform": ("euler", "sgm_uniform"),
    "dpmpp_2m-simple": ("dpmpp_2m", "simple"),
}


def to_pil(image):
    pixels = image[0].detach().cpu().float().clamp(0, 1).numpy()
    return Image.fromarray((pixels * 255).round().astype(np.uint8))


def to_tensor(image):
    return torch.from_numpy(np.array(image.convert("RGB"))).unsqueeze(0).float() / 255


def geometry(width, height, padding, patch):
    final_width, final_height = width * 3, height * 3
    work_width = (final_width + 31) // 32 * 32
    work_height = (final_height + 31) // 32 * 32
    x_edges, y_edges = square_tile_grid(work_width, work_height, "9")
    layout = (x_edges, y_edges)
    tiles = vision._tiles(work_width, work_height, work_width, work_height, padding, patch, layout)

    def source_box(box):
        x1, y1, x2, y2 = box
        return (x1 // 3, y1 // 3, min(width, (x2 + 2) // 3), min(height, (y2 + 2) // 3))

    boxes = []
    for index, tile in enumerate(tiles):
        row, column = divmod(index, len(x_edges) - 1)
        core = (x_edges[column], y_edges[row], x_edges[column + 1], y_edges[row + 1])
        padded = tuple(v * vision.LATENT_SCALE for v in (tile.left, tile.top, tile.right, tile.bottom))
        boxes.append({
            "tile": index + 1,
            "core_canvas_xyxy": core,
            "padded_canvas_xyxy": padded,
            "core_source_xyxy": source_box(core),
            "padded_source_xyxy": source_box(padded),
        })
    return (final_width, final_height), (work_width, work_height), tiles, boxes


def core_caption_contexts(source, original_prompt, clip, boxes, tiles, final_size, features):
    original = to_pil(source)
    width, height = original.size
    overview = original.copy()
    overview.thumbnail((640, 640), Image.Resampling.LANCZOS)
    sx, sy = overview.width / width, overview.height / height
    instruction = (
        "Image 1 is the full composition. Its yellow rectangle marks the padded tile and cyan marks "
        "the inner tile. Image 2 is the clean padded tile for nearby context. Image 3 is the clean inside "
        "of the cyan rectangle. Give one short descriptive sentence about only the visible content in "
        "image 3. Name the visible part of any partial face, figure or object without completing it. "
        "A person or object seen only outside image 3 must stay out of the answer. Describe background "
        "and texture alone where applicable. Use the original creative prompt to clarify what you see. "
        "Output only the caption in English. Original creative prompt:\n" + original_prompt
    )
    template = clip.tokenizer.llama_template.replace("{}", vision.VISION_BLOCK + "\n{}", 1)

    captions, denoiser_crops = [], []
    for box, tile in zip(boxes, tiles, strict=True):
        mm.throw_exception_if_processing_interrupted()
        reference = overview.copy()
        painter = ImageDraw.Draw(reference)
        for region, color in (("padded", "#f7cb29"), ("core", "#24faff")):
            x1, y1, x2, y2 = box[f"{region}_source_xyxy"]
            painter.rectangle((int(x1 * sx), int(y1 * sy), min(reference.width-1, round(x2*sx)-1),
                               min(reference.height-1, round(y2*sy)-1)), outline=color, width=3)
        x1, y1, x2, y2 = box["padded_source_xyxy"]
        padded = source[:, y1:y2, x1:x2, :]
        cx1, cy1, cx2, cy2 = box["core_source_xyxy"]
        core = source[:, cy1:cy2, cx1:cx2, :]
        tokens = clip.tokenize(instruction, images=[to_tensor(reference), padded, core],
                               min_length=1, thinking=False)
        image_token_count = sum(isinstance(token, dict) and token.get("type") == "image"
                                for row in tokens.get("qwen3vl_4b", ()) for token, *_ in row)
        if image_token_count != 3:
            raise ValueError("The Krea 2 captioner must receive all three images.")
        raw = clip.decode(clip.generate(tokens, do_sample=False, max_length=96))
        caption = re.sub(r"^caption\s*:\s*", "", raw.rsplit("</think>", 1)[-1].strip().strip('"'),
                         flags=re.IGNORECASE).strip()
        if not caption or "<think>" in caption:
            raise ValueError(f"No usable caption for tile {box['tile']}: {raw!r}")
        box["caption"] = caption
        captions.append(caption)
        denoiser_crops.append(vision._source_crop(source, tile, *final_size))
    return [vision._encode_tile(clip, template, caption, crop, features)
            for caption, crop in zip(captions, denoiser_crops, strict=True)]


class STKrea2TurboMatrixTest:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",), "model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",),
            "original_prompt": ("STRING", {"multiline": True}),
            "seed": ("INT", {"default": 40309561, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
            "denoise": ("FLOAT", {"default": .8, "min": .01, "max": 1., "step": .01}),
            "padding": ("INT", {"default": 256, "min": 0, "max": 512, "step": 32}),
            "variant": (["all", *VARIANTS], {"default": "all"}),
        }}

    CATEGORY = "Sweet Tea/Tests"
    FUNCTION = "upscale"
    RETURN_TYPES = ("IMAGE",)
    DESCRIPTION = "Compare Krea 2 Turbo samplers with nine Vision tiles, at exact 3× size."

    @torch.inference_mode(False)
    @torch.no_grad()
    def upscale(self, image, model, clip, vae, original_prompt, seed, steps=8, denoise=.8,
                padding=256, variant="all", *, variants=None, on_prepared=None, on_render=None):
        from comfy.ldm.krea2.model import SingleStreamDiT
        from comfy.text_encoders.krea2 import Krea2Tokenizer

        if image.shape[0] != 1:
            raise ValueError("Run this test node with one source image at a time.")
        if not isinstance(clip.tokenizer, Krea2Tokenizer):
            raise ValueError("Load Qwen3-VL-4B with CLIPLoader type krea2.")
        diffusion = model.get_model_object("diffusion_model")
        if not isinstance(diffusion, SingleStreamDiT):
            raise ValueError("Use the native Krea 2 Turbo model.")
        source = image[..., :3].clone()
        height, width = source.shape[1:3]
        final, work, tiles, boxes = geometry(width, height, padding, diffusion.patch)
        final_width, final_height = final
        work_width, work_height = work
        started = time.monotonic()
        contexts = core_caption_contexts(source, original_prompt, clip, boxes, tiles, final,
                                         diffusion.txtlayers * diffusion.txtdim)
        caption_time = time.monotonic() - started
        large = comfy.utils.common_upscale(source.movedim(-1, 1), final_width, final_height,
                                          "lanczos", "disabled")
        if work != final:
            large = F.pad(large, (0, work_width - final_width, 0, work_height - final_height),
                          mode="replicate")
        latent = vae.encode_tiled(large.movedim(1, -1).contiguous())
        if latent.shape[-2:] != (work_height // 8, work_width // 8):
            raise ValueError("Use the Krea 2 Qwen Image VAE with 8× spatial compression.")
        if on_prepared:
            on_prepared({
                "final_size": final, "working_canvas": work, "tile_count": len(tiles),
                "padding": padding, "tiles": boxes, "caption_and_encode_seconds": round(caption_time, 2),
            })

        # ST's shared latent wrapper blends tile predictions on every denoising
        # step. Leaving its canvas-position patch unregistered keeps IDs local.
        tiled = model.clone()
        wrapper = vision.TiledVision(tiles, contexts, diffusion.patch, *latent.shape[-2:])
        tiled.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, vision.CANVAS_ORIGIN, wrapper)
        positive = [[contexts[0], {}]]
        outputs = []
        names = variants if variants is not None else (tuple(VARIANTS) if variant == "all" else (variant,))
        for name in names:
            sampler, scheduler = VARIANTS[name]
            if sampler not in KSampler.SAMPLERS or scheduler not in KSampler.SCHEDULERS:
                raise ValueError(f"ComfyUI does not support {sampler} / {scheduler}.")
            sigmas = KSampler(tiled, steps, torch.device("cpu"), sampler, scheduler, denoise).sigmas.tolist()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            sampled, = common_ksampler(tiled, seed, steps, 1.0, sampler, scheduler, positive, [],
                                       {"samples": latent}, denoise=denoise)
            torch.cuda.synchronize()
            sampling_seconds = time.monotonic() - started
            # Offload FP8 weights before the VAE temporarily enters inference mode.
            mm.unload_all_models()
            with torch.inference_mode():
                decoded = vae.decode_tiled(sampled["samples"], overlap=16)
                result = decoded.reshape(-1, *decoded.shape[-3:])[:, :final_height, :final_width, :3]
                if tuple(result.shape) != (1, final_height, final_width, 3) or not torch.isfinite(result).all():
                    raise RuntimeError(f"Invalid {name} output shape/values: {tuple(result.shape)}")
            torch.cuda.synchronize()
            if on_render:
                on_render(name, result, {
                    "sampler": sampler, "scheduler": scheduler, "sigmas": sigmas,
                    "seconds": round(time.monotonic() - started, 2),
                    "sampling_seconds": round(sampling_seconds, 2),
                    "peak_torch_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                    "peak_torch_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
                })
            outputs.append(result)
        return (torch.cat(outputs, dim=0),)


NODE_CLASS_MAPPINGS = {"STKrea2TurboMatrixTest": STKrea2TurboMatrixTest}
NODE_DISPLAY_NAME_MAPPINGS = {"STKrea2TurboMatrixTest": "ST Krea 2 Turbo Sampler Matrix (Test)"}
