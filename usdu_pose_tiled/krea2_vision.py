"""Krea 2 vision-guided diffusion over a shared latent canvas."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from nodes import common_ksampler


LATENT_SCALE = 8
VISION_MAX_SIDE = 448
CANVAS_ORIGIN = "sweet_tea_krea2_canvas_origin"
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


@dataclass(frozen=True)
class Tile:
    top: int
    bottom: int
    left: int
    right: int
    weight: torch.Tensor

    @property
    def region(self):
        return (..., slice(self.top, self.bottom), slice(self.left, self.right))


def _axis_tiles(length, boundaries, padding):
    intervals = [
        (max(0, start - padding), min(length, end + padding))
        for start, end in zip(boundaries, boundaries[1:])
    ]
    for index, (start, end) in enumerate(intervals):
        weight = torch.ones(end - start)
        before = intervals[index - 1][1] - start if index else 0
        after = end - intervals[index + 1][0] if index + 1 < len(intervals) else 0
        if before:
            t = (torch.arange(before) + 0.5) / before
            weight[:before] *= (1 - torch.cos(math.pi * t)) / 2
        if after:
            t = (torch.arange(after) + 0.5) / after
            weight[-after:] *= (1 + torch.cos(math.pi * t)) / 2
        yield start, end, weight


def _tiles(width, height, core_width, core_height, padding, patch, tile_grid=None):
    padding = (padding // LATENT_SCALE // patch) * patch
    if tile_grid is None:
        x_edges = (*range(0, width, core_width), width)
        y_edges = (*range(0, height, core_height), height)
    else:
        x_edges, y_edges = tile_grid
    cols = tuple(_axis_tiles(width // LATENT_SCALE, [edge // LATENT_SCALE for edge in x_edges], padding))
    rows = tuple(_axis_tiles(height // LATENT_SCALE, [edge // LATENT_SCALE for edge in y_edges], padding))
    return [
        Tile(top, bottom, left, right, vertical[:, None] * horizontal[None, :])
        for top, bottom, vertical in rows
        for left, right, horizontal in cols
    ]


def _source_crop(source, tile, canvas_width, canvas_height):
    height, width = source.shape[1:3]
    left = tile.left * LATENT_SCALE * width // canvas_width
    right = math.ceil(tile.right * LATENT_SCALE * width / canvas_width)
    top = tile.top * LATENT_SCALE * height // canvas_height
    bottom = math.ceil(tile.bottom * LATENT_SCALE * height / canvas_height)
    crop = source[:, top:bottom, left:right, :3]
    crop_height, crop_width = crop.shape[1:3]
    scale = VISION_MAX_SIDE / max(crop_height, crop_width)
    if scale < 1:
        size = (max(1, round(crop_height * scale)), max(1, round(crop_width * scale)))
        crop = F.interpolate(crop.movedim(-1, 1), size=size, mode="area").movedim(1, -1)
    return crop.contiguous()


def _encode_tile(clip, template, prompt, crop, context_features):
    tokens = clip.tokenize(prompt, images=[crop], llama_template=template)
    if not any(isinstance(token, dict) and token.get("type") == "image"
               for row in tokens.get("qwen3vl_4b", ()) for token, *_ in row):
        raise ValueError("Krea2 Vision requires a Qwen3-VL encoder that supports image tokens. "
                         "Use a current ComfyUI and load Qwen3-VL-4B with CLIPLoader type krea2.")
    context = clip.encode_from_tokens(tokens)
    if context.ndim != 3 or context.shape[0] != 1 or context.shape[-1] != context_features:
        raise ValueError("Krea2 Vision requires Qwen3-VL-4B loaded with CLIPLoader type krea2.")
    return context.to(device="cpu")


def _position_on_canvas(args):
    offset = args["transformer_options"].get(CANVAS_ORIGIN)
    if offset is None:
        return args
    ids = args["img_ids"].clone()
    ids[..., 1] += offset[0]
    ids[..., 2] += offset[1]
    return {**args, "img_ids": ids}


class TiledVision:
    def __init__(self, tiles, contexts, patch, latent_height, latent_width):
        self.tiles = tiles
        self.contexts = contexts
        self.patch = patch
        self.weights = torch.zeros((latent_height, latent_width))
        for tile in tiles:
            self.weights[tile.region] += tile.weight

    def __call__(self, executor, x, timesteps, context, attention_mask=None,
                 ref_latents=None, transformer_options=None, **kwargs):
        from comfy.model_management import throw_exception_if_processing_interrupted

        if ref_latents:
            raise ValueError("Apply Krea2 reference edits before Vision upscale.")
        options = transformer_options or {}
        groups = options.get("cond_or_uncond") or [0]
        group_size = x.shape[0] // len(groups)
        combined = torch.zeros_like(x, dtype=torch.float32)
        for tile, tile_context in zip(self.tiles, self.contexts):
            throw_exception_if_processing_interrupted()
            region = tile.region
            weight = tile.weight.to(x.device)
            offset = (tile.top // self.patch, tile.left // self.patch)
            for index, group in enumerate(groups):
                rows = slice(index * group_size, (index + 1) * group_size)
                local_options = {**options, CANVAS_ORIGIN: offset, "cond_or_uncond": [group]}
                if "uuids" in options:
                    local_options["uuids"] = options["uuids"][index:index + 1]
                if group == 0:
                    cond = tile_context.to(device=context.device, dtype=context.dtype)
                    cond = cond.expand(group_size, -1, -1)
                    mask = None
                else:
                    cond = context[rows]
                    mask = attention_mask[rows] if attention_mask is not None else None
                result = executor(x[rows][region], timesteps[rows], cond, mask,
                                  None, local_options, **kwargs)
                combined[rows][region] += result.float() * weight
        return combined / self.weights.to(x.device)


@torch.inference_mode()
def upscale_with_vision(source, image, model, clip, prompt, negative, vae, core_width,
                        core_height, padding, seed, steps, cfg, sampler_name, scheduler,
                        denoise, tile_grid=None):
    from comfy.ldm.krea2.model import SingleStreamDiT
    from comfy.model_management import throw_exception_if_processing_interrupted
    from comfy.patcher_extension import WrappersMP
    from comfy.text_encoders.krea2 import Krea2Tokenizer

    if clip is None or not isinstance(clip.tokenizer, Krea2Tokenizer):
        raise ValueError("Connect Qwen3-VL-4B loaded with CLIPLoader type krea2 to the Vision clip input.")
    diffusion_model = model.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, SingleStreamDiT):
        raise ValueError("Krea2 Vision requires the native Krea 2 diffusion model.")
    if cfg != 1.0 and any("area" in meta or "mask" in meta or meta.get("control")
                          or meta.get("reference_latents") for _, meta in negative):
        raise ValueError("Use plain text negative conditioning with Krea2 Vision.")
    if denoise == 0:
        return (image,)

    height, width = image.shape[1:3]
    patch = diffusion_model.patch
    tiles = _tiles(width, height, core_width, core_height, padding, patch, tile_grid)
    tiled = len(tiles) > 1
    template = clip.tokenizer.llama_template.replace("{}", VISION_BLOCK + "\n{}", 1)
    prompt = prompt.strip() or "Preserve the source composition, colors and identity. Refine fine detail."
    features = diffusion_model.txtlayers * diffusion_model.txtdim
    outputs = []

    # Each image needs its own vision contexts. Encode once per tile before sampling.
    for source_frame, frame in zip(source.split(1), image.split(1)):
        contexts = []
        for tile in tiles:
            throw_exception_if_processing_interrupted()
            crop = _source_crop(source_frame, tile, width, height)
            contexts.append(_encode_tile(clip, template, prompt, crop, features))
        frame = frame[..., :3]
        latent = vae.encode_tiled(frame) if tiled else vae.encode(frame)
        if latent.shape[-2:] != (height // LATENT_SCALE, width // LATENT_SCALE):
            raise ValueError("Krea2 Vision requires the Krea 2 compatible Qwen Image VAE (8x spatial compression).")

        frame_model = model
        if tiled:
            frame_model = model.clone()
            wrapper = TiledVision(tiles, contexts, patch, *latent.shape[-2:])
            frame_model.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, CANVAS_ORIGIN, wrapper)
            frame_model.set_model_patch(_position_on_canvas, "post_input")
        # The first context gives Comfy the correct conditioning shape and is used
        # directly when the entire canvas fits in one tile.
        positive = [[contexts[0], {}]]
        sampled = common_ksampler(
            model=frame_model, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name,
            scheduler=scheduler, positive=positive, negative=negative,
            latent={"samples": latent}, denoise=denoise,
        )[0]["samples"]
        decoded = vae.decode_tiled(sampled, overlap=16) if tiled else vae.decode(sampled)
        outputs.append(decoded.reshape(-1, *decoded.shape[-3:]))

    return (torch.cat(outputs),)
