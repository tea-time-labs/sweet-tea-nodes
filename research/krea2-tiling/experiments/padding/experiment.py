import contextlib
import gc
import json
import os
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

root = Path("/content/st-krea-3x-padding")
out = root / "results"
if case := os.environ.get("ST_KREA_PROMPT_CASE"):
    out = out / case
out.mkdir(exist_ok=True, parents=True)
os.chdir(root / "ComfyUI")
sys.path[0:0] = [str(root / "ComfyUI"), str(root / "sweet-tea-nodes")]

import comfy.options
comfy.options.enable_args_parsing()
sys.argv[:] = [sys.argv[0], "--use-pytorch-cross-attention"]

import folder_paths
import nodes
import numpy as np
from PIL import Image, ImageDraw
import torch
import comfy.model_management as mm

torch.set_grad_enabled(False)
# Colab Torch 2.14 rejects wrapping Comfy's quantized weight tensors as
# Parameters inside inference_mode when they are moved between devices.
# Keep gradients off, using the exact body of the ST Vision sampler.
from usdu_pose_tiled import krea2_vision as vision
from usdu_pose_tiled.krea2_control import STKrea2UltimateUpscale, square_tile_grid

vision.upscale_with_vision = vision.upscale_with_vision.__wrapped__

for model_type in ("diffusion_models", "text_encoders", "vae"):
    folder_paths.add_model_folder_path(model_type, str(root / "weights" / model_type), is_default=True)

def announce(*args):
    print(*args, flush=True)

def pil(tensor):
    return Image.fromarray((tensor[0].detach().cpu().float().clamp(0, 1).numpy() * 255).round().astype(np.uint8))

def tensor(image):
    return torch.from_numpy(np.array(image.convert("RGB"))).unsqueeze(0).float() / 255

def png(tensor_image, path):
    pil(tensor_image).save(path, compress_level=2)

prompt = (
    "A quiet natural 35mm editorial photograph of an old waterfront promenade on a rainy morning. "
    "One woman in a mustard-yellow raincoat stands in the right half holding a single deep burgundy umbrella. "
    "The scene is wide and environmental, the woman and the umbrella are a quarter of the image height together. "
    "A continuous softly weathered dark iron railing runs diagonally from the lower left to the middle right. "
    "Beyond it the left half is open slate-green water and pearly fog, with subtle ripples. "
    "Wet textured stone pavement fills the foreground, muted cool tones, realistic fine fabric and metal detail."
)
prompt = os.environ.get("ST_KREA_SOURCE_PROMPT", prompt)
diffusion = "krea2_turbo_fp8_scaled.safetensors"
text_encoder = "qwen3vl_4b_fp8_scaled.safetensors"
model, = nodes.UNETLoader().load_unet(diffusion, "default")
clip, = nodes.CLIPLoader().load_clip(text_encoder, "krea2", "default")
vae, = nodes.VAELoader().load_vae("qwen_image_vae.safetensors")
source_file = out / "source.png"
if source_file.exists():
    source = tensor(Image.open(source_file))
else:
    text, = nodes.CLIPTextEncode().encode(clip, prompt)
    latent, = nodes.EmptyLatentImage().generate(1024, 1024, 1)
    sample, = nodes.common_ksampler(model, 87923142, 8, 1., "euler", "simple", text, text, latent)
    source, = nodes.VAEDecode().decode(vae, sample)
    png(source, source_file)
    source = tensor(Image.open(source_file))
(out / "original-prompt.txt").write_text(prompt + "\n")
announce("Source saved", tuple(source.shape))

fname = out / "report.json"
report = json.loads(fname.read_text()) if fname.exists() else {
    "gpu": torch.cuda.get_device_name(0), "gpu_total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20),
    "denoise": .8, "steps": 8, "cfg": 1, "scheduler": "simple", "sampler": "euler",
    "source_seed": 87923142, "upscale_seed": 40309561, "source_size": [1024, 1024],
    "canvas_size": [3072, 3072], "tile_grid": [3, 3],
    "torch": torch.__version__, "runs": {}, "captions": {}, "caption_seconds": {},
    "comfyui": "7c78b5641ab2d6cb7f7766a183a8a43a59de0a8b",
    "st_nodes": "8c97fe110adb6e3a1c92976c2fd7f3d82c2e0935",
    "weights": "Comfy-Org/Krea-2@588feaf955b08b9ac4723eeb751752ab045dba71",
}

def write_report():
    fname.write_text(json.dumps(report, indent=2) + "\n")

layout = square_tile_grid(3072, 3072, "9")
qwen_instruction = (
    "The first image is a wider reference. Its cyan outline locates the crop shown cleanly in the second image. "
    "Write one short descriptive sentence for the second image only, using 18 to 38 words. "
    "Describe the colors, textures, and shapes visible in the clean crop, including where they enter and leave the crop. "
    "For a partial subject describe only the visible fragment, without completing it. "
    "When the crop is empty or background, describe the background alone. "
    "Use the original prompt and the wider image only to clarify ambiguous visual evidence. "
    "Omit everything that is outside the clean crop or only named by the original prompt. "
    "Output the caption alone, in English. Original generation prompt follows:\n" + prompt
)

def tile_boxes(tile):
    left, top, right, bottom = (tile.left * 8, tile.top * 8, tile.right * 8, tile.bottom * 8)
    return left//3, top//3, (right+2)//3, (bottom+2)//3

def tile_captions(padding):
    k = str(padding)
    if k in report["captions"]:
        return [item["caption"] for item in report["captions"][k]]
    started = time.monotonic()
    c = []
    tiles = vision._tiles(3072, 3072, 1024, 1024, padding, 2, layout)
    if os.environ.get("ST_KREA_CAPTION_SCOPE") == "core":
        from core_captions import core_captions
        rows = core_captions(source, prompt, clip, layout, tiles)
        report["captions"][k] = rows
        report.setdefault("caption_seconds", {})[k] = round(time.monotonic()-started, 1)
        report["caption_scope"] = "core only, full scene and padded crop for context"
        write_report()
        return [item["caption"] for item in rows]
    for index, t in enumerate(tiles, 1):
        x1, y1, x2, y2 = tile_boxes(t)
        board = pil(source).resize((512, 512), Image.Resampling.LANCZOS)
        painter = ImageDraw.Draw(board)
        painter.rectangle((x1//2, y1//2, min(511,x2//2-1), min(511,y2//2-1)),
                          outline="#24faff", width=3)
        crop = source[:, y1:y2, x1:x2, :]
        tok = clip.tokenize(qwen_instruction, images=[tensor(board), crop], min_length=1, thinking=False)
        image_tokens = sum(isinstance(s[0], dict) and s[0].get("type") == "image"
                           for row in tok["qwen3vl_4b"] for s in row)
        assert image_tokens == 2, image_tokens
        response = clip.decode(clip.generate(tok, do_sample=False, max_length=96))
        caption = response.rsplit("</think>", 1)[-1].strip().strip('"').strip()
        caption = re.sub(r"^caption\s*:\s*", "", caption, flags=re.IGNORECASE)
        if not caption or "<think>" in caption:
            raise ValueError(f"Unusable Qwen caption on tile {index}: {response!r}")
        c.append({"tile": index, "source_box_xyxy": [x1,y1,x2,y2], "caption": caption})
        announce("Caption", padding, index, caption)
    report["captions"][k] = c
    report.setdefault("caption_seconds", {})[k] = round(time.monotonic()-started, 1)
    write_report()
    return [item["caption"] for item in c]

def refresh():
    gc.collect()
    mm.soft_empty_cache()
    torch.cuda.synchronize()

def summarize(image):
    work = pil(image)
    back = np.array(work.resize((1024,1024), Image.Resampling.BOX), dtype=np.float32)
    src = np.array(pil(source), dtype=np.float32)
    return {"source_downsample_mae_8bit": round(float(np.abs(back-src).mean()), 3)}

def run(name, padding, captioned, position):
    output = out / f"{name}.png"
    checkpoint = out / f"{name}.latent.pt"
    if output.exists() and report["runs"].get(name, {}).get("status") == "ok":
        return
    row = {"padding_px": padding, "captions": "grounded" if captioned else "shared",
           "positions": position, "largest_tile_px": 1024 + padding*2}
    try:
        if captioned:
            texts = iter(tile_captions(padding))
        encode = vision._encode_tile
        native_sample = vision.common_ksampler
        native_decode = vae.decode_tiled

        def sample_with_checkpoint(*args, **kwargs):
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                return ({"samples": saved},)
            result = native_sample(*args, **kwargs)
            work = checkpoint.with_suffix(".tmp")
            torch.save(result[0]["samples"].detach().cpu(), work)
            work.replace(checkpoint)
            return result

        def decode_in_inference_mode(*args, **kwargs):
            if os.environ.get("ST_KREA_CONTROL_MODE"):
                # Offload FP8 adapters before entering the VAE inference context.
                mm.unload_all_models()
            with torch.inference_mode():
                return native_decode(*args, **kwargs)

        def encode_local(clip, template, prompt_text, crop, features):
            return encode(clip, template, next(texts), crop, features)
        refresh()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        row["recovered_latent"] = checkpoint.exists()
        announce("Starting render", name)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(vision, "common_ksampler", sample_with_checkpoint))
            stack.enter_context(patch.object(vae, "decode_tiled", decode_in_inference_mode))
            if captioned:
                stack.enter_context(patch.object(vision, "_encode_tile", encode_local))
            if position == "local":
                stack.enter_context(patch.object(vision, "_position_on_canvas", lambda args: args))
            image, = STKrea2UltimateUpscale().upscale(
                source, model, [], [], vae, 3., 40309561, 8, 1., "euler", "simple",
                .8, 1024, padding, 0, "Vision", 0., "", "", "", 1024, .1, .4,
                clip=clip, vision_prompt=prompt, tile_count="9",
            )
        torch.cuda.synchronize()
        assert tuple(image.shape) == (1,3072,3072,3), image.shape
        assert torch.isfinite(image).all()
        row.update(status="ok", seconds=round(time.monotonic()-started, 1),
                   peak_torch_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20),
                   peak_torch_reserved_mib=round(torch.cuda.max_memory_reserved()/2**20),
                   **summarize(image))
        png(image, output)
        checkpoint.unlink(missing_ok=True)
    except torch.cuda.OutOfMemoryError as exc:
        row.update(status="out_of_memory", error=str(exc))
        mm.unload_all_models()
    report["runs"][name] = row
    announce("RESULT", name, row)
    write_report()

selected = os.environ.get("ST_KREA_EXPERIMENT_RUNS", "").split(",")
for spec in (
    ("p256_grounded_local", 256, True, "local"),
    ("p384_grounded_local", 384, True, "local"),
    ("p512_grounded_local", 512, True, "local"),
    ("p256_shared_canvas", 256, False, "canvas"),
    ("p256_grounded_canvas", 256, True, "canvas"),
    ("p256_shared_local", 256, False, "local"),
    ("p096_shared_canvas", 96, False, "canvas"),
):
    if not selected[0] or spec[0] in selected:
        run(*spec)
announce("Experiment completed", list(report["runs"]))
