import os
import time
from pathlib import Path
import subprocess
import sys

recover = subprocess.run([sys.executable, str(Path(__file__).with_name("recover_control.py"))], check=False)
print("Control recovery process exited", recover.returncode, flush=True)
from unittest.mock import patch

os.environ["ST_KREA_EXPERIMENT_RUNS"] = "sequential_only"
import experiment as exp
from experiment import torch, vision

import comfy.sample
import comfy.utils

name = "p256_grounded_local_sequential"
if exp.report["runs"].get(name, {}).get("status") == "ok":
    raise SystemExit(0)

padding = 256
canvas_size = 3072
model, source, vae, clip = exp.model, exp.source, exp.vae, exp.clip
tiles = vision._tiles(canvas_size, canvas_size, 1024, 1024, padding, 2, exp.layout)
captions = exp.tile_captions(padding)
x_edges, y_edges = exp.layout
captions_template = clip.tokenizer.llama_template.replace("{}", vision.VISION_BLOCK + "\n{}", 1)
diffusion = model.get_model_object("diffusion_model")
features = diffusion.txtlayers * diffusion.txtdim
exp.refresh()
torch.cuda.reset_peak_memory_stats()
started = time.monotonic()
contexts = [
    [[vision._encode_tile(
        clip, captions_template, caption, vision._source_crop(source, tile, canvas_size, canvas_size), features
    ), {}]]
    for tile, caption in zip(tiles, captions)
]
upscaled = comfy.utils.common_upscale(source.movedim(-1, 1), canvas_size, canvas_size, "lanczos", "disabled").movedim(1, -1)
canvas = vae.encode_tiled(upscaled).clone()
noise = comfy.sample.prepare_noise(canvas, 40309561)
committed = torch.zeros(canvas.shape[-2:], dtype=torch.bool, device=canvas.device)
# Start with the tile that owns the person, then move out through adjacent cores.
order = [5, 4, 2, 8, 1, 7, 3, 0, 6]
stats = []
snapshot = exp.out / f"{name}.latent.pt"

def keep_finished_latent_clean(sigma, noise, latent_image, **kwargs):
    # Comfy normally re-noises sampled areas used as inpaint context.
    # Supply the already-finished neighbor itself at every evaluation.
    return latent_image

if snapshot.exists():
    saved = torch.load(snapshot, map_location="cpu", weights_only=True)
    canvas = saved["canvas"]
    stats = saved["stats"]
else:
    with patch.object(model.model, "scale_latent_inpaint", keep_finished_latent_clean):
        for index in order:
            tile = tiles[index]
            row, col = divmod(index, 3)
            x1, x2 = [v // 8 for v in x_edges[col:col+2]]
            y1, y2 = [v // 8 for v in y_edges[row:row+2]]
            known = committed[tile.top:tile.bottom, tile.left:tile.right]
            fraction = float(known.float().mean())
            mask = None if not known.any() else (~known)[None, :, :].float()
            exp.announce("Sequential tile", index+1, "finished context fraction", round(fraction, 3))
            initial = canvas[tile.region].contiguous()
            result = comfy.sample.sample(model, noise[tile.region].contiguous(), 8, 1., "euler", "simple",
                                         contexts[index], [], initial, denoise=.8,
                                         noise_mask=mask, seed=40309561)
            core = result[..., y1-tile.top:y2-tile.top, x1-tile.left:x2-tile.left]
            assert not committed[y1:y2, x1:x2].any()
            canvas[..., y1:y2, x1:x2] = core.to(canvas.device)
            committed[y1:y2, x1:x2] = True
            stats.append({"tile": index+1, "finished_context_fraction": round(fraction, 4)})
        assert committed.all()
    torch.save({"canvas": canvas.cpu(), "stats": stats}, snapshot)
with torch.inference_mode():
    result = vae.decode_tiled(canvas, overlap=16)
image = result.reshape(-1, *result.shape[-3:])
torch.cuda.synchronize()
assert tuple(image.shape) == (1, 3072, 3072, 3)
summary = {"status": "ok", "padding_px": padding, "captions": "grounded", "positions": "local",
           "largest_tile_px": 1536, "seconds": round(time.monotonic()-started, 1),
           "peak_torch_allocated_mib": round(torch.cuda.max_memory_allocated()/2**20),
           "peak_torch_reserved_mib": round(torch.cuda.max_memory_reserved()/2**20),
           "tiling": "sequential, commit core, use clean finished context", "sequence": stats,
           **exp.summarize(image)}
exp.png(image, exp.out / f"{name}.png")
exp.report["runs"][name] = summary
exp.write_report()
snapshot.unlink(missing_ok=True)
exp.announce("SEQUENTIAL RESULT", summary)
