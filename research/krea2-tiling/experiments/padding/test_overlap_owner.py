"""Test one known portrait overlap with the same nine padded tile predictions."""
from dataclasses import replace
import os
from pathlib import Path
from unittest.mock import patch

root = Path("/content/st-krea-3x-padding")
os.environ["ST_KREA_PROMPT_CASE"] = "portrait"
os.environ["ST_KREA_SOURCE_PROMPT"] = (root / "results/portrait/original-prompt.txt").read_text().strip()
os.environ["ST_KREA_EXPERIMENT_RUNS"] = "overlap_owner_only"
import experiment as exp
from experiment import torch, vision

NativeVision = vision.TiledVision
name = "p256_grounded_local_eye_owner"
owner_index = 3  # Middle-left tile contains the eye, nose and lower face.
source_box = (0, 269, 265, 424)  # x1, y1, x2, y2 in 1024 px source pixels.

class OwnedOverlap(NativeVision):
    def __init__(self, tiles, contexts, patch_size, height, width):
        owner = tiles[owner_index]
        y = (torch.arange(height).float() + .5) * 8/3
        x = (torch.arange(width).float() + .5) * 8/3
        left, top, right, bottom = source_box
        y_feather = torch.minimum((y - top)/17, (bottom-y)/17).clamp(0,1)
        x_feather = ((right-x)/24).clamp(0,1) * (x >= left)
        alpha = y_feather[:, None] * x_feather[None, :]
        owner_window = torch.zeros_like(alpha, dtype=torch.bool)
        owner_window[owner.region] = True
        assert not (alpha[~owner_window] > 0).any()
        adjusted = []
        for i, tile in enumerate(tiles):
            local_alpha = alpha[tile.region]
            weight = tile.weight * (1-local_alpha)
            if i == owner_index:
                weight += local_alpha
            adjusted.append(replace(tile, weight=weight))
        super().__init__(adjusted, contexts, patch_size, height, width)
        assert self.weights.min().item() > 0

def probe_single_tile_views():
    import comfy.sample
    import comfy.utils

    tile_list = vision._tiles(3072, 3072, 1024, 1024, 256, 2, exp.layout)
    if all((exp.out / f"isolated_tile_{i+1}.png").exists() for i in (0, 3)):
        return
    captions = exp.tile_captions(256)
    template = exp.clip.tokenizer.llama_template.replace("{}", vision.VISION_BLOCK + "\n{}", 1)
    diffusion = exp.model.get_model_object("diffusion_model")
    features = diffusion.txtlayers * diffusion.txtdim
    resized = comfy.utils.common_upscale(exp.source.movedim(-1, 1), 3072, 3072, "lanczos", "disabled").movedim(1, -1)
    latent = exp.vae.encode_tiled(resized)
    noise = comfy.sample.prepare_noise(latent, 40309561)
    tests = {}
    for i in (0, 3):
        tile = tile_list[i]
        filename = exp.out / f"isolated_tile_{i+1}.png"
        w = (tile.right-tile.left)*8
        h = (tile.bottom-tile.top)*8
        crop = vision._source_crop(exp.source, tile, 3072, 3072)
        embedding = vision._encode_tile(exp.clip, template, captions[i], crop, features)
        print("ISOLATED TILE START", i+1, w, h, flush=True)
        result = comfy.sample.sample(exp.model, noise[tile.region].contiguous(), 8, 1., "euler", "simple",
                                     [[embedding, {}]], [], latent[tile.region].contiguous(),
                                     denoise=.8, seed=40309561)
        with torch.inference_mode():
            pixels = exp.vae.decode_tiled(result, overlap=16)
        pixels = pixels.reshape(-1, *pixels.shape[-3:])
        assert tuple(pixels.shape) == (1, h, w, 3)
        exp.png(pixels, filename)
        tests[str(i+1)] = {"output_size": [w, h], "tile_box_xyxy":
                           [tile.left*8, tile.top*8, tile.right*8, tile.bottom*8]}
        print("ISOLATED TILE RESULT", i+1, [w,h], flush=True)
    exp.report["isolated_tile_probes"] = tests
    exp.write_report()

probe_single_tile_views()

with patch.object(vision, "TiledVision", OwnedOverlap):
    exp.run(name, 256, True, "local")
row = exp.report["runs"][name]
row["eye_overlap_owner_tile"] = owner_index+1
row["eye_source_box_xyxy"] = source_box
exp.write_report()
print("EYE OVERLAP RESULT", row, flush=True)
