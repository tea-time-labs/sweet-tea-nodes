"""Replay the archived portrait with temporal ownership of its eye overlap.

Run beside the archived experiment.py and test_overlap_owner.py on the pinned
Colab runtime. Capture both tiles' flow predictions on the identical noisy
window, and check our conversion against ComfyUI's own denoised callback.
"""
from pathlib import Path
import ast
import json
import os
from unittest.mock import patch

root = Path("/content/st-krea-3x-padding")
portrait = root / "results/portrait"
os.environ["ST_KREA_PROMPT_CASE"] = "portrait"
os.environ["ST_KREA_SOURCE_PROMPT"] = (portrait / "original-prompt.txt").read_text().strip()
os.environ["ST_KREA_EXPERIMENT_RUNS"] = "timing_probe"
import experiment as exp
import latent_preview
from dataclasses import replace
from experiment import vision, torch

# Execute just the archived owner class with its exact original feather.
native = vision.TiledVision
module = ast.parse((Path(__file__).parent / "test_overlap_owner.py").read_text())
declaration = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "OwnedOverlap")
scope = dict(NativeVision=native, source_box=(0, 269, 265, 424), owner_index=3,
             torch=torch, replace=replace)
exec(compile(ast.Module(body=[declaration], type_ignores=[]), "archived_owner", "exec"), scope)
Owned = scope["OwnedOverlap"]
model_sampling = exp.model.get_model_object("model_sampling")
# Same global latent rectangle inside both tiles. Offsets from the final
# output are x=0 and y=768 pixels. This includes both alternative eye sites.
top, bottom, left, right = 96, 160, 0, 112
window = (..., slice(top, bottom), slice(left, right))
prepare = latent_preview.prepare_callback


def run(mode, owner_steps):
    out = portrait / mode
    out.mkdir(exist_ok=True)
    records = []
    pending = {}
    snapshots = []

    class Timed:
        def __init__(self, tiles, contexts, patch_size, h, w):
            self.blend = native(tiles, contexts, patch_size, h, w)
            self.owner = Owned(tiles, contexts, patch_size, h, w)
            self.step = 0

        def __call__(self, executor, x, timesteps, *args, **kwargs):
            self.step += 1
            selected = self.owner if self.step in owner_steps else self.blend
            calls = 0

            def capture(*inputs, **options):
                nonlocal calls
                pred = executor(*inputs, **options)
                tile = selected.tiles[calls]
                calls += 1
                if calls in (1, 4):
                    local = (..., slice(top - tile.top, bottom - tile.top),
                             slice(left - tile.left, right - tile.left))
                    pending[str(calls)] = pred[local].detach().float().cpu()
                return pred

            pred = selected(capture, x, timesteps, *args, **kwargs)
            assert calls == 9 and timesteps.numel() == 1
            pending["merged"] = pred[window].detach().float().cpu()
            pending["sigma"] = float(timesteps.item())
            return pred

    def traced_callback(model, steps, *args, **kwargs):
        default = prepare(model, steps, *args, **kwargs)

        def callback(i, denoised, noisy, total):
            assert total == 8 and i == len(records)
            x = noisy[window].detach().float().cpu()
            actual = denoised[window].detach().float().cpu()
            # CFG=1, and Krea2 uses the flow parameterization.
            sigma = torch.tensor([pending["sigma"]])
            previews = {k: model_sampling.calculate_denoised(sigma, pending[k], x)
                        for k in ("1", "4", "merged")}
            error = float((previews["merged"] - actual).abs().max())
            records.append(dict(step=i+1, sigma=float(sigma.item()),
                                owner=i+1 in owner_steps, callback_max_error=error))
            snapshots.append(previews)
            pending.clear()
            if default is not None:
                return default(i, denoised, noisy, total)
        return callback

    with patch.object(vision, "TiledVision", Timed), patch.object(latent_preview, "prepare_callback", traced_callback):
        exp.run(mode, 256, True, "local")
    assert exp.report["runs"][mode]["status"] == "ok" and len(records) == 8
    assert max(item["callback_max_error"] for item in records) < .001, records
    exp.mm.unload_all_models()
    process_out = exp.model.model.process_latent_out
    with torch.inference_mode():
        for i, views in enumerate(snapshots):
            for key, view in views.items():
                decoded = exp.vae.decode(process_out(view))
                image = decoded.reshape(-1, *decoded.shape[-3:])
                assert image.shape[1:3] == ((bottom-top)*8, (right-left)*8)
                exp.png(image, out / f"step-{i+1}-{key}.png")
    (out / "steps.json").write_text(json.dumps(records, indent=2))
    print("TRACE COMPLETE", mode, flush=True)


run("owner_last_four", {5,6,7,8})
run("owner_first_four", {1,2,3,4})
