import os
import time

os.environ["ST_KREA_EXPERIMENT_RUNS"] = "decode_only"
import experiment as exp
from experiment import torch

for kind in ("depth", "canny"):
    case = f"p256_grounded_local_{kind}"
    latent_path = exp.out / f"{case}.latent.pt"
    if exp.report["runs"].get(case, {}).get("status") == "ok" or not latent_path.is_file():
        continue
    try:
        # A fresh process has no quantized diffusion model to unpatch while
        # loading the unquantized VAE in its required inference context.
        sample = torch.load(latent_path, map_location="cpu", weights_only=True)
        started = time.monotonic()
        with torch.inference_mode():
            pixels = exp.vae.decode_tiled(sample, overlap=16)
        pixels = pixels.reshape(-1, *pixels.shape[-3:])
        assert tuple(pixels.shape) == (1, 3072, 3072, 3)
        assert torch.isfinite(pixels).all()
        mask_config = {"thresholds": [.1, .4]} if kind == "canny" else {"estimator": "Depth Anything V2 Large", "near_is_white": True}
        result = {"status": "ok", "padding_px": 256, "captions": "grounded", "positions": "local",
                  "largest_tile_px": 1536, "control": kind, "control_lora_strength": .7,
                  "control_guide": mask_config, "recovered_latent": True,
                  "decode_seconds": round(time.monotonic()-started, 1), **exp.summarize(pixels)}
        exp.png(pixels, exp.out / f"{case}.png")
        exp.report["runs"][case] = result
        exp.write_report()
        latent_path.unlink()
        exp.announce("RECOVERED CONTROL RESULT", result)
    except Exception as exc:
        exp.announce("Control recovery failed", kind, repr(exc))
        raise
