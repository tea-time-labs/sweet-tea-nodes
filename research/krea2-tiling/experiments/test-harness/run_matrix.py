"""Run the registered test node using ComfyUI's native model loaders and sampler."""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from zipfile import ZIP_STORED, ZipFile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CASES = json.loads((HERE / "scenes.json").read_text())


def save_json(path, payload):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def save_png(image, path):
    temp = path.with_suffix(".tmp")
    to_pil(image).save(temp, format="PNG", compress_level=2)
    temp.replace(path)


def pack_case(case_dir):
    bundles = ROOT / "bundles"
    bundles.mkdir(exist_ok=True)
    final = bundles / f"{case_dir.name}.zip"
    temp = final.with_suffix(".tmp")
    with ZipFile(temp, "w", compression=ZIP_STORED) as archive:
        for path in sorted(case_dir.iterdir()):
            if not path.name.endswith(".tmp"):
                archive.write(path, f"{case_dir.name}/{path.name}")
    temp.replace(final)


def run_case(index, case, model, clip, vae, variants):
    case_dir = ROOT / "results" / case["id"]
    case_dir.mkdir(exist_ok=True, parents=True)
    report_path, source_path = case_dir / "report.json", case_dir / "source.png"
    width, height, original_prompt = case["width"], case["height"], case["prompt"]
    source_seed, upscale_seed = 87923142 + 191 * index, 40309561 + 131 * index
    source_spec = {"steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "denoise": 1.0}
    spec = {"steps": 8, "cfg": 1.0, "denoise": .8, "positions": "local", "control": "Vision"}
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        "scene": case, "source_size": [width, height],
        "source_seed": source_seed, "source_sampler": source_spec, "upscale_seed": upscale_seed,
        "upscale_settings": spec, "runs": {}, "node": "STKrea2TurboMatrixTest",
        "colab_gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
        "comfyui": "7c78b5641ab2d6cb7f7766a183a8a43a59de0a8b",
        "st_nodes": "8c97fe110adb6e3a1c92976c2fd7f3d82c2e0935",
        "weights": "Comfy-Org/Krea-2@588feaf955b08b9ac4723eeb751752ab045dba71",
    }
    (case_dir / "original-prompt.txt").write_text(original_prompt + "\n")
    print("CASE START", case["id"], width, height, flush=True)
    if source_path.exists():
        with Image.open(source_path) as image:
            source = to_tensor(image)
    else:
        cond, = nodes.CLIPTextEncode().encode(clip, original_prompt)
        latent, = nodes.EmptyLatentImage().generate(width, height, 1)
        (sampled,) = nodes.common_ksampler(model, source_seed, 8, 1.0, "euler", "simple",
                                          cond, cond, latent)
        mm.unload_all_models()
        with torch.inference_mode():
            source, = nodes.VAEDecode().decode(vae, sampled)
        if source.shape[1:3] != (height, width) or not torch.isfinite(source).all():
            raise RuntimeError(f"Wrong source shape/values: {tuple(source.shape)}")
        save_png(source, source_path)
        with Image.open(source_path) as image:
            source = to_tensor(image)
        print("SOURCE SAVED", case["id"], flush=True)
        save_json(report_path, report)
    pending = tuple(name for name in variants if report["runs"].get(name, {}).get("status") != "ok"
                    or not (case_dir / f"{name}.png").exists())

    def on_prepared(details):
        report["geometry"] = details
        save_json(report_path, report)
        print("PREPARED", case["id"], details["working_canvas"],
              details["caption_and_encode_seconds"], flush=True)

    def on_render(name, image, details):
        target = case_dir / f"{name}.png"
        save_png(image, target)
        report["runs"][name] = {**details, "status": "ok", "file": target.name}
        save_json(report_path, report)
        pack_case(case_dir)
        print("RENDER SAVED", case["id"], name, details["seconds"], flush=True)

    if pending:
        instance = NODE_CLASS_MAPPINGS["STKrea2TurboMatrixTest"]()
        instance.upscale(source, model, clip, vae, original_prompt, upscale_seed, variants=pending,
                         on_prepared=on_prepared, on_render=on_render)
    pack_case(case_dir)
    print("CASE COMPLETE", case["id"], list(report["runs"]), flush=True)


def load_runtime():
    global mm, nodes, torch, Image, NODE_CLASS_MAPPINGS, MATRIX_VARIANTS, to_pil, to_tensor
    comfy_path = ROOT / "ComfyUI"
    os.chdir(comfy_path)
    sys.path[:0] = [str(comfy_path), str(ROOT / "sweet-tea-nodes")]
    sys.argv[:] = [sys.argv[0], "--use-pytorch-cross-attention"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.model_management as mm
    import folder_paths
    import nodes
    import torch
    from PIL import Image
    from krea16_test_node import NODE_CLASS_MAPPINGS, MATRIX_VARIANTS, to_pil, to_tensor

    torch.set_grad_enabled(False)
    nodes.NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS)
    for model_type in ("diffusion_models", "text_encoders", "vae"):
        folder_paths.add_model_folder_path(model_type, str(ROOT / "weights" / model_type), is_default=True)
    model, = nodes.UNETLoader().load_unet("krea2_turbo_fp8_scaled.safetensors", "default")
    clip, = nodes.CLIPLoader().load_clip("qwen3vl_4b_fp8_scaled.safetensors", "krea2", "default")
    vae, = nodes.VAELoader().load_vae("qwen_image_vae.safetensors")
    return model, clip, vae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int, default=len(CASES))
    parser.add_argument("--variants", nargs="+")
    args = parser.parse_args()
    if not 1 <= args.first <= args.last <= len(CASES):
        parser.error(f"Expected case bounds within 1–{len(CASES)}")
    model, clip, vae = load_runtime()
    if args.variants and any(name not in MATRIX_VARIANTS for name in args.variants):
        parser.error(f"Choose variants from {MATRIX_VARIANTS}")
    passed, failed = [], []
    for i in range(args.first, args.last + 1):
        case = CASES[i - 1]
        started = time.monotonic()
        try:
            run_case(i, case, model, clip, vae, args.variants or MATRIX_VARIANTS)
            passed.append(case["id"])
        except Exception:
            traceback.print_exc()
            failed.append(case["id"])
            print("CASE FAILED", case["id"], flush=True)
        finally:
            mm.unload_all_models()
            gc.collect()
            mm.soft_empty_cache()
            print("CASE ELAPSED", case["id"], round(time.monotonic() - started, 2), flush=True)
    print("MATRIX COMPLETE", "passed", passed, "failed", failed, flush=True)
    sys.exit(bool(failed))
