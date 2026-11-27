"""Run a caption experiment against the archived matched source PNGs and their exact geometry."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from caption_methods import METHODS, describe

REFERENCE = "original_t4"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def execute(split, methods, all_tiles):
    comfy = ROOT / "ComfyUI"
    os.chdir(comfy)
    sys.path[:0] = [str(comfy), str(ROOT / "sweet-tea-nodes"), str(ROOT / "test-harness")]
    sys.argv[:] = [sys.argv[0], "--use-pytorch-cross-attention"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import folder_paths
    import nodes
    import numpy as np
    from PIL import Image
    import torch
    torch.set_grad_enabled(False)
    folder_paths.add_model_folder_path("text_encoders", str(ROOT / "weights/text_encoders"), is_default=True)
    clip, = nodes.CLIPLoader().load_clip("qwen3vl_4b_fp8_scaled.safetensors", "krea2", "default")
    selected = json.loads((HERE / "protocol.json").read_text())[split]
    method_hash = hashlib.sha256((HERE / "caption_methods.py").read_bytes()).hexdigest()
    for case, tiles in selected.items():
        folder = ROOT / "results" / case
        source_file = folder / "source.png"
        report = json.loads((folder / "report.json").read_text())
        with Image.open(source_file) as image:
            source = torch.from_numpy(np.array(image.convert("RGB"))).unsqueeze(0).float() / 255
        boxes = report["geometry"]["tiles"]
        assert [x["tile"] for x in boxes] == list(range(1, 10))
        for method in methods:
            out = HERE / "outputs" / method / case
            out.mkdir(parents=True, exist_ok=True)
            for i in (range(1, 10) if all_tiles else tiles):
                path = out / f"{i}.json"
                source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
                if path.exists():
                    record = json.loads(path.read_text())
                    if record["method_sha256"] != method_hash or record["source_sha256"] != source_hash:
                        raise RuntimeError(f"Refusing to mix changed inputs or methods into {path}")
                    continue
                tile = boxes[i-1]
                started = time.monotonic()
                if method == REFERENCE:
                    from krea16_test_node import describe_cores
                    caption, = describe_cores(source, report["scene"]["prompt"], clip, [tile])
                    notes = []
                else:
                    caption, notes = describe(method, clip, source, report["scene"]["prompt"], tile)
                torch.cuda.synchronize()
                record = dict(method=method, case=case, tile=i, core=tile["core_source_xyxy"],
                              padded=tile["padded_source_xyxy"], caption=caption, notes=notes,
                              seconds=round(time.monotonic()-started, 2), gpu=torch.cuda.get_device_name(0),
                              source_sha256=source_hash, method_sha256=method_hash)
                temp = path.with_suffix(".tmp")
                temp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
                temp.replace(path)
                print(case, i, method, record["seconds"], flush=True)
    print("CAPTION RUN COMPLETE", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("split", choices=("development","confirmation"))
    ap.add_argument("methods", nargs="+", choices=(*METHODS, REFERENCE))
    ap.add_argument("--all-tiles", action="store_true")
    args = ap.parse_args()
    execute(args.split, args.methods, args.all_tiles)
