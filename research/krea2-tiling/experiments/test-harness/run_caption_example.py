"""Rerender one case with the same settings and selected edited captions."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import run_matrix as matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    args = parser.parse_args()
    baseline = json.loads((args.source_dir / "report.json").read_text())
    source_file = args.source_dir / "source.png"
    case = baseline["scene"]["id"]
    edits = json.loads((matrix.HERE / f"{case}-caption-edits.json").read_text())
    tiles = baseline["geometry"]["tiles"]
    captions = [edits.get(str(tile["tile"]), tile["caption"]) for tile in tiles]
    out = matrix.ROOT / "caption-results" / case
    out.mkdir(exist_ok=True, parents=True)
    model, clip, vae = matrix.load_runtime()
    with matrix.Image.open(source_file) as image:
        source = matrix.to_tensor(image)
    print("CAPTION RERENDER START", case, flush=True)

    def on_prepared(details):
        expected = [{**tile, "caption": caption} for tile, caption in zip(tiles, captions, strict=True)]
        assert json.loads(json.dumps(details["tiles"])) == expected

    def on_render(variant, image, details):
        target, report = out / "edited-captions.png", out / "report.json"
        matrix.save_png(image, target)
        matrix.save_json(report, {
            **details, "case": case, "status": "ok", "file": target.name,
            "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
            "seed": baseline["upscale_seed"], "output_size": baseline["geometry"]["final_size"],
            "edited_tiles": [{"tile": tile["tile"], "before": tile["caption"], "after": caption}
                             for tile, caption in zip(tiles, captions, strict=True) if caption != tile["caption"]],
            "colab_gpu": matrix.torch.cuda.get_device_name(0),
        })
        archive = matrix.ROOT / "bundles" / f"captions-{case}.zip"
        temp = archive.with_suffix(".tmp")
        with ZipFile(temp, "w", compression=ZIP_STORED) as z:
            for path in (report, target):
                z.write(path, f"{case}/{path.name}")
        temp.replace(archive)

    node = matrix.NODE_CLASS_MAPPINGS["STKrea2TurboMatrixTest"]()
    node.upscale(source, model, clip, vae, baseline["scene"]["prompt"], baseline["upscale_seed"],
                 variant="euler-simple", reference_captions=captions,
                 on_prepared=on_prepared, on_render=on_render)
    print("CAPTION RERENDER COMPLETE", case, flush=True)
