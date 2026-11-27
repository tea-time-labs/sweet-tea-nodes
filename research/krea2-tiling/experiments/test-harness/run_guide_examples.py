"""Test the two optional guides on an unchanged source from the sampler matrix."""

import argparse
import hashlib
import json
from zipfile import ZIP_STORED, ZipFile

import run_matrix as matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=[scene["id"] for scene in matrix.CASES])
    args = parser.parse_args()
    baseline = matrix.ROOT / "results" / args.case
    original = json.loads((baseline / "report.json").read_text())
    source_path = baseline / "source.png"
    report_path = matrix.ROOT / "guide-results" / args.case / "report.json"
    out = report_path.parent
    out.mkdir(exist_ok=True, parents=True)
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        "case": args.case, "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_size": original["source_size"], "output_size": original["geometry"]["final_size"],
        "seed": original["upscale_seed"], "sampler": "euler", "scheduler": "simple",
        "steps": 8, "cfg": 1.0, "denoise": .8, "guide_strength": .7, "runs": {},
    }
    model, clip, vae = matrix.load_runtime()
    report["colab_gpu"] = matrix.torch.cuda.get_device_name(0)
    captions = [tile["caption"] for tile in original["geometry"]["tiles"]]
    with matrix.Image.open(source_path) as image:
        source = matrix.to_tensor(image)
    for mode in ("Depth", "Canny"):
        name = mode.lower()
        target = out / f"vision-{name}.png"
        guide = out / f"{name}-guide.png"
        if report["runs"].get(name, {}).get("status") == "ok" and target.exists() and guide.exists():
            continue
        print("GUIDE START", args.case, mode, flush=True)

        def on_prepared(details):
            rows = json.loads(json.dumps(details["tiles"]))
            assert rows == original["geometry"]["tiles"], "Reference tile crops or captions changed"

        def on_guide(image):
            matrix.save_png(image, guide)

        def on_render(variant, image, details):
            matrix.save_png(image, target)
            report["runs"][name] = {"status": "ok", "file": target.name, "guide": guide.name, **details}
            matrix.save_json(report_path, report)
            print("GUIDE SAVED", args.case, name, details, flush=True)

        node = matrix.NODE_CLASS_MAPPINGS["STKrea2TurboMatrixTest"]()
        node.upscale(source, model, clip, vae, original["scene"]["prompt"], original["upscale_seed"],
                     variant="euler-simple", control=mode, reference_captions=captions,
                     on_prepared=on_prepared, on_render=on_render, on_guide=on_guide)
        matrix.mm.unload_all_models()
        matrix.mm.soft_empty_cache()
    archive = matrix.ROOT / "bundles" / f"guides-{args.case}.zip"
    temp = archive.with_suffix(".tmp")
    with ZipFile(temp, "w", compression=ZIP_STORED) as z:
        for path in sorted(out.iterdir()):
            z.write(path, f"{args.case}/{path.name}")
    temp.replace(archive)
    print("GUIDES COMPLETE", args.case, flush=True)
