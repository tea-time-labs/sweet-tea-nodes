"""Check the more widely spaced Beta schedule against the same Vision settings."""

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
    out = matrix.ROOT / "beta-results" / args.case
    out.mkdir(exist_ok=True, parents=True)
    model, clip, vae = matrix.load_runtime()
    captions = [tile["caption"] for tile in original["geometry"]["tiles"]]
    with matrix.Image.open(source_path) as image:
        source = matrix.to_tensor(image)
    print("BETA START", args.case, flush=True)

    def on_prepared(details):
        rows = json.loads(json.dumps(details["tiles"]))
        assert rows == original["geometry"]["tiles"], "Reference tile crops or captions changed"

    def on_render(variant, image, details):
        target = out / (variant + ".png")
        matrix.save_png(image, target)
        file = out / "report.json"
        matrix.save_json(file, {
            **details, "status": "ok", "file": target.name, "case": args.case,
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "source_size": original["source_size"], "output_size": original["geometry"]["final_size"],
            "seed": original["upscale_seed"], "steps": 8, "cfg": 1.0, "denoise": .8,
            "colab_gpu": matrix.torch.cuda.get_device_name(0),
        })
        archive = matrix.ROOT / "bundles" / f"beta-{args.case}.zip"
        temp = archive.with_suffix(".tmp")
        with ZipFile(temp, "w", compression=ZIP_STORED) as z:
            for path in (file, target):
                z.write(path, f"{args.case}/{path.name}")
        temp.replace(archive)

    node = matrix.NODE_CLASS_MAPPINGS["STKrea2TurboMatrixTest"]()
    node.upscale(source, model, clip, vae, original["scene"]["prompt"], original["upscale_seed"],
                 variant="euler-beta", reference_captions=captions, on_prepared=on_prepared,
                 on_render=on_render)
    print("BETA COMPLETE", args.case, flush=True)
