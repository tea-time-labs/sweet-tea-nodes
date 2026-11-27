"""Stage the archived experiments and input images on a fresh Colab runtime."""
import argparse
import json
from pathlib import Path
import shutil

LAB = Path(__file__).resolve().parent


def stage(destination):
    matrix = destination / "st-krea16"
    padding = destination / "st-krea-3x-padding"
    if matrix.exists() or padding.exists():
        raise SystemExit("These replay folders already exist. Use a fresh runtime or a fresh staging directory.")
    matrix.mkdir(parents=True)
    padding.mkdir()
    for name in ("matrix-as-executed", "test-harness", "caption-methods", "boundary-study"):
        shutil.copytree(LAB / "experiments" / name, matrix / name)
    shutil.copytree(LAB / "experiments/source-overlap", matrix / "trial")
    shutil.copytree(LAB / "inputs/matrix", matrix / "results")
    shutil.copytree(LAB / "inputs/padding/portrait", matrix / "results/portrait")
    # The two original runners expect different directories for the same pinned assets.
    for name in ("ComfyUI", "weights", "sweet-tea-nodes"):
        (padding / name).symlink_to(matrix / name, target_is_directory=True)
    for path in (LAB / "experiments/padding").iterdir():
        shutil.copy2(path, padding / path.name)
    shutil.copy2(LAB / "experiments/portrait-coordination/trace_portrait.py", padding)
    for source in sorted((LAB / "inputs/padding").iterdir()):
        target = padding / "results"
        if source.name != "waterfront":
            target = target / source.name
        shutil.copytree(source, target, dirs_exist_ok=True)
        # Archived reports are kept intact in the repository. Several GPU scripts
        # skip completed run IDs even when their output image is absent.
        path = target / "report.json"
        report = json.loads(path.read_text())
        report["runs"] = {}
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Staged the two replay layouts under {destination}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("/content"),
                        help="Use a temporary folder for an offline layout check. GPU scripts expect /content.")
    stage(parser.parse_args().destination)
