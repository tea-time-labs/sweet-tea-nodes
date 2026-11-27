import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_STORED

root = Path("/content/st-krea-3x-padding")
prompts = {
    "station": (
        "An architectural photograph of a quiet angular concrete train station. The camera is low near the left side, "
        "looking diagonally down the platform toward a single red ticket kiosk on the far right. "
        "Overhead steel trusses and a thin teal handrail converge toward the same vanishing point. "
        "Wide grey paving in the foreground, strong clean geometry, slanted morning sunlight, fine concrete pores."
    ),
    "portrait": (
        "A natural close portrait on film of two women sitting back to back, one with dark curly hair and a small silver "
        "earring in the middle left, the other with straight auburn hair on the right. The closer woman's face is in profile. "
        "A thin strand of hair crosses her cheek, layered grey knit fabric, a quietly lit olive background. "
        "Restrained fine skin texture, soft realistic light, distinct faces with no other people."
    ),
    "cafe": (
        "An editorial lifestyle photograph for a small independent coffee company. A barista in a dark blue apron "
        "hands a white ceramic latte cup across the counter to a customer in a rust-colored cardigan. Both people are "
        "visible in three-quarter profile, their hands meeting around the cup near the center. A second green cup "
        "rests on a saucer in the foreground, pastry case and soft window light in the background, natural expressions, "
        "realistic ceramics, subtle knit texture, warm neutral colors with space at the upper left."
    ),
    "hikers": (
        "An outdoor apparel campaign photograph of two adult hikers helping each other cross a shallow mountain stream. "
        "One hiker in a bright red shell stands on a rock at center left and holds the hand of a second hiker in a blue "
        "shell stepping up from the lower right. The joined hands and loose backpack straps are clearly visible. An "
        "angled log and rippling water lead into a soft open mountain backdrop, natural afternoon light and wet stone textures."
    ),
    "illustration": (
        "A print-ready children's-book cover illustration in hand-printed indigo, moss green and golden yellow. Two "
        "children in rain boots play fetch with a round yellow ball and one small terrier in a park. The child at the "
        "left has just tossed the ball toward the dog at the center right, and the second child reaches for the dog's "
        "lead at the lower right. Bold flat color shapes, thin ink outlines, visible paper grain, open light sky for "
        "title placement, no printed text."
    ),
    "wedding": (
        "A candid wedding photograph of a newly married couple dancing in a softly lit reception hall. The bride in "
        "a cream dress and the groom in a charcoal suit face one another, arms linked at center, fabric folds catching "
        "warm window light. Two softly defocused relatives applaud in the far background, behind the couple. "
        "Realistic skin and hand anatomy, floorboards receding diagonally, believable fabric, warm documentary color."
    ),
}
# The original queue was created before the prompt categories were updated.
aliases = {"still_life": "cafe", "glacier": "hikers", "owl": "wedding"}
parser = argparse.ArgumentParser()
parser.add_argument("cases", nargs="*", choices=tuple(prompts) + tuple(aliases))
parser.add_argument("--modes", default="vision,depth,canny")
args = parser.parse_args()
modes = args.modes.split(",")
assert all(m in ("vision", "depth", "canny") for m in modes)
batches = root / "batches"
batches.mkdir(exist_ok=True)
successful = []
failed = []

def execute(script, env):
    return subprocess.run([sys.executable, "-u", str(root/script)], env=env).returncode

if args.cases and args.cases[0] == "still_life":
    print("Starting portrait overlap diagnostic", flush=True)
    owner_status = execute("test_overlap_owner.py", os.environ.copy())
    print("Portrait overlap diagnostic exit", owner_status, flush=True)
    print("Starting portrait core-caption diagnostic", flush=True)
    core_status = execute("test_core_captions.py", os.environ.copy())
    print("Portrait core-caption diagnostic exit", core_status, flush=True)

for case_id in args.cases or prompts:
    case = aliases.get(case_id, case_id)
    print("PROMPT START", case, flush=True)
    env = {**os.environ, "ST_KREA_PROMPT_CASE": case, "ST_KREA_SOURCE_PROMPT": prompts[case]}
    if case in ("cafe", "hikers", "illustration", "wedding"):
        env["ST_KREA_CAPTION_SCOPE"] = "core"
    folder = root / "results" / case
    for mode in modes:
        print("MODE START", case, mode, flush=True)
        if mode == "vision":
            code = execute("experiment.py", {**env, "ST_KREA_EXPERIMENT_RUNS": "p256_grounded_local"})
        else:
            code = execute("test_controls.py", {**env, "ST_KREA_CONTROL_MODE": mode})
            if code:
                execute("recover_control.py", env)
        print("MODE EXIT", case, mode, code, flush=True)
    if (folder/"report.json").exists():
        report = json.loads((folder/"report.json").read_text())
        keys = ["p256_grounded_local" + ("" if m == "vision" else f"_{m}") for m in modes]
        missing = [k for k in keys if report["runs"].get(k, {}).get("status") != "ok"]
        if missing:
            failed.append((case, missing))
        else:
            successful.append(case)
        work = batches / f"{case}.partial"
        with ZipFile(work, "w", ZIP_STORED) as archive:
            for file in folder.iterdir():
                if file.suffix in (".png", ".txt", ".json"):
                    archive.write(file, arcname=f"{case}/{file.name}")
        work.replace(batches / f"{case}.zip")
        print("PROMPT SAVED", case, "missing", missing, flush=True)
    else:
        failed.append((case, ["report"]))
print("BATTERY END", "complete", successful, "incomplete", failed, flush=True)
raise SystemExit(bool(failed))
