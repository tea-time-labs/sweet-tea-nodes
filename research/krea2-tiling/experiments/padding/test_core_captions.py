"""Compare captioning the writable core with captioning the full padded crop."""
from pathlib import Path
import os
import time
from unittest.mock import patch

from PIL import Image, ImageDraw

root = Path("/content/st-krea-3x-padding")
os.environ["ST_KREA_PROMPT_CASE"] = "portrait"
os.environ["ST_KREA_SOURCE_PROMPT"] = (root / "results/portrait/original-prompt.txt").read_text().strip()
os.environ["ST_KREA_EXPERIMENT_RUNS"] = "core_caption_only"
import experiment as exp
from experiment import vision

name = "p256_grounded_core_caption"
if exp.report["runs"].get(name, {}).get("status") == "ok":
    raise SystemExit(0)
tiles = vision._tiles(3072, 3072, 1024, 1024, 256, 2, exp.layout)
from core_captions import core_captions

captions = exp.report.get("core_caption_probe")
if captions is None:
    start = time.monotonic()
    captions = core_captions(exp.source, exp.prompt, exp.clip, exp.layout, tiles)
    exp.report["core_caption_probe"] = captions
    exp.report["core_caption_seconds"] = round(time.monotonic()-start,1)
    exp.write_report()
with patch.object(exp, "tile_captions", lambda _: [x["caption"] for x in captions]):
    exp.run(name,256,True,"local")
row=exp.report["runs"][name]
row["caption_scope"] = "core only, full scene and padded crop as context"
exp.write_report()
print("CORE SCOPE RESULT",row,flush=True)
