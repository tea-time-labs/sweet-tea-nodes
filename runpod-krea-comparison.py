from pathlib import Path
from uuid import uuid4
import json
import time
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

root = Path.cwd()
inp = root / "input" / "st-krea-vision-fixture.png"
inp.parent.mkdir(exist_ok=True)
w, h = 832, 576
x, y = np.meshgrid(np.arange(w), np.arange(h))
gradient = np.stack((64 + x * 90 / w, 91 + y * 100 / h, 110 + (x + y) * 60 / (w + h)), -1)
image = Image.fromarray(gradient.astype("uint8"))
draw = ImageDraw.Draw(image)
# Red lettering, fabric-like lines and arcs cross the two output tile seams.
for i in range(17):
    offset = i * 6
    draw.arc((85 + offset, 84 + offset, 780 - offset, 518 - offset), 8, 342,
             fill=(22 + i * 5, 123 + i * 3, 206 - i * 7), width=3)
for j in range(36):
    draw.line((325 + j * 6, 235, 270 + j * 6, 405), fill=(191, 71, 50), width=2)
draw.rounded_rectangle((55, 70, 245, 251), radius=12, fill=(228, 212, 179))
draw.rounded_rectangle((583, 364, 777, 517), radius=12, fill=(65, 36, 34))
draw.text((86, 105), "LEFT / LINEN", fill=(46, 53, 62))
draw.text((608, 395), "RIGHT / SLATE", fill=(235, 221, 204))
image.save(inp)

api = "http://127.0.0.1:8188"
def call(path, body=None):
    encoded = json.dumps(body).encode() if body else None
    request = urllib.request.Request(api + path, data=encoded, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)

def node(cls, **inputs):
    return {"class_type": cls, "inputs": inputs}

def run(mode):
    prompt = "Preserve the image, composition, colors and label text. Refine existing small details."
    graph = {
        "1": node("LoadImage", image=inp.name),
        "2": node("UNETLoader", unet_name="krea2_turbo_fp8_scaled.safetensors", weight_dtype="default"),
        "3": node("CLIPLoader", clip_name="qwen3vl_4b_fp8_scaled.safetensors", type="krea2", device="default"),
        "4": node("VAELoader", vae_name="qwen_image_vae.safetensors"),
        "5": node("CLIPTextEncode", clip=["3", 0], text=prompt),
        "6": node("ConditioningZeroOut", conditioning=["5", 0]),
        "7": node("STKrea2UltimateUpscale", image=["1", 0], model=["2", 0],
            clip=["3", 0], vision_prompt=prompt, positive=["5", 0], negative=["6", 0],
            vae=["4", 0], upscale_by=2., seed=32079, steps=8, cfg=1.,
            sampler_name="euler", scheduler="simple", denoise=.25, tile_size=1024,
            tile_padding=64, mask_blur=16, control_mode=mode, control_strength=.7,
            depth_lora="depth-control-lora.safetensors", canny_lora="NK2E-canny-v0.1.safetensors",
            tile_lora="",
            depth_resolution=1024, canny_low=.1, canny_high=.2),
        "8": node("SaveImage", images=["7", 0], filename_prefix="krea-vision-" + mode.lower()),
    }
    start = time.monotonic()
    receipt = call("/prompt", {"prompt": graph, "client_id": str(uuid4())})
    prompt_id = receipt["prompt_id"]
    print(mode, "queued", prompt_id, flush=True)
    while time.monotonic() - start < 900:
        history = call("/history/" + prompt_id).get(prompt_id)
        if not history:
            time.sleep(2)
            continue
        if history["status"]["status_str"] != "success":
            raise RuntimeError(history["status"])
        outputs = history["outputs"]["8"]["images"]
        assert len(outputs) == 1
        file = root / "output" / outputs[0]["subfolder"] / outputs[0]["filename"]
        with Image.open(file) as result:
            assert result.size == (1664, 1152)
        print(mode, round(time.monotonic() - start, 2), "seconds", file, flush=True)
        return
    raise TimeoutError(mode)

for mode in ("Off", "Vision"):
    run(mode)
