from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

import torch

root = Path("/content/st-krea16")
root.mkdir(exist_ok=True)
print("Runtime GPU", torch.cuda.get_device_name(0), "| Torch", torch.__version__, flush=True)

repos = {
    "ComfyUI": ("Comfy-Org/ComfyUI", "7c78b5641ab2d6cb7f7766a183a8a43a59de0a8b"),
    "sweet-tea-nodes": ("tea-time-labs/sweet-tea-nodes", "8c97fe110adb6e3a1c92976c2fd7f3d82c2e0935"),
}
weights = [
    "diffusion_models/krea2_turbo_fp8_scaled.safetensors",
    "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
    "vae/qwen_image_vae.safetensors",
]
hf_base = "https://huggingface.co/Comfy-Org/Krea-2/resolve/588feaf955b08b9ac4723eeb751752ab045dba71"
weight_sizes = dict(zip(weights, (13141730784, 5242467968, 253806246)))

def checkout(task):
    name, (repo, ref) = task
    target = root / name
    if not target.exists():
        target.mkdir()
        subprocess.run(["git", "init", "-q", str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "fetch", "-q", "--depth=1", f"https://github.com/{repo}.git", ref], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)
    print(name, ref, flush=True)

def download(relative):
    target = root / "weights" / relative
    target.parent.mkdir(exist_ok=True, parents=True)
    if target.is_file() and target.stat().st_size == weight_sizes[relative]:
        return
    work = target.with_suffix(".partial")
    subprocess.run(["curl", "-fsSL", "--retry", "3", "--retry-delay", "1",
                    f"{hf_base}/{relative}", "-o", str(work)], check=True)
    if work.stat().st_size != weight_sizes[relative]:
        raise RuntimeError(f"Wrong size for {relative}")
    work.replace(target)
    print("Downloaded", relative, target.stat().st_size, flush=True)

def install():
    comfy = root / "ComfyUI"
    checkout(("ComfyUI", repos["ComfyUI"]))
    deps = []
    for line in (comfy / "requirements.txt").read_text().splitlines():
        item = line.strip()
        if not item or item.startswith(("#", "comfyui-", "comfy-angle", "PyOpenGL")) or item in ("torch", "torchvision"):
            continue
        deps.append(item)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *deps], check=True)
    print("ComfyUI dependencies ready", flush=True)

with ThreadPoolExecutor(max_workers=5) as pool:
    work = [pool.submit(install), pool.submit(checkout, ("sweet-tea-nodes", repos["sweet-tea-nodes"]))]
    work.extend(pool.submit(download, weight) for weight in weights)
    for item in work:
        item.result()
print("Setup complete", flush=True)
