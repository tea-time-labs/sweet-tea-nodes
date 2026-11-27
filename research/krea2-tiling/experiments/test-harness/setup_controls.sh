#!/bin/bash
set -euo pipefail
cd /content/st-krea16
mkdir -p controls weights/loras weights/depth-anything-v2-large
checkout() {
  local dir="$1" repo="$2" rev="$3"
  mkdir -p "$dir"
  git -C "$dir" init -q
  git -C "$dir" fetch -q --depth=1 "https://github.com/$repo.git" "$rev"
  git -C "$dir" checkout -q --detach FETCH_HEAD
}
fetch() {
  local url="$1" dest="$2"
  curl -fsSL --retry 2 "$url" -o "$dest.partial"
  mv "$dest.partial" "$dest"
}
checkout controls/facok facok/comfyui-krea2-controlnet 79ebfd3bd80d2180b334dd7ce57f3c9ddaa0848f &
a=$!
checkout controls/nk2e Nynxz/ComfyUI-NK2E ea321e1603e019f90971c743adce17146756fdac &
b=$!
fetch https://huggingface.co/Patil/Krea-2-depth-controlnet/resolve/21889413aa7282a6e78bd510247cceccad034b24/depth-control-lora.safetensors weights/loras/depth-control-lora.safetensors &
c=$!
fetch https://huggingface.co/nynxz/NK2E/resolve/bfaa0a77c79669508915a953f0cecb103d95f970/comfy/canny_v0.1/NK2E-canny-v0.1.safetensors weights/loras/NK2E-canny-v0.1.safetensors &
d=$!
depth_base=https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf/resolve/7581137eff8d4e94f6e796d3baea0e9fa79b22d2
for f in config.json preprocessor_config.json; do fetch "$depth_base/$f" "weights/depth-anything-v2-large/$f"; done
fetch "$depth_base/model.safetensors" weights/depth-anything-v2-large/model.safetensors &
e=$!
for job in "$a" "$b" "$c" "$d" "$e"; do wait "$job"; done
stat -c '%s %n' weights/loras/*.safetensors weights/depth-anything-v2-large/model.safetensors
touch controls/setup-complete
