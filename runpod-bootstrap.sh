#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p /tmp/krea-results
cd /tmp/krea-results
python -m http.server 8888 --bind 0.0.0.0 >/tmp/http.log 2>&1 &
trap 'printf "FAILED during setup or comparison\n" > /tmp/krea-results/status.txt; tail -qc 12000 /tmp/bench.log /tmp/comfy.log /tmp/setup.log /tmp/model.log /tmp/clip.log /tmp/vae.log > /tmp/krea-results/diagnostics.txt 2>/dev/null || true; sleep 1200' ERR
printf 'Preparing ComfyUI and models' > status.txt
git clone -q --depth 1 https://github.com/Comfy-Org/ComfyUI.git /tmp/ComfyUI
cd /tmp/ComfyUI
git clone -q --branch feat/krea2-tile-vision --depth 1 https://github.com/tea-time-labs/sweet-tea-nodes.git custom_nodes/sweet-tea-nodes
test "$(git -C custom_nodes/sweet-tea-nodes rev-parse HEAD)" = 3a8db800d33140552f65531e6e718f40aab15185
mkdir -p models/diffusion_models models/text_encoders models/vae
fetch() {
  curl -fsSL --retry 3 --connect-timeout 20 --max-time 2400 "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/$1" -o "models/$1.tmp"
  mv "models/$1.tmp" "models/$1"
}
fetch diffusion_models/krea2_turbo_fp8_scaled.safetensors >/tmp/model.log 2>&1 & model=$!
fetch text_encoders/qwen3vl_4b_fp8_scaled.safetensors >/tmp/clip.log 2>&1 & clip=$!
fetch vae/qwen_image_vae.safetensors >/tmp/vae.log 2>&1 & vae=$!
python -m pip install -r requirements.txt -r custom_nodes/sweet-tea-nodes/requirements.txt >/tmp/setup.log 2>&1 & deps=$!
for job in "$model" "$clip" "$vae" "$deps"; do wait "$job"; done
git rev-parse HEAD > /tmp/krea-results/environment.txt
git -C custom_nodes/sweet-tea-nodes rev-parse HEAD >> /tmp/krea-results/environment.txt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> /tmp/krea-results/environment.txt
python -c 'import torch; print(torch.__version__,torch.cuda.is_available())' >> /tmp/krea-results/environment.txt
printf 'Starting ComfyUI with full weights' > /tmp/krea-results/status.txt
curl -fsSL https://raw.githubusercontent.com/tea-time-labs/sweet-tea-nodes/qualify/krea2-vision-gpu/runpod-krea-comparison.py -o /tmp/krea-bench.py
python main.py --listen 127.0.0.1 --port 8188 >/tmp/comfy.log 2>&1 &
for i in $(seq 1 120); do curl -fsS http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && break; sleep 2; done
python /tmp/krea-bench.py >/tmp/bench.log 2>&1
cp input/st-krea-vision-fixture.png /tmp/krea-results/source.png
cp output/krea-vision-off_00001_.png /tmp/krea-results/off.png
cp output/krea-vision-vision_00001_.png /tmp/krea-results/vision.png
cp /tmp/bench.log /tmp/krea-results/timings.txt
printf 'PASS: Off and Vision produced 1664 x 1152 images' > /tmp/krea-results/status.txt
sleep 1800
