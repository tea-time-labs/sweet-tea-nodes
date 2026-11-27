# From 1 MP to 9 MP with Krea 2 Turbo

[Download the complete illustrated handoff](https://raw.githubusercontent.com/tea-time-labs/sweet-tea-nodes/refs/heads/research/krea2-tile-coordination-handoff/research/krea2-tiling/Krea2-upscaling-handoff.pdf)

The artist wanted a generative 3× enlargement worth inspecting at 100%. Fabric, foliage and subtle distant structures can gain new detail. Subject count, pose, contacts, perspective and the direction of a shoe must survive. The target was about one megapixel enlarged to nine, using nine padded windows, the workflow’s Qwen3-VL encoder, Vision by default and optional Canny or Depth. The trials covered many different prompts, including handoffs and other interactions.

The portrait trace identifies one specific failure. Tiles 1 and 4 saw the same real eye in their padded source crops. Neither failed-run caption said “eye”. On the same first-step noisy canvas, their predictions put it at two different heights. Blending retained both. Hand-assigning the entire eye region to tile 4 on steps 1–4 preserved one eye, even after ordinary blending resumed. Starting the same intervention on step 5 was too late on this seed. Core-only writing also produced two eyes. An automatic source-matching rule left the duplication in both the portrait and the fox family. The hand-selected eye result establishes a mechanism in one case. General feature coordination at high denoise remains open.

## Read the record

| Material | Where to look |
|---|---|
| Illustrated account from the creative brief through the failed pilot | [Main PDF](https://raw.githubusercontent.com/tea-time-labs/sweet-tea-nodes/refs/heads/research/krea2-tile-coordination-handoff/research/krea2-tiling/Krea2-upscaling-handoff.pdf) |
| All 16 full prompts, seeds, clean PNGs, rectangles and generated captions | [Matrix inputs](./inputs/matrix) and [earlier padding and guide inputs](./inputs/padding) |
| All 48 sampler comparisons and eight selected follow-ups | [Illustrated matrix review](./evidence/reports/matrix.pdf) and [machine-readable findings](./evidence/matrix/review-observations.json) |
| Native-dimension JPEG for each of the 16 Euler/simple upscales | [Full-size gallery](./evidence/matrix/full-size). Open a JPEG at 100%. The manifest identifies each original lossless render by SHA-256. |
| 144 core captions scored individually against their pixels | [Tap-through audit](./evidence/reports/tile-audit.pdf), [grade and reason for each tile](./evidence/matrix/tile-audit.tsv) |
| Two-half method, clean-core and context checks, exact alternative text, stored judgments and landmark scores | [Methods PDF](./evidence/reports/caption-methods.pdf), [scores and caption outputs](./evidence/caption-methods) |
| Padded or full-scene boundary views, numeric coordinates and spotlight | [Boundary PDF](./evidence/reports/boundary-study.pdf) and [raw grades and inputs](./evidence/boundary-study) |
| Same-step upper/lower tile estimates, merged estimates, active rules and native callback errors | [Timing PDF](./evidence/reports/portrait-trace.pdf) and [48 saved lossless windows](./evidence/portrait-coordination) |
| Failed core-only and automatic overlap trials | [Pilot PDF](./evidence/reports/ownership-pilot.pdf), [actual scores and decisions](./evidence/source-overlap), [withdrawn prototype and original diff](./experiments/source-overlap) |
| Exact source path, SHA-256 and file-pixel crop behind each saved comparison | [Figure manifest](./evidence/figures/manifest.json). The four boundary cue thumbnails were reconstructed using the archived cue function and verified against its recorded pre-resize RGB hashes. |
| SHA-256 inventory of inputs, snapshots, reports, review images and experiment files | [Checksums](./checksums.sha256) |

The source images were generated for these studies. The PDF reproduces selected aligned crops. The supplements include the rest of the visual reviews. The 16 gallery JPEGs retain output dimensions and use lossy compression. Use the original source PNGs, archived trace PNGs and model outputs from a replay for pixel-level evaluation.

## Colab replay

Run the following in a shell on a **fresh GPU runtime**. In a Colab notebook, put `%%bash` at the top of a shell cell. The matched portrait used an NVIDIA L4 with Torch 2.14.1+cu130. The fresh-tile caption comparisons also have T4 records. The setup uses the runtime's Torch and checks the sizes of the three downloaded weights. Compare outputs by saved source, captions, seed, device and Torch version. A seed alone does not ensure bit-identical output on a different stack.

```bash
git clone --branch research/krea2-tile-coordination-handoff --single-branch \
  https://github.com/tea-time-labs/sweet-tea-nodes.git /content/st-handoff
cd /content/st-handoff/research/krea2-tiling
python prepare_replay.py
python /content/st-krea16/test-harness/setup_runtime.py
```

The staging script builds the original two `/content` directory layouts. Both point at the same pinned repositories and weights. It copies saved captions, exact PNGs and reports. The padding runners mark completions solely by run ID, so the staging copy clears their completed runs. Repository evidence retains the original records. For an offline layout check, pass `--destination` with an existing empty directory.

### Reproduce the double eye, then vary only the timing

```bash
cd /content/st-krea-3x-padding
ST_KREA_PROMPT_CASE=portrait \
  ST_KREA_SOURCE_PROMPT="$(cat results/portrait/original-prompt.txt)" \
  ST_KREA_EXPERIMENT_RUNS=p256_grounded_local python -u experiment.py
python -u test_overlap_owner.py
python -u trace_portrait.py
```

The first command renders ordinary blending using the portrait’s archived padded captions. The second renders isolated tiles 1 and 4 using registered windows of the original source latent and initial noise, then grants tile 4 the manually marked region for all eight steps. The third applies the same mask in steps 1–4 or 5–8. It saves full results and eight × three decoded predictions per intervention under `/content/st-krea-3x-padding/results/portrait`. The trace windows are 896 × 512 output pixels at x=0, y=768. Tile ID 1 is upper left. Tile ID 4 is middle left. The separate `test_core_captions.py` tests changing the textual target in this portrait.

### Replay the waterfront and the 16-prompt matrix

```bash
cd /content/st-krea-3x-padding
python -u experiment.py
python -u test_sequential.py
python -u /content/st-krea16/matrix-as-executed/run_matrix.py --first 1 --last 16
```

The first runner contains the padding, coordinate and shared/grounded text arms. The sequential control was a separate eight-steps-per-tile path with committed clean cores. Its legacy recovery helper looks for interrupted control outputs and may print an unavailable recovery before starting the sequential test. The 48-run matrix uses the **exact three Python/JSON files extracted from the executed Colab bundle** in [matrix-as-executed](./experiments/matrix-as-executed). It reads the supplied source PNGs and regenerates captions with the original procedure. The later [test harness](./experiments/test-harness) supports supplying saved captions and layering guides on Vision. The production code at the pinned baseline exposes separate control modes and passes `vision_prompt` into its Vision path.

### Replay caption measurements and matched text interventions

```bash
cd /content/st-krea16
python -u caption-methods/run.py development core context_check split_check
python -u caption-methods/run.py confirmation original_t4 split_check
python -u caption-methods/duel.py
PYTHONPATH=/content/st-krea16/caption-methods python -u boundary-study/run_boundary.py
python -u boundary-study/render_boundary.py
```

To render the exact saved two-half proposals independently of any newly generated caption, use a fresh staging layout and the preserved text records:

```bash
cd /content/st-krea16
cp -R /content/st-handoff/research/krea2-tiling/evidence/caption-methods/outputs caption-methods/
python -u caption-methods/render.py split_check 01-florist-handoff 10-wildlife-family
```

### Replay optional structural guidance

```bash
cd /content/st-krea16
bash test-harness/setup_controls.sh
python -u test-harness/run_guide_examples.py 03-library-architecture
python -u test-harness/run_guide_examples.py 10-wildlife-family
```

The guide script prepares Canny edges or Depth Anything V2 from the full source and projects it into the actual padded windows. Both arms preserve each tile’s Vision input and text. Guide strength is 0.7 in these two matched examples. The guide setup script pins adapter repositories, model revisions and weights.

### Replay the withdrawn source-score experiment separately

On an **independent fresh L4 runtime**, run the initial setup block, then:

```bash
cd /content/st-krea16
python -u trial/run.py portrait-blend portrait-core portrait 10-wildlife-family
```

This historical runner applies its experimental patch to the clean pinned checkout before sampling. Invoke it once with all desired cases. `portrait-blend` is an optional ordinary-blend control. `portrait-core` restricts every write to the core. `portrait` and `10-wildlife-family` apply the failed source-score rule. The report covers the recorded runs. New replays write outputs and decisions under `auto-results/`. The original patch remains under research, outside the production node path.

## Provenance and document rebuild

| Component | Archived revision |
|---|---|
| Sweet Tea Nodes | [8c97fe1](https://github.com/tea-time-labs/sweet-tea-nodes/tree/8c97fe110adb6e3a1c92976c2fd7f3d82c2e0935) |
| ComfyUI | [7c78b56](https://github.com/Comfy-Org/ComfyUI/tree/7c78b5641ab2d6cb7f7766a183a8a43a59de0a8b) |
| Krea 2 Turbo FP8, Qwen3-VL-4B FP8, Qwen Image VAE | [Comfy-Org/Krea-2 at 588feaf](https://huggingface.co/Comfy-Org/Krea-2/tree/588feaf955b08b9ac4723eeb751752ab045dba71) |

From this directory, with Pillow and ReportLab installed:

```bash
sha256sum --check --quiet checksums.sha256
python build_handoff.py --revision "$(git rev-parse HEAD)" --output /tmp/Krea2-upscaling-handoff.pdf
```

The PDF’s links identify the exact code/evidence snapshot used when the report was built. This handoff assembled retained Colab evidence and checked it offline. The dedicated branch records the research and its failed prototype. A production change still needs a method that generalizes to held-out images while preserving coherent new detail.

