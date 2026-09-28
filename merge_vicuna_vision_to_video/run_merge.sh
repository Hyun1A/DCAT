#!/usr/bin/env bash
set -u

DIR=$(cd "$(dirname "$0")"; pwd)
cd "$DIR"
mkdir -p logs results logs/merged_ckpt

CKPT=xxx/xxx/vicuna_models/checkpoints
BASE=$CKPT/vicuna-7b-v1.5
AUDIO=$CKPT/multimodal-vicuna-7b-v1.5-audio-naivemc
VIDEO=$CKPT/multimodal-vicuna-7b-v1.5-video-naivemc
VISION=$CKPT/multimodal-vicuna-7b-v1.5-vision-naivemc

CONDA_ENV="${CONDA_ENV:-xxx_conda_env}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
unset PYTHONSTARTUP
export PYTHONBREAKPOINT=0

gpu="${GPU:-0}"

MODAL_CACHE_DIR="${MODAL_CACHE_DIR:-/dev/shm/modal_cache}"
VIDEOMME_SRC="${VIDEOMME_SRC:-xxx/xxx/data/videomme}"
VIDEOMME_DST="${VIDEOMME_DST:-/dev/shm/videomme}"

if [ ! -f "$VIDEOMME_DST/videomme_test_flat.json" ]; then
  if [ ! -f "$VIDEOMME_SRC/videomme_test_flat.json" ]; then
    echo "✗ VideoMME source not found at $VIDEOMME_SRC — cannot stage video cache"
    exit 1
  fi
  echo "▶ Staging VideoMME data into RAM: $VIDEOMME_SRC -> $VIDEOMME_DST"
  mkdir -p "$VIDEOMME_DST"
  rsync -a "$VIDEOMME_SRC/" "$VIDEOMME_DST/"
fi

AVQA_SRC="${AVQA_SRC:-xxx/xxx/data/AVQA_MUSIC-AVQA_supp/data}"
AVQA_DST="${AVQA_DST:-/dev/shm/avqa_data/data}"
if [ ! -f "$AVQA_DST/test/avqa-test_mm_video.json" ]; then
  if [ ! -f "$AVQA_SRC/test/avqa-test_mm_video.json" ]; then
    echo "✗ AVQA source not found at $AVQA_SRC — cannot stage calibration data"
    exit 1
  fi
  echo "▶ Staging AVQA data into RAM: $AVQA_SRC -> $AVQA_DST"
  mkdir -p "$AVQA_DST"
  rsync -a "$AVQA_SRC/" "$AVQA_DST/"
fi

if [ -f "$MODAL_CACHE_DIR/READY" ] && [ ! -f "$MODAL_CACHE_DIR/video.pt" ]; then
  echo "▶ Stale modal cache (READY without video.pt) — clearing $MODAL_CACHE_DIR"
  rm -rf "$MODAL_CACHE_DIR"
fi

if [ ! -f "$MODAL_CACHE_DIR/READY" ]; then
  echo "▶ Preloading VideoMME video into $MODAL_CACHE_DIR (one-time, env=$CONDA_ENV)..."
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  CUDA_VISIBLE_DEVICES=$gpu conda run --no-capture-output -n "$CONDA_ENV" \
    python3 preload_modal_cache.py \
      --video-checkpoint "$VIDEO" \
      --model-base "$BASE" \
      --videomme-root "$VIDEOMME_DST" \
      --cache-dir "$MODAL_CACHE_DIR" \
      --skip-audio \
      2>&1 | tee "logs/preload_modal_cache.log"
  if [ ! -f "$MODAL_CACHE_DIR/video.pt" ]; then
    echo "⚠ Preload failed (no video.pt produced) — will fall back to per-sample decode"
    rm -f "$MODAL_CACHE_DIR/READY"
  fi
else
  echo "▶ Reusing existing modal cache at $MODAL_CACHE_DIR"
fi


ETA=1e5

TOP_P=128

ALPHA=0.5

BETA=0.3

START_LAYER=0
END_LAYER=31

PROJ_GROUPS=all

NUM_SAMPLES=256
NUM_ALIGN=128
VME_EVAL=1800
VME_SEED=77

SKIP_BEFORE_FLAG="--skip-before-accuracy"

VERBOSE=1
if [ "$VERBOSE" -eq 1 ]; then
  VERBOSE_FLAG="--verbose"
else
  VERBOSE_FLAG=""
fi

tag="DCAT_eta${ETA}_k${TOP_P}_b${BETA}_L${START_LAYER}-${END_LAYER}_${PROJ_GROUPS}"

echo "[launch] GPU=$gpu  η=$ETA  k=$TOP_P  α=$ALPHA (auto)  β=$BETA (fixed)  tag=$tag"
CUDA_VISIBLE_DEVICES=$gpu conda run --no-capture-output -n "$CONDA_ENV" \
  python mi_surrogate_merge_eval.py \
    --model-base "$BASE" --recipient-model-path "$VIDEO" --donor-model-paths "$VISION" \
    --proj-groups "$PROJ_GROUPS" \
    --start-layer "$START_LAYER" --end-layer "$END_LAYER" \
    --calib-modality video --calib-dataset avqa \
    --num-samples "$NUM_SAMPLES" --num-align-samples "$NUM_ALIGN" \
    --eval-mode videomme --num-eval-samples "$VME_EVAL" --videomme-seed "$VME_SEED" \
    --eta "$ETA" --top-p "$TOP_P" --alpha "$ALPHA" --beta "$BETA" --fixed-beta \
    --eps-solve 1e-6 \
    --output-dir "logs/merged_ckpt/${tag}" \
    --result-dir "results" \
    --config-tag "$tag" \
    $VERBOSE_FLAG \
    $SKIP_BEFORE_FLAG
