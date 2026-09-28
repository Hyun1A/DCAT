#!/usr/bin/env bash
set -u

DIR=$(cd "$(dirname "$0")"; pwd)
cd "$DIR"
mkdir -p logs results logs/merged_ckpt

CKPT=xxx/xxx/llama_models/checkpoints
BASE=$CKPT/llama3.1-8b-instruct
VIDEO=$CKPT/llava-video-llama-3.1-8b
VISION=$CKPT/llava-llama3.1-8b

CONDA_ENV="${CONDA_ENV:-xxx_conda_env}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
unset PYTHONSTARTUP
export PYTHONBREAKPOINT=0

VIDEOMME_SRC="${VIDEOMME_SRC:-xxx/xxx/data/videomme}"
VIDEOMME_DST="${VIDEOMME_DST:-/dev/shm/videomme}"

if [ ! -f "$VIDEOMME_DST/videomme_test_flat.json" ]; then
  if [ ! -f "$VIDEOMME_SRC/videomme_test_flat.json" ]; then
    echo "✗ VideoMME source not found at $VIDEOMME_SRC"
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
    echo "✗ AVQA source not found at $AVQA_SRC"
    exit 1
  fi
  echo "▶ Staging AVQA data into RAM: $AVQA_SRC -> $AVQA_DST"
  mkdir -p "$AVQA_DST"
  rsync -a "$AVQA_SRC/" "$AVQA_DST/"
fi


gpu="${GPU:-0}"

ETA=1e6

TOP_P=256

ALPHA=0.5
BETA=0.95

START_LAYER=0
END_LAYER=31

PROJ_GROUPS=all

NUM_SAMPLES=256
NUM_ALIGN=128
VME_EVAL=900
VME_SEED=77

SKIP_BEFORE_FLAG="--skip-before-accuracy"

VERBOSE=1
if [ "$VERBOSE" -eq 1 ]; then
  VERBOSE_FLAG="--verbose"
else
  VERBOSE_FLAG=""
fi

tag="DCAT_llama_eta${ETA}_k${TOP_P}_b${BETA}_L${START_LAYER}-${END_LAYER}_${PROJ_GROUPS}"

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
