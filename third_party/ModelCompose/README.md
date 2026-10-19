# ModelCompose (vendored for DCAT)

This directory contains the `modelcompose` package used by the Vicuna experiments in DCAT. It is
based on ModelCompose (https://github.com/THUNLP-MT/ModelCompose) as distributed in
MLLMerging / OptMerge (https://github.com/WalkerWorldPeace/MLLMerging, commit `8f205d4`), and is
released under the Apache License 2.0 (see `LICENSE`). Third-party components keep their own
licenses (`modelcompose/model/multimodal_encoder/beats/LICENSE_BEATs.txt`,
`modelcompose/model/multimodal_encoder/video_llama/models/ImageBind/LICENSE`).

Changes relative to the upstream copy:

- `model/builder.py`, `model/language_model/multimodal_llama.py`, `model/multimodal_arch.py`,
  `mm_utils.py` and related files are the versions the DCAT results were produced with
  (adapter loading for the released `*-naivemc` checkpoints and merged DCAT adapters).
- Encoder weights are no longer looked up at fixed absolute paths. `model/multimodal_encoder/builder.py`
  resolves the BEATs, LanguageBind-Video and CLIP encoders from `$MODELCOMPOSE_ENCODER_DIR` first and
  from the package directory second; see the DCAT README for the expected layout.
- `pyproject.toml` declares no pinned dependencies; install the environment from
  `requirements_vicuna.txt` in the repository root.

The DCAT scripts add this directory to `sys.path` themselves, so installing it is optional.
`pip install -e third_party/ModelCompose --no-deps` makes `import modelcompose` work elsewhere.
