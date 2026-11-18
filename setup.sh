#!/usr/bin/env bash
# setup.sh — download the Qwen3.8-27B assets (tokenizer files + the NInfer
# engine artifact) into ./models/. The Flash-Next model for the infernix
# engine is a separate one-time manual download (see .env.example); this
# script never fetches it.
set -euo pipefail

# Requirements: Docker + the NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
#   sudo apt-get install -y nvidia-container-toolkit
#   sudo nvidia-ctk runtime configure --runtime=docker
#   sudo systemctl restart docker

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Pick up model selection from .env (same keys docker compose uses):
#   MODEL_REPO, MODEL_SUBDIR
if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  . "${ROOT_DIR}/.env"
  set +a
fi

MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models}"
# Tokenizer source (Apache-2.0, RTX 5090 NVFP4 export of Qwen/Qwen3.8-27B):
#   https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090
# The bench suite needs ./models/<subdir>/tokenizer.json for token-accurate
# padding. (The NInfer engine reads its own tokenizer and chat template from
# the .ninfer artifact.)
MODEL_REPO="${MODEL_REPO:-gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090}"
MODEL_SUBDIR="${MODEL_SUBDIR:-qwen3.8-27b-nvfp4}"
# Tokenizer revision (known-good state, 2026-08-22) — pinned so a re-run
# always fetches exactly what was tested. Override via .env; set
# MODEL_REVISION= (empty) to always track the latest tokenizer files.
MODEL_REVISION="${MODEL_REVISION:-0cc27958cefbbe231782ec8511de8c4eb5233348}"
# NInfer engine artifact (v3 .ninfer, single ~17 GB file: weights, config,
# bindings, tokenizer, chat template, MTP companion weights). Pinned to a
# known-good revision so a re-run fetches exactly what was tested
# (2026-09-29). Override via .env; set NINFER_REVISION= (empty) to always
# track the latest artifact instead.
NINFER_ARTIFACT_REPO="${NINFER_ARTIFACT_REPO:-neroued/Qwen3.8-27B-nvfp4-NInfer}"
NINFER_ARTIFACT="${NINFER_ARTIFACT:-qwen3_8_27b_nvfp4.ninfer}"
NINFER_REVISION="${NINFER_REVISION:-a107ba1b5b0609d9ca90d5ed61439f5f5cc7d64d}"
VENV_DIR="${ROOT_DIR}/.venv_download"

mkdir -p "${MODEL_DIR}"

echo "Setup root: ${ROOT_DIR}"
echo "Model dir:  ${MODEL_DIR}/${MODEL_SUBDIR}"
echo "Setting up virtual environment..."

# Create the virtual environment if it doesn't exist. uv when it is on PATH,
# stdlib venv otherwise so a host without uv can still run setup.
if [ ! -d "${VENV_DIR}" ]; then
  if command -v uv >/dev/null 2>&1; then
    echo "Creating venv with uv for ${VENV_DIR}"
    uv venv "${VENV_DIR}"
  else
    echo "uv not on PATH; creating venv with python3 -m venv for ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
  fi
fi

# Activate the virtual environment
source "${VENV_DIR}/bin/activate"

# Resolve the uv binary. sudo resets PATH to the restricted secure_path, so a
# per-user uv install (~/.local/bin or ~/.cargo/bin) is invisible to this
# script; probe the usual install locations, including the invoking user's
# home (under sudo $HOME is /root). If uv truly is absent, install_pkgs
# below falls back to the venv's pip instead.
UV="$(command -v uv || true)"
if [ -z "${UV}" ]; then
  UV_PROBE_DIRS=("${HOME}/.local/bin" "${HOME}/.cargo/bin" /usr/local/bin)
  if [ -n "${SUDO_USER:-}" ]; then
    UV_PROBE_DIRS+=("/home/${SUDO_USER}/.local/bin" "/home/${SUDO_USER}/.cargo/bin")
  fi
  for d in "${UV_PROBE_DIRS[@]}"; do
    if [ -x "${d}/uv" ]; then
      UV="${d}/uv"
      break
    fi
  done
fi
[ -n "${UV}" ] && echo "Using uv: ${UV}"

# Install packages into the venv with uv when available, else with the
# venv's pip. uv-created venvs ship without pip, so bootstrap it via
# ensurepip (bundled with CPython) on first use.
install_pkgs() {
  if [ -n "${UV}" ]; then
    "${UV}" pip install --python "${VENV_DIR}/bin/python" "$@"
  else
    if ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
      echo "uv not found; bootstrapping pip into ${VENV_DIR} (ensurepip)..."
      "${VENV_DIR}/bin/python" -m ensurepip --upgrade
    fi
    "${VENV_DIR}/bin/python" -m pip install "$@"
  fi
}

# Install huggingface_hub (pinned) if the new 'hf' CLI is not present.
if ! command -v hf >/dev/null 2>&1; then
  echo "Installing huggingface_hub (pinned)..."
  install_pkgs "huggingface_hub[cli]==1.27.0"
fi

# Benchmark-suite deps (benchmarks/*.py, for local runs): tokenizers for
# token-accurate padding, httpx for the fast streaming transport.
echo "Installing benchmark dependencies into ${VENV_DIR} (tokenizers, httpx)..."
install_pkgs tokenizers httpx

# Pinned, accelerated download via Xet (replacing hf_transfer). The engine
# artifact carries its own tokenizer, so the ~18.8 GB of HF weights are not
# downloaded — only the small tokenizer files for the bench suite.
REV_ARGS=()
[ -n "${MODEL_REVISION}" ] && REV_ARGS+=(--revision "${MODEL_REVISION}")

echo "About to download hf tokenizer files from ${MODEL_REPO}${MODEL_REVISION:+ @ ${MODEL_REVISION}}:"
echo "hf download ${MODEL_REPO} tokenizer.json tokenizer_config.json ${REV_ARGS[@]+${REV_ARGS[@]}} --local-dir ${MODEL_DIR}/${MODEL_SUBDIR}"

HF_XET_HIGH_PERFORMANCE=1 hf download \
  "${MODEL_REPO}" \
  tokenizer.json \
  tokenizer_config.json \
  ${REV_ARGS[@]+"${REV_ARGS[@]}"} \
  --local-dir "${MODEL_DIR}/${MODEL_SUBDIR}"

NINFER_REV_ARGS=()
[ -n "${NINFER_REVISION}" ] && NINFER_REV_ARGS+=(--revision "${NINFER_REVISION}")

echo "About to download the NInfer artifact (~17 GB) ${NINFER_ARTIFACT}${NINFER_REVISION:+ @ ${NINFER_REVISION}}:"
echo "hf download ${NINFER_ARTIFACT_REPO} ${NINFER_ARTIFACT} ${NINFER_REV_ARGS[@]+${NINFER_REV_ARGS[@]}} --local-dir ${MODEL_DIR}"

HF_XET_HIGH_PERFORMANCE=1 hf download \
  "${NINFER_ARTIFACT_REPO}" \
  "${NINFER_ARTIFACT}" \
  ${NINFER_REV_ARGS[@]+"${NINFER_REV_ARGS[@]}"} \
  --local-dir "${MODEL_DIR}"

# Deactivate the virtual environment
deactivate

echo ""
echo "Download completed!"
echo "You can now start the model using 'docker compose up -d --build' (engine per COMPOSE_PROFILES in .env; switch with ./switch.sh)."
