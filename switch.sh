#!/usr/bin/env bash
# switch.sh — switch the engine serving the OpenAI-compatible endpoint on
# host port 8020: `ninfer` (Qwen3.8-27B NVFP4, default), `infernix`
# (Qwen3.8-Flash-Next, a 125B MoE served from system RAM + SSD overflow),
# or hand the box over to PolyStrata (github.com/VecSzn/PolyStrata).
#
# Usage: ./switch.sh {ninfer|infernix|polystrata [model]}
#
# polystrata is not a compose service: it tears the compose stack down,
# raises the ZFS ARC (PolyStrata streams its GGUF experts through the ARC
# as its page cache - the Qwen engines must NOT boot against a fat ARC),
# and execs the run-<model>.sh that PolyStrata's setup wrote. The optional
# [model] is a regex matched against run-*.sh (newest first), so "glm"
# starts GLM 5.3 Flash and "dsv4" DeepSeek V4 Flash; the newest run script
# is the default. Once handed over, model-to-model switching goes through
# PolyStrata itself (python3 polystrata.py --model <name>) - no teardown,
# no switch.sh; its server answers on the same port 8020 and, without an
# api_key in strata-*.json, is open to anything that can reach it - put a
# gateway in front before exposing it. Coming back is a normal engine
# switch (./switch.sh ninfer): the PolyStrata server is torn down and the
# ARC reset automatically.
#
# Steps:
#   1. `docker compose down --remove-orphans` — stop the current stack.
#   2. Set COMPOSE_PROFILES in .env to the selected engine — compose starts
#      only the services of that profile (the engine + the bench service on
#      the ninfer profile). .env is the single source of truth, so plain
#      `docker compose` commands work from here on.
#   3. Remove containers `down` left behind: compose's `down` skips services
#      whose profile just went inactive, so the previous engine would
#      survive and keep holding host port 8020 + the GPU. Leftovers are
#      found via the compose project label.
#   4. `docker compose up -d` — with `--build` only when the selected
#      engine's image was never built (a first build is slow: NInfer and
#      Infernix compile their C++/CUDA engines). Rebuild the current sibling
#      checkout state deliberately with `docker compose up -d --build`.
#   5. Poll http://localhost:<port>/v1/models until it answers (both engines
#      publish host port 8020 by default; an infernix first boot reads the
#      ~130 GB Flash-Next model + runs the RAM ledger and expert-cache
#      warmup, so it gets a longer wait window).
set -euo pipefail

ENGINE="${1:-}"
case "${ENGINE}" in
  ninfer | infernix | polystrata) ;;
  *)
    echo "usage: $(basename "$0") {ninfer|infernix|polystrata [model]}" >&2
    exit 2
    ;;
esac

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
cd "${ROOT_DIR}"

if [ ! -f "${ENV_FILE}" ]; then
  echo "No .env found — run: cp .env.example .env   (then fill in the secrets)" >&2
  exit 1
fi

# Read one KEY=value line from .env (empty when the key is absent).
get_env() { grep -E "^$1=" "${ENV_FILE}" | tail -1 | cut -d= -f2- || true; }

# --- PolyStrata side of the switch (not a compose service: `down` cannot reach it) ---
ZFS_ARC_MAX=/sys/module/zfs/parameters/zfs_arc_max
# Where PolyStrata lives (the run-*.sh scripts it launches). Resolution:
# exported POLYSTRATA_DIR > POLYSTRATA_DIR in .env > $HOME/repos/PolyStrata.
POLY_REPO="${POLYSTRATA_DIR:-$(get_env POLYSTRATA_DIR)}"
POLY_REPO="${POLY_REPO:-$HOME/repos/PolyStrata}"
# PolyStrata's own default is 8080; this stack moves it to 8020 so every
# engine answers one port (the compose stack is down while PolyStrata runs,
# so nothing competes for it). PolyStrata's setup rewrites its strata-*.json
# and run-*.sh back to 8080 (with a freshly minted api_key) on a model
# reinstall - re-apply the 8020/no-key config after reinstalls.
POLY_PORT=8020
# ARC budget for PolyStrata's expert arena, in GiB (the 96 GiB source
# machine uses 76); override with POLY_ARC_GIB in .env.
POLY_ARC_GIB="$(get_env POLY_ARC_GIB)"; POLY_ARC_GIB="${POLY_ARC_GIB:-76}"
# Identity check, not just liveness: on the shared 8020 a live /health could
# equally be the compose engine, and treating that as PolyStrata would stall
# the revert loop waiting for a server that was never there.
poly_running() { curl -fsS -m 2 "http://127.0.0.1:${POLY_PORT}/health" 2>/dev/null | grep -q '"service": *"strata"'; }

if [ "$(get_env COMPOSE_PROFILES)" = "polystrata" ]; then
  if poly_running; then
    echo "==> stopping PolyStrata server (serve/server.py)"
    pkill -f "serve/server.py" || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do poly_running || break; sleep 2; done
  fi
  if [ -e "${ZFS_ARC_MAX}" ]; then
    echo "==> zfs_arc_max -> default (the Qwen engines must not boot against a fat ARC)"
    echo 0 | sudo tee "${ZFS_ARC_MAX}" >/dev/null
  fi
fi

# 1. Stop the current stack (COMPOSE_PROFILES still selects the engine that
# is currently running, so `down` reaches all of its services).
echo "==> docker compose down --remove-orphans (current engine)"
docker compose down --remove-orphans

# 2. Point .env at the selected engine.
if grep -q '^COMPOSE_PROFILES=' "${ENV_FILE}"; then
  sed -i "s/^COMPOSE_PROFILES=.*/COMPOSE_PROFILES=${ENGINE}/" "${ENV_FILE}"
else
  printf '\n# Engine selection (see switch.sh): ninfer or infernix\nCOMPOSE_PROFILES=%s\n' "${ENGINE}" >> "${ENV_FILE}"
fi
echo "==> .env: COMPOSE_PROFILES=${ENGINE}"

# 3. Remove containers `down` left behind (the previous engine's services
# are now in the inactive profile — compose skips them in `down`).
PROJECT_NAME="$(get_env COMPOSE_PROJECT_NAME)"
if [ -z "${PROJECT_NAME}" ]; then
  PROJECT_NAME="$(docker compose config | sed -n 's/^name: *//p')"
fi
LEFTOVERS="$(docker ps -aq --filter "label=com.docker.compose.project=${PROJECT_NAME}" || true)"
if [ -n "${LEFTOVERS}" ]; then
  echo "==> removing containers left behind by the previous engine"
  # shellcheck disable=SC2086
  docker rm -f ${LEFTOVERS}
fi

# 3b. polystrata: no image to build, no stack to bring up - hand the box over.
if [ "${ENGINE}" = "polystrata" ]; then
  if [ -e "${ZFS_ARC_MAX}" ]; then
    echo "==> zfs_arc_max -> ${POLY_ARC_GIB} GiB (PolyStrata serves its expert arena from the ARC)"
    echo $((POLY_ARC_GIB * 1024 * 1024 * 1024)) | sudo tee "${ZFS_ARC_MAX}" >/dev/null
  else
    echo "==> no ZFS on this host - skipping ARC tuning (PolyStrata streams its experts through the ARC; expect a slower cold path)"
  fi
  RUN_SH="$(ls -t "${POLY_REPO}"/run-*.sh 2>/dev/null | grep -m1 -- "${2:-}" || true)"
  if [ -z "${RUN_SH}" ]; then
    # ${2:-} not $2: `set -u` would abort this very error line when no model was given.
    echo "No run-*.sh matching '${2:-<any>}' in ${POLY_REPO} - installed: $(ls -1t "${POLY_REPO}"/run-*.sh 2>/dev/null | xargs -r -n1 basename | sed 's/^run-//; s/\.sh$//' | tr '\n' ' ') (part of a name is enough; install more with PolyStrata's setup)." >&2
    exit 1
  fi
  echo "==> $(basename "${RUN_SH}") - leave this terminal attached: closing it stops the model"
  echo "    switch installed models: python3 ${POLY_REPO}/polystrata.py --model <name>  |  web: http://127.0.0.1:${POLY_PORT}"
  echo "    back to the compose engines:  ./switch.sh ninfer   (from another terminal)"
  exec bash "${RUN_SH}"
fi

# 4. Bring the stack up on the selected engine.
# The engine image name is the compose project name + the service name of
# the active profile (ninfer / infernix).
IMAGE="${PROJECT_NAME}-${ENGINE}:latest"
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "==> docker compose up -d (image ${IMAGE} present — no rebuild)"
  docker compose up -d
else
  echo "==> docker compose up -d --build (image ${IMAGE} missing — slow one-time build)"
  docker compose up -d --build
fi

# 5. Smoke test: poll /v1/models on the engine's host port. Both engines
# publish NINFER_HOST_PORT and take the same NINFER_API_KEY.
API_KEY="$(get_env NINFER_API_KEY)"
HOST_PORT="$(get_env NINFER_HOST_PORT)"
HOST_PORT="${HOST_PORT:-8020}"
URL="http://localhost:${HOST_PORT}/v1/models"
AUTH_ARGS=()
[ -n "${API_KEY}" ] && AUTH_ARGS=(-H "Authorization: Bearer ${API_KEY}")

# infernix first boot reads the ~130 GB Flash-Next model + runs the RAM
# ledger and expert-cache warmup, so it gets a longer window than ninfer.
POLL_MIN=20
case "${ENGINE}" in infernix) POLL_MIN=40 ;; esac

echo "==> waiting for ${ENGINE} to answer ${URL} (a first-boot image build is also slow; infernix first boot reads the ~130 GB model)..."
READY=0
for i in $(seq 1 $((POLL_MIN * 12))); do   # 12 polls/min (5 s each) -> POLL_MIN minutes
  if curl -fsS ${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"} "${URL}" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if [ $((i % 12)) -eq 0 ]; then
    echo "    ... still waiting ($((i / 12)) min elapsed)"
  fi
  sleep 5
done

if [ "${READY}" -eq 1 ]; then
  echo
  echo "Engine '${ENGINE}' is serving: ${URL}"
  echo "Boot log: docker logs -f ${ENGINE}"
else
  echo
  echo "NOT READY after ${POLL_MIN} min — inspect: docker logs --tail 80 ${ENGINE}" >&2
  exit 1
fi
