#!/usr/bin/env bash
# switch.sh — switch the engine serving the OpenAI-compatible endpoint on
# host port 8020: `ninfer` (Qwen3.8-27B NVFP4, default) or `infernix`
# (Qwen3.8-Flash-Next, a 125B MoE served from system RAM + SSD overflow).
#
# Usage: ./switch.sh {ninfer|infernix}
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
  ninfer | infernix) ;;
  *)
    echo "usage: $(basename "$0") {ninfer|infernix}" >&2
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
