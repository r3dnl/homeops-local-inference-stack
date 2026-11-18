# My local inference compose stack for Ninfer/Infernix on RTX5090

A minimal local inference stack: one GPU, two switchable engines —
[NInfer](https://github.com/Neroued/ninfer) and
[Infernix](https://github.com/Wallawalla47/Infernix) — one OpenAI-compatible
endpoint on port **8020**, plus a terminal-only benchmark service.
Stripped-down, cleaned version of `~/repos/qwen3.8-27b-rtx5090`.

| Engine (compose profile) | Model served | Format |
|---|---|---|
| [`ninfer`](https://github.com/Neroued/ninfer) (default) | Qwen3.8-27B NVFP4 | single `.ninfer` artifact, ~17 GB |
| [`infernix`](https://github.com/Wallawalla47/Infernix) | Qwen3.8-Flash-Next (125B MoE) | `.infernix` shards + n-gram volume, ~130 GB |

Both publish host port 8020 with the same OpenAI-compatible API, so clients
(Pi, Open WebUI, Newelle, …) don't notice a switch. Only one engine runs at
a time (one GPU).

## Why this shape

- **Docker Compose**: the whole stack — engine, benchmark service, model
  mounts, GPU reservation, ports — is one declarative file. `docker compose
  up -d` brings it up on any box with Docker + the NVIDIA container toolkit;
  `down` takes it all out. No host Python environments, no install scripts
  mutating the system, and `docker compose config` shows the exact effective
  setup. Rebuild after touching an engine checkout with one `--build`.
- **`./switch.sh <engine>`**: one command swaps the backend *and* the model
  it serves (Qwen3.8-27B ↔ Qwen3.8-Flash-Next) behind the same endpoint —
  profile flip, clean teardown of the old engine (compose's `down` strands
  inactive-profile containers on the GPU), build-if-missing, and a smoke
  test that waits until `/v1/models` actually answers. Serve-config tweaks
  (context length, concurrency, KV dtype, draft tokens) are plain `.env`
  keys, so switching model *or* tuning a backend is one edit + one switch —
  no compose edits, and clients never change their URL or key.

## Dropped vs the source repo (on purpose)

- The **vLLM** fallback engine and the **ninfer-custom** fork (rollback/fork
  variants you don't need on a clean box)
- **PolyStrata** handover (host-process mode, ZFS ARC tuning)
- The **caddy gateway**, **dcgm-exporter**, and the Prometheus/Grafana
  wiring (expose 8020 directly; observability lives in the homeops cluster)
- The **Streamlit bench UI** — the benchmark service is headless REST,
  driven from the terminal

Those all still exist in `~/repos/qwen3.8-27b-rtx5090` if you ever need to
grow a piece back.

## Requirements

- Docker + NVIDIA Container Toolkit (comment block at the top of `setup.sh`
  links the install guide)
- The engine source clones to build the images — house convention puts
  third-party clones under `~/repos` (`~/repos/ninfer`, `~/repos/Infernix`);
  point `NINFER_BUILD_DIR` / `INFERNIX_BUILD_DIR` at them with absolute
  paths in `.env` (unset, compose falls back to sibling checkouts
  `../ninfer` / `../Infernix`)
- ~20 GB disk for `./models/` (27B), plus ~130 GB NVMe for the Flash-Next
  model if you use the infernix engine

## First run

```sh
cp .env.example .env        # fill in NINFER_API_KEY (or leave empty on a trusted LAN)
./setup.sh                  # downloads tokenizer files + the .ninfer artifact (~17 GB)
docker compose up -d --build  # first build compiles the engine (slow, one-time)
# — or, after .env exists, just:
./switch.sh ninfer
```

Smoke test:

```sh
curl -s http://localhost:8020/v1/models | head   # add -H "Authorization: Bearer <key>" if set
```

## Switching engines

```sh
./switch.sh infernix    # stop 27B, start Flash-Next (first boot reads ~130 GB; up to 40 min)
./switch.sh ninfer      # back
```

`switch.sh` flips `COMPOSE_PROFILES` in `.env` (the single source of truth),
runs `docker compose down --remove-orphans`, removes containers the profile
change strands (they would keep holding port 8020 + the GPU), brings the
stack up (building only if the image was never built), and polls
`/v1/models` until the engine answers.

The Flash-Next model is **not** downloaded by `setup.sh` — one-time manual
download into the dir `INFERX_MODELS_DIR` points at (see `.env.example`;
the `<models dir>/ngram` mountpoint dir must exist).

## Benchmarks (terminal-only)

The `bench` service (ninfer profile) autostarts with the stack, touches the
GPU only while a run is in flight, and speaks a small REST API on loopback
port 8040 (endpoints documented in `benchmarks/server.py`):

```sh
curl -s 127.0.0.1:8040/status                       # idle/active + last run
curl -s -X POST 127.0.0.1:8040/bench -d '{"quick": true}' -H 'content-type: application/json'
curl -s 127.0.0.1:8040/runs                         # run index, newest first
```

Runs also work without the service, straight from the terminal (deps land in
`.venv_download` via `setup.sh`):

```sh
.venv_download/bin/python benchmarks/context_bench.py --help
.venv_download/bin/python benchmarks/parallel_bench.py --help
```

Results (report JSONs + `runs.jsonl` index) land in `benchmarks/results/`,
shared between service runs and local runs.

## Day-2 operations

```sh
docker logs -f ninfer | tail      # engine log (also: infernix, bench)
docker restart ninfer             # restart one service
docker compose down               # stop everything
docker compose up -d --build      # rebuild after touching the engine checkouts
```

Performance knobs are `.env`-overridable serve flags (context, concurrency,
KV dtype, draft tokens) — see the comment blocks in `.env.example` and
`docker-compose.yml`. The shipped profile assumes a dedicated GPU.
