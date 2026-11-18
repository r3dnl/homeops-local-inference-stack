# Benchmark control service (on-demand; idle-safe).
#
# Runs benchmarks/context_bench.py as a subprocess behind a small
# FastAPI control/results API (benchmarks/server.py); Prometheus scrapes
# its /metrics for the per-run benchmark numbers. The container itself
# never touches the GPU — it only issues OpenAI-API requests to the
# engine (BENCH_BASE_URL). Results + the run index land in the shared
# ./benchmarks/results bind mount (same folder local runs write to).

FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    httpx \
    tokenizers

# The suite (executed via subprocess) and the headless control API.
# results/ is bind-mounted over by compose (shared with host-side runs);
# ui/ (Streamlit app, own service) is not part of this image.
COPY benchmarks/context_bench.py benchmarks/parallel_bench.py benchmarks/server.py /app/benchmarks/

EXPOSE 8040

CMD ["python", "-m", "uvicorn", "benchmarks.server:app", "--host", "0.0.0.0", "--port", "8040"]
