FROM python:3.13-slim

# install uv (no managed Python downloads; the base image provides Python 3.13)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# install dependencies from the frozen lockfile first (better layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY predict.py model.bin ./

EXPOSE 9696

ENTRYPOINT ["/app/.venv/bin/uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
