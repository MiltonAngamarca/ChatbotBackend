FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /code


ENV UV_SYSTEM_PYTHON=1
COPY pyproject.toml uv.lock ./
RUN uv pip install --system -r pyproject.toml

COPY . .


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]