# ── Stage 1: build ──────────────────────────────────────────────
FROM python:3.11-alpine AS builder

RUN apk add --no-cache uv

WORKDIR /build

COPY pyproject.toml README.md ./
COPY pyscnomics/ pyscnomics/

# Install into a virtual env
RUN uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --no-cache . \
    && find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type d -name "test" -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -path "*/matplotlib/mpl-data/sample_data" -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -name "*.pyc" -delete 2>/dev/null || true \
    && find /opt/venv -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true

# ── Stage 2: runtime ───────────────────────────────────────────
FROM python:3.11-alpine

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=9999 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy only the virtual env from builder
COPY --from=builder /opt/venv /opt/venv

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/api/')" || exit 1

CMD ["sh", "-c", "pyscnomics --api 1 --port ${PORT}"]
