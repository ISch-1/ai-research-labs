# Lab A model API. Build from the repository root:
#   docker build -f lab-a-predictive/Dockerfile -t ai-research-labs/lab-a-api .
#   docker run -p 8000:8000 ai-research-labs/lab-a-api       # -> http://localhost:8000/docs
#
# Stage 1 trains and exports the models (downloads the datasets);
# stage 2 contains only the API runtime and the exported model files.

FROM python:3.12-slim AS trainer
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY pyproject.toml README.md ./
COPY common common
COPY lab-a-predictive/src lab-a-predictive/src
RUN pip install --no-cache-dir ".[lab-a]"
COPY lab-a-predictive/configs lab-a-predictive/configs
ARG EXPORT_CONFIG=lab-a-predictive/configs/a5_export.yaml
ARG GIT_COMMIT=unknown
ENV AIRLAB_DATA_DIR=/tmp/data GIT_COMMIT=${GIT_COMMIT}
RUN python -m lab_a.serving.export --config ${EXPORT_CONFIG} \
    && mkdir -p /models && cp -r "$(python -c "import yaml,sys; print(yaml.safe_load(open('${EXPORT_CONFIG}'))['model_dir'])")"/. /models/

FROM python:3.12-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
WORKDIR /app
COPY pyproject.toml README.md ./
COPY common common
COPY lab-a-predictive/src lab-a-predictive/src
RUN pip install --no-cache-dir ".[serve]" && rm -rf common lab-a-predictive
COPY --from=trainer /models /app/models
ENV MODEL_DIR=/app/models PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "lab_a.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
