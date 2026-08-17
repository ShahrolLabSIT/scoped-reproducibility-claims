# syntax=docker/dockerfile:1.7

# One amd64 image is built once per verification and then addressed by its registry digest on every Linux host, including the ARM runner through QEMU.
FROM --platform=linux/amd64 python:3.12.10-slim-bookworm@sha256:97983fa8cc88343512862c62307159a82261c3528dc025f79e5a3f7af43e50b4 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SRC_CASE_CONTAINER=linux-amd64 \
    SRC_CASE_IMAGE=locally-built \
    TZ=UTC

WORKDIR /workspace

FROM base AS analysis

COPY pyproject.toml README.md ./
COPY Dockerfile ./
COPY repro_case ./repro_case
RUN python -m pip install --no-cache-dir .

COPY case-study ./case-study
COPY .github/workflows ./.github/workflows
COPY tests ./tests
COPY main.tex manuscript.sty references.bib ./
COPY figures ./figures
COPY tables ./tables

FROM base AS runtime

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        biber \
        latexmk \
        texlive-bibtex-extra \
        texlive-fonts-recommended \
        texlive-latex-extra \
        texlive-publishers \
    && rm -rf /var/lib/apt/lists/*

COPY --from=analysis /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=analysis /usr/local/bin/src-case /usr/local/bin/src-case
COPY --from=analysis /workspace /workspace

ENTRYPOINT ["src-case"]
