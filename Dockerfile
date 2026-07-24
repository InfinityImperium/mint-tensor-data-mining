# MINT experiment environment (CPU).
# The image contains ONLY the environment; the repo (code + datasets) is
# volume-mounted at runtime, so code changes never require a rebuild.
#
#   docker build -t mint-env .
#   docker run --rm -it -v "$PWD":/work mint-env python mplot_python/taipei_test.py
#
# pyscamp 4.0.1 has no prebuilt wheels: pip builds it from source, which is
# why build-essential + cmake are installed (build takes a few minutes).

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

ENV MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

# Largest layer first so it caches independently of requirement changes.
RUN pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Sanity check at build time: fail the build if anything doesn't import.
RUN python -c "import torch, tensorly, kneed, scipy, pandas, matplotlib, pyscamp; \
    import numpy as np; \
    m = pyscamp.abjoin_matrix(np.random.randn(2000), np.random.randn(2000), 50, mheight=32, mwidth=32, threshold=-1); \
    assert m.shape == (32, 32); print('pyscamp matrix-summary OK')"

WORKDIR /work
CMD ["bash"]
