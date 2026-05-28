FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/uamfst

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["python", "-m"]
CMD ["uamfst.cellarmaster.vintage", "--gin_file", "configs/experiment/_smoke.gin"]
