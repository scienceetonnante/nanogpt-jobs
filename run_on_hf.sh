#!/usr/bin/env bash
# Run a Python file of this folder, unchanged, on a Hugging Face GPU.
#
#   ./run_on_hf.sh train.py [flavor] [timeout]    # by default a100-large and 2h, for example: l4x1 4h
#
# 1. This folder is copied to the bucket <you>/<folder name>, in code/ (without hidden files and out/).
# 2. On the GPU machine, out/ is a link to out/ in the bucket: what the code writes there is kept on the Hub.
# 3. The packages of requirements.txt are installed and the file runs, as with `uv run` on your machine.
set -euo pipefail

SCRIPT=$1
FLAVOR=${2:-a100-large}
TIMEOUT=${3:-2h}
OUT=out                                    # folder where the code saves its results
PROJECT=$(basename "$PWD")
BUCKET="$(hf auth whoami -q)/$PROJECT"

# 1. Copy the code to the bucket
hf buckets create "$BUCKET" --private --exist-ok
hf sync . "hf://buckets/$BUCKET/code" --delete --exclude ".*" --exclude "*__pycache__*" --exclude "$OUT/*"

# 2 and 3. Run it on a GPU machine, with out/ linked to the bucket
hf jobs run --flavor "$FLAVOR" --timeout "$TIMEOUT" --detach --name "$PROJECT-$(basename "$SCRIPT" .py)" \
    --volume "hf://buckets/$BUCKET:/bucket" \
    ghcr.io/astral-sh/uv:python3.12-bookworm \
    bash -c "
        set -e
        cp -r /bucket/code /work && cd /work
        mkdir -p /bucket/$OUT && ln -s /bucket/$OUT $OUT
        PYTHONUNBUFFERED=1 uv run --with-requirements requirements.txt $SCRIPT
    "

echo
echo "Follow the job:              hf jobs logs -f <job id>"
echo "When it is done, download:   hf sync hf://buckets/$BUCKET/$OUT ./$OUT"
