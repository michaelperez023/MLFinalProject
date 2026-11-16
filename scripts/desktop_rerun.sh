#!/usr/bin/env bash
# Unattended re-run of Project Marrow on a Linux machine with an NVIDIA GPU and conda:
#
#   bash scripts/desktop_rerun.sh WORK_DIR
#
#  1. creates two conda environments: Python 3.8 with requirements-2019.txt, which can load the
#     2019 scikit-learn models, and Python 3.12 with scripts/requirements-rerun.txt (PyTorch, Keras)
#  2. downloads the dataset from Mendeley Data, checks its SHA-256 and extracts it
#  3. builds the pipeline's train/test pickles with Data_Preprocessing/PictureProcessor.py
#  4. scores the 2019 models with ML_Models_Saved/Model_Benchmark.py and checks that the AVG
#     column of Results/Test_Results.xlsx is reproduced
#  5. retrains the 2019 CNN architecture with CNN_Model/retrain_2026.py (2026 addition)
#  6. writes WORK_DIR/results/rerun_2026.json and WORK_DIR/results/figures/
#
# Everything is written inside WORK_DIR: environments, package caches, the dataset, logs and
# results. The repository is only read. WORK_DIR must be a new or empty folder, outside the
# repository; the script marks it as its own (.project-marrow-rerun) and later accepts it again.
# Each step records that it has finished, so the script can be stopped and started again:
# finished steps are skipped, and an interrupted retrain continues with the runs it has not
# finished. A step is redone when its inputs change, including the requirements of the
# environment it runs in. Logs and timings: WORK_DIR/logs/.
#
# Settings (environment variables):
#   CONDA_EXE        conda executable (default: ~/anaconda3/bin/conda)
#   MLF_SEEDS        seeds of the CNN retrain (default: "42 43 44")
#   MLF_CONV_BLOCKS  CNN depths to retrain (default: "2 3 4")
#   MLF_EPOCHS       maximum epochs per retrain (default: 40)
#   MLF_IMAGE_SIZE   CNN input size in pixels (default: 256)
#   MLF_ALLOW_CPU=1  retrain on the CPU when PyTorch sees no GPU (default: stop)
# For testing the script on a small fake dataset:
#   MLF_DATASET_URL, MLF_DATASET_SHA256, MLF_DATASET_SIZE, MLF_EXPECTED_COUNTS (JSON),
#   MLF_PY312_REQUIREMENTS (another requirements file for the Python 3.12 environment)

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: bash scripts/desktop_rerun.sh WORK_DIR" >&2
    exit 2
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
inside_repo() { case "$1/" in "$REPO"/*) return 0 ;; *) return 1 ;; esac; }
case "$1" in /*) WORK="$1" ;; *) WORK="$PWD/$1" ;; esac
# Resolve symbolic links in the part of WORK_DIR that exists before creating anything
existing="$WORK" missing=""
while [[ ! -d "$existing" ]]; do
    if [[ -e "$existing" || -L "$existing" ]]; then
        echo "WORK_DIR must be a folder; $existing is a file or a broken link" >&2
        exit 2
    fi
    name="${existing##*/}"
    existing="${existing%/*}"
    existing="${existing:-/}"
    case "$name" in
        "") ;;  # a repeated or trailing slash
        . | ..)
            echo "WORK_DIR must not contain . or .. after the part that exists: $1" >&2
            exit 2
            ;;
        *) missing="/$name$missing" ;;
    esac
done
WORK="$(cd "$existing" && pwd -P)"
[[ -z "$missing" ]] || WORK="${WORK%/}$missing"
if inside_repo "$WORK"; then
    echo "WORK_DIR must not be inside the repository ($REPO)" >&2
    exit 2
fi
MARKER="$WORK/.project-marrow-rerun"
if [[ -d "$WORK" && ! -f "$MARKER" && -n "$(ls -A "$WORK")" ]]; then
    echo "WORK_DIR must be a new or empty folder; $WORK holds other files" >&2
    exit 2
fi
mkdir -p "$WORK"
[[ -f "$MARKER" ]] || echo "Working folder of $REPO/scripts/desktop_rerun.sh" > "$MARKER"

CONDA="${CONDA_EXE:-$HOME/anaconda3/bin/conda}"
SEEDS="${MLF_SEEDS:-42 43 44}"
CONV_BLOCKS="${MLF_CONV_BLOCKS:-2 3 4}"
EPOCHS="${MLF_EPOCHS:-40}"
IMAGE_SIZE="${MLF_IMAGE_SIZE:-256}"
ALLOW_CPU="${MLF_ALLOW_CPU:-0}"
PY312_REQUIREMENTS="${MLF_PY312_REQUIREMENTS:-$REPO/scripts/requirements-rerun.txt}"

ENVS="$WORK/envs"
LOGS="$WORK/logs"
STATE="$WORK/state"
RESULTS="$WORK/results"
PY38="$ENVS/py38/bin/python"
PY312="$ENVS/py312/bin/python"
mkdir -p "$ENVS" "$LOGS" "$STATE" "$RESULTS" "$WORK/tmp" "$WORK/cache"

# Keep every cache and temporary file inside WORK_DIR
export TMPDIR="$WORK/tmp"
export XDG_CACHE_HOME="$WORK/cache/xdg"
export PIP_CACHE_DIR="$WORK/cache/pip"
export CONDA_PKGS_DIRS="$WORK/cache/conda-pkgs"
export CONDA_REGISTER_ENVS=false                  # do not list the environments in ~/.conda/environments.txt
export MPLCONFIGDIR="$WORK/cache/matplotlib"
export KERAS_HOME="$WORK/cache/keras"
export TORCH_HOME="$WORK/cache/torch"
export TRITON_CACHE_DIR="$WORK/cache/triton"
export CUDA_CACHE_PATH="$WORK/cache/nv"
export PYTHONPYCACHEPREFIX="$WORK/cache/pycache"  # no __pycache__ folders in the repository
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PYTHONUNBUFFERED=1
export KERAS_BACKEND=torch
# The repository's modules, without installing it
export PYTHONPATH="$REPO/HPC:$REPO/CNN_Model"

if command -v flock > /dev/null; then
    exec 9> "$WORK/.lock"
    flock -n 9 || { echo "Another desktop_rerun.sh is using $WORK" >&2; exit 1; }
fi

now() { date '+%Y-%m-%d %H:%M:%S'; }

sha256() {  # sha256 FILE...: one hash of the files' contents
    if command -v sha256sum > /dev/null; then
        cat "$@" | sha256sum | cut -d' ' -f1
    else
        cat "$@" | shasum -a 256 | cut -d' ' -f1
    fi
}

timing() { printf '%s\t%s\t%s\t%s\t%s\n' "$@" >> "$LOGS/timings.tsv"; }

# step NAME KEY COMMAND...: runs COMMAND with a log and a timing record, unless the step
# already finished with the same KEY (a hash of its inputs and settings).
step() {
    local name="$1" key="$2"
    shift 2
    local log="$LOGS/$name.log" done_file="$STATE/$name.done" start end status
    if [[ -f "$done_file" && "$(cat "$done_file")" == "$key" ]]; then
        echo "[$(now)] $name: finished before, skipped"
        timing "$name" "$(date +%s)" "$(date +%s)" 0 skipped
        return 0
    fi
    rm -f "$done_file"
    echo "[$(now)] $name: started, log in $log"
    start=$(date +%s)
    set +e
    { echo "[$(now)] $name"; (set -e; "$@"); } 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
    set -e
    end=$(date +%s)
    if [[ $status -ne 0 ]]; then
        timing "$name" "$start" "$end" "$((end - start))" failed
        echo "[$(now)] $name: FAILED (status $status) after $((end - start)) s; see $log" >&2
        exit "$status"
    fi
    timing "$name" "$start" "$end" "$((end - start))" ok
    echo "$key" > "$done_file"
    echo "[$(now)] $name: finished in $((end - start)) s"
}

create_env() {  # create_env PREFIX PYTHON_VERSION REQUIREMENTS [PIP_OPTIONS...]
    local prefix="$1" version="$2" requirements="$3"
    shift 3
    rm -rf "$prefix"
    # conda writes a usage token and its list of environments to ~/.conda, so it gets a home
    # folder inside WORK_DIR
    mkdir -p "$WORK/cache/conda-home"
    HOME="$WORK/cache/conda-home" "$CONDA" create --yes --quiet --prefix "$prefix" \
        --solver=libmamba --override-channels --channel conda-forge "python=$version" pip
    "$prefix/bin/python" -m pip install "$@" --requirement "$requirements"
    "$prefix/bin/python" -m pip check
}

check_gpu() {
    "$PY312" - << 'EOF'
import os
import sys

import torch

if torch.cuda.is_available():
    print("PyTorch", torch.__version__, "CUDA", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))
    sys.exit(0)
print("PyTorch", torch.__version__, "(CUDA", torch.version.cuda, ") sees no GPU: is the NVIDIA driver installed and recent enough?")
sys.exit(0 if os.environ.get("MLF_ALLOW_CPU") == "1" else 1)
EOF
}

record_environment() {
    "$PY38" "$REPO/scripts/env_info.py" > "$LOGS/env_py38.json"
    "$PY312" "$REPO/scripts/env_info.py" > "$LOGS/env_py312.json"
    "$PY38" -m pip freeze > "$LOGS/pip_freeze_py38.txt"
    "$PY312" -m pip freeze > "$LOGS/pip_freeze_py312.txt"
    if command -v nvidia-smi > /dev/null; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$LOGS/gpu.txt" || true
        nvidia-smi > "$LOGS/nvidia-smi.txt" 2>&1 || true
    else
        echo "no nvidia-smi" > "$LOGS/gpu.txt"
    fi
    { grep -m1 "model name" /proc/cpuinfo 2> /dev/null | cut -d: -f2- | sed 's/^ //' \
        || sysctl -n machdep.cpu.brand_string 2> /dev/null || uname -m; } > "$LOGS/cpu.txt"
    # The commit, with -dirty for changed tracked files. Not `git describe --dirty`, which
    # rewrites .git/index when its file timestamps are out of date; git status does not.
    local commit
    commit="$(git -C "$REPO" rev-parse HEAD 2> /dev/null)" || commit=unknown
    if [[ "$commit" != unknown ]] && [[ -n "$(GIT_OPTIONAL_LOCKS=0 git -C "$REPO" status --porcelain \
        --untracked-files=no 2> /dev/null)" ]]; then
        commit="$commit-dirty"
    fi
    echo "$commit" > "$LOGS/repo_commit.txt"
    MLF_SEEDS="$SEEDS" MLF_CONV_BLOCKS="$CONV_BLOCKS" MLF_EPOCHS="$EPOCHS" MLF_IMAGE_SIZE="$IMAGE_SIZE" \
        "$PY312" - "$LOGS/settings.json" << 'EOF'
import json
import os
import sys

settings = {
    "seeds": [int(s) for s in os.environ["MLF_SEEDS"].split()],
    "conv_blocks": [int(c) for c in os.environ["MLF_CONV_BLOCKS"].split()],
    "max_epochs": int(os.environ["MLF_EPOCHS"]),
    "image_size": int(os.environ["MLF_IMAGE_SIZE"]),
    "allow_cpu": os.environ.get("MLF_ALLOW_CPU") == "1",
}
with open(sys.argv[1], "w") as f:
    json.dump(settings, f, indent=1)
EOF
}

echo "[$(now)] Project Marrow re-run: repository $REPO, working folder $WORK"
[[ -x "$CONDA" ]] || { echo "conda not found at $CONDA; set CONDA_EXE" >&2; exit 1; }

# 1. Environments, rebuilt when their requirements change. The keys of the steps that run in
# an environment include its key, so that those steps are redone with a rebuilt environment.
PY38_KEY="py3.8-$(sha256 "$REPO/requirements-2019.txt")"
PY312_KEY="py3.12-$(sha256 "$PY312_REQUIREMENTS")"
step env_py38 "$PY38_KEY" \
    create_env "$ENVS/py38" 3.8 "$REPO/requirements-2019.txt" --only-binary :all:
step env_py312 "$PY312_KEY" \
    create_env "$ENVS/py312" 3.12 "$PY312_REQUIREMENTS"
step gpu_check "$PY312_KEY-$ALLOW_CPU" check_gpu
record_environment

# 2. Dataset
DATASET_ARGS=()
if [[ -n "${MLF_DATASET_URL:-}" ]]; then DATASET_ARGS+=(--url "$MLF_DATASET_URL"); fi
if [[ -n "${MLF_DATASET_SHA256:-}" ]]; then DATASET_ARGS+=(--sha256 "$MLF_DATASET_SHA256"); fi
if [[ -n "${MLF_DATASET_SIZE:-}" ]]; then DATASET_ARGS+=(--size "$MLF_DATASET_SIZE"); fi
if [[ -n "${MLF_EXPECTED_COUNTS:-}" ]]; then DATASET_ARGS+=(--expected-counts "$MLF_EXPECTED_COUNTS"); fi
step dataset "$(sha256 "$REPO/scripts/fetch_dataset.py")-${MLF_DATASET_SHA256:-official}-$PY312_KEY" \
    "$PY312" "$REPO/scripts/fetch_dataset.py" --downloads "$WORK/downloads" --data-dir "$WORK/data" \
    ${DATASET_ARGS[@]+"${DATASET_ARGS[@]}"}

# 3. The pipeline's pickles, built as in 2019 (Python 3.8, Pillow 6, nearest-neighbour resizing)
PREPROCESS_KEY="$(sha256 "$WORK/data/chest_xray.json" "$REPO/Data_Preprocessing/PictureProcessor.py")"
step preprocess "$PREPROCESS_KEY-$PY38_KEY" \
    "$PY38" "$REPO/Data_Preprocessing/PictureProcessor.py" --dataset-dir "$WORK/data/chest_xray" \
    --out-dir "$WORK/pipeline" --seed 4612

# 4. The 2019 models on the test split, compared with Results/Test_Results.xlsx
SCORE_KEY="$(sha256 "$STATE/preprocess.done" "$REPO/ML_Models_Saved/Model_Benchmark.py" \
    "$REPO/HPC/ML_DataPipeline/DataPipeline.py" "$REPO/CNN_Model/pneumonia_cnn.py")"
step score_2019_sklearn "$SCORE_KEY-$PY38_KEY" \
    "$PY38" "$REPO/ML_Models_Saved/Model_Benchmark.py" --data-dir "$WORK/pipeline" --models "*.pickle" \
    --out "$RESULTS/benchmark_2019_sklearn.xlsx" --metrics-json "$RESULTS/benchmark_2019_sklearn.json"
step score_2019_cnn "$SCORE_KEY-$PY312_KEY" \
    "$PY312" "$REPO/ML_Models_Saved/Model_Benchmark.py" --data-dir "$WORK/pipeline" --models "*.h5" \
    --out "$RESULTS/benchmark_2019_cnn.xlsx" --metrics-json "$RESULTS/benchmark_2019_cnn.json"
CHECK_KEY="$(sha256 "$RESULTS/benchmark_2019_sklearn.json" "$RESULTS/benchmark_2019_cnn.json" \
    "$REPO/scripts/check_2019_results.py" "$REPO/Results/Test_Results.xlsx")"
step check_2019 "$CHECK_KEY-$PY312_KEY" \
    "$PY312" "$REPO/scripts/check_2019_results.py" --expected "$REPO/Results/Test_Results.xlsx" \
    --reproduced "$RESULTS/benchmark_2019_sklearn.json" "$RESULTS/benchmark_2019_cnn.json" \
    --out "$RESULTS/check_2019.json"

# 5. 2026 retrain of the CNN. Started again with the same code, data and environment, it skips
# the runs it finished; when one of these changed, the earlier runs and resized images are
# removed first, as retrain_2026.py itself only compares the training settings.
read -r -a SEED_LIST <<< "$SEEDS"
read -r -a CONV_BLOCK_LIST <<< "$CONV_BLOCKS"
GPU_ARGS=(--require-gpu)
if [[ "$ALLOW_CPU" == "1" ]]; then GPU_ARGS=(); fi
RUNS_KEY="$(sha256 "$WORK/data/chest_xray.json" "$REPO/CNN_Model/retrain_2026.py" \
    "$REPO/CNN_Model/pneumonia_cnn.py")-$PY312_KEY"
if [[ -f "$STATE/runs.key" && "$(cat "$STATE/runs.key")" != "$RUNS_KEY" ]]; then
    echo "[$(now)] The code, data or environment of the retrain changed; removing the earlier runs"
    rm -rf "$WORK/runs" "$WORK/cache/cnn"
fi
echo "$RUNS_KEY" > "$STATE/runs.key"
step retrain_2026 "$RUNS_KEY-$SEEDS-$CONV_BLOCKS-$EPOCHS-$IMAGE_SIZE" \
    "$PY312" "$REPO/CNN_Model/retrain_2026.py" --dataset-dir "$WORK/data/chest_xray" \
    --cache-dir "$WORK/cache/cnn" --out-dir "$WORK/runs" --conv-blocks "${CONV_BLOCK_LIST[@]}" \
    --seeds "${SEED_LIST[@]}" \
    --epochs "$EPOCHS" --image-size "$IMAGE_SIZE" ${GPU_ARGS[@]+"${GPU_ARGS[@]}"}

# 6. Summary and figures (always redone; it only reads the files above)
"$PY312" "$REPO/scripts/summarize_rerun.py" --work-dir "$WORK" --repo "$REPO" 2>&1 | tee "$LOGS/summary.log"

echo "[$(now)] Done. Results: $RESULTS/rerun_2026.json and $RESULTS/figures/"
