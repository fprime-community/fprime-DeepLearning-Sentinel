#!/usr/bin/env bash
# One-shot setup for a rented GPU box, after `rsync` of src/ scripts/ tests/ docs/ *.md
# requirements.txt .gitignore (never .env, never runs/). Holds no secrets: R2 values are
# exported into the fitting shell over stdin and live nowhere on the box (D15, D26).
#
#   ssh pod "bash -s" < scripts/pod_setup.sh                     # then, with the env exported:
#   .venv/bin/python scripts/fit_folds.py --task m1-g8.9.10 --detector gru-quantile \
#                                       --device cuda --threads 1 --determinism-check
set -o pipefail
cd /root/Sentinel
echo START $(date -u +%FT%TZ)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3.12-venv > /root/apt.log 2>&1 && echo APT_OK || { echo APT_FAILED; tail -3 /root/apt.log; exit 1; }
python3.12 -m venv .venv && .venv/bin/pip install -q --upgrade pip 2>&1 | tail -1
# CUDA 12.6 build first: the pod driver (550, CUDA 12.4) cannot run PyPI's default cu130 wheel,
# and ==2.13.0 in requirements.txt accepts 2.13.0+cu126 so nothing is re-downloaded.
.venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cu126 "torch==2.13.0+cu126" 2>&1 | grep -vE "^\s*$" | tail -2
.venv/bin/pip install -q -r requirements.txt 2>&1 | grep -vE "^\s*$" | tail -2
echo PIP_DONE $(date -u +%FT%TZ)
.venv/bin/python -c "import torch,numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-', 'numpy', numpy.__version__)"
.venv/bin/python -m pytest -q 2>&1 | tail -3
echo PYTEST_EXIT ${PIPESTATUS[0]} $(date -u +%FT%TZ)
PYTHONPATH=src .venv/bin/python -m sentinel_eval selftest 2>&1 | tail -1
PYTHONPATH=src .venv/bin/python scripts/check_no_list.py 2>&1 | head -1
echo SETUP_DONE $(date -u +%FT%TZ)
