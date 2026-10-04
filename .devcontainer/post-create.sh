#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_NAME="lerobot"
CONDA_BASE="${HOME}/miniforge3"
MARKER="# >>> rbe501 lerobot conda >>>"

echo "==> Installing system packages (MuJoCo viewer, build tools)..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  pkg-config \
  libgl1 \
  libglfw3 \
  libglib2.0-0 \
  libgomp1 \
  libsm6 \
  libxext6 \
  libxrender1

if [[ ! -x "${CONDA_BASE}/bin/conda" ]]; then
  echo "Miniforge not found at ${CONDA_BASE}/bin/conda" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  echo "==> Creating conda env '${CONDA_ENV_NAME}' (Python 3.12)..."
  conda create -y -n "${CONDA_ENV_NAME}" python=3.12
else
  echo "==> Conda env '${CONDA_ENV_NAME}' already exists; skipping create."
fi

conda activate "${CONDA_ENV_NAME}"

echo "==> Installing ffmpeg (LeRobot video / TorchCodec)..."
conda install -y -c conda-forge ffmpeg

echo "==> Installing LeRobot (SO-101 Feetech + record/train workflows)..."
python -m pip install -U pip wheel
python -m pip install "lerobot[feetech,core_scripts,training]"

echo "==> Installing this repo (editable)..."
python -m pip install -e .

echo "==> Smoke tests..."
python -c "import mujoco; print('mujoco', mujoco.__version__)"
python -c "import lerobot; print('lerobot', lerobot.__version__)"

if ! grep -qF "${MARKER}" "${HOME}/.bashrc" 2>/dev/null; then
  cat >>"${HOME}/.bashrc" <<EOF

${MARKER}
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate ${CONDA_ENV_NAME}
# <<< rbe501 lerobot conda <<<
EOF
fi

echo "==> Done. Reopen the terminal or rebuild if the Python interpreter is not selected."
