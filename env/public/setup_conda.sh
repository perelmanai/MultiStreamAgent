#!/bin/bash
# Set up a standalone conda environment for MultiStreamAgent.
#
# Usage:
#   bash env/public/setup_conda.sh           # Create the environment
#   conda activate multi_stream_agent        # Activate it
#   bash env/public/setup_conda.sh --force   # Recreate from scratch
#
# Prerequisites:
#   - conda (Miniconda or Anaconda)
#   - NVIDIA GPU with CUDA 12.4+ (for local model inference)

set -eo pipefail

ENV_NAME="multi_stream_agent"
PYTHON_VERSION="3.12"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
FORCE=false
if [[ "${1:-}" == "--force" ]]; then
    FORCE=true
fi

# ---------------------------------------------------------------------------
# Ensure conda is usable
# ---------------------------------------------------------------------------
eval "$(conda shell.bash hook 2>/dev/null)" || true
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Install from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# ---------------------------------------------------------------------------
# Handle existing environment
# ---------------------------------------------------------------------------
if conda env list | grep -qw "${ENV_NAME}"; then
    if [[ "${FORCE}" == "true" ]]; then
        echo "Removing existing '${ENV_NAME}' environment..."
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "Environment '${ENV_NAME}' already exists."
        echo "  Use --force to recreate, or just: conda activate ${ENV_NAME}"
        exit 0
    fi
fi

# ---------------------------------------------------------------------------
# Step 1: Create conda environment
# ---------------------------------------------------------------------------
echo "=== Step 1: Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VERSION}) ==="
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
conda activate "${ENV_NAME}"
echo "Active Python: $(which python) ($(python --version))"

pip install uv
export VIRTUAL_ENV="${CONDA_PREFIX}"

# ---------------------------------------------------------------------------
# Step 2: Install PyTorch with CUDA 12.4
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Installing PyTorch (CUDA 12.4) ==="
uv pip install torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(f'  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# ---------------------------------------------------------------------------
# Step 3: Install dependencies
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: Installing dependencies ==="
uv pip install \
    "transformers>=4.43.0" \
    "huggingface-hub" \
    "safetensors" \
    "accelerate" \
    "gradio" \
    "google-genai"

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "import transformers; print(f'transformers {transformers.__version__}')"
python -c "import google.genai; print(f'google-genai {google.genai.__version__}')"

echo ""
echo "========================================"
echo "Environment '${ENV_NAME}' is ready!"
echo "========================================"
echo ""
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Run with:"
echo "  export GEMINI_API_KEY='your-api-key'"
echo "  ./env/public/run.sh python app.py"
