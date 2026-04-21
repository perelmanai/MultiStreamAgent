#!/bin/bash
# Activate the conda env and run a command.
#
# Usage:
#   ./env/public/run.sh python app.py
#   ./env/public/run.sh pip install some-package
#
# Set GEMINI_API_KEY in your environment before running:
#   export GEMINI_API_KEY="your-api-key"

set -eo pipefail

ENV_NAME="multi_stream_agent"

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo "Warning: GEMINI_API_KEY is not set. Gemini backend will not work."
    echo "  Set it with: export GEMINI_API_KEY='your-api-key'"
fi

# Initialize conda shell functions
eval "$(conda shell.bash hook 2>/dev/null)" || {
    echo "Error: conda not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
}

conda activate "${ENV_NAME}" 2>/dev/null || {
    echo "Error: conda env '${ENV_NAME}' not found. Run: bash env/public/setup_conda.sh"
    exit 1
}

exec "$@"
