#!/usr/bin/env bash
# setup.sh — one-shot environment setup for cross-attention forking experiments
# Tested on Vast.ai A100 (CUDA 12.4, Python 3.12, Ubuntu 22.04)
# Run once before any experiment script.

set -e

echo "============================================================"
echo "  Cross-Attention Forking — Environment Setup"
echo "============================================================"

# ── 1. Clone RDT backbone ──────────────────────────────────────
echo ""
echo "[1/4] Cloning RoboticsDiffusionTransformer..."
if [ ! -d "/workspace/RoboticsDiffusionTransformer" ]; then
    git clone https://github.com/thu-ml/RoboticsDiffusionTransformer.git \
        /workspace/RoboticsDiffusionTransformer
else
    echo "  Already cloned, skipping."
fi

# ── 2. Install Python dependencies ────────────────────────────
echo ""
echo "[2/4] Installing Python dependencies..."

pip install --upgrade pip --quiet

pip install \
    torch==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --quiet

pip install \
    "diffusers>=0.27.0" \
    "huggingface_hub>=0.23.0" \
    "transformers==4.36.0" \
    "timm==0.9.12" \
    "packaging>=24.0" \
    safetensors \
    sentencepiece \
    protobuf \
    matplotlib \
    numpy \
    pandas \
    pyarrow \
    --quiet

echo "  Dependencies installed."

# ── 3. Copy experiment scripts into workspace ──────────────────
echo ""
echo "[3/4] Copying experiment scripts to /workspace..."

SCRIPTS=(
    run_crossattn_fork_170m_2.py
    auth_routing.py
    verification_harness.py
    adversarial_gradient_test.py
    run_phase1_3.py
    run_rt1_regression.py
    run_fork2_g1.py
    patch_g1_for_seeds.py
    analyze_fork_mechanistic.py
    gen_trajectories.py
    run_lora_comparison.py
    test_shared_self_attention_interleaving.py
)

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

for f in "${SCRIPTS[@]}"; do
    if [ -f "$REPO_DIR/$f" ]; then
        cp "$REPO_DIR/$f" /workspace/"$f"
        echo "  copied $f"
    else
        echo "  WARNING: $f not found in $REPO_DIR"
    fi
done

# ── 4. Verify GPU ──────────────────────────────────────────────
echo ""
echo "[4/4] Verifying GPU..."
python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "============================================================"
echo "  Setup complete. See README.md for experiment commands."
echo "============================================================"
