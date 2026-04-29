#!/usr/bin/env bash
# reproduce.sh — runs all experiments from the paper in sequence
# Expected runtime on A100-SXM4-40GB: ~3-4 hours total
# Run setup.sh first.

set -e

cd /workspace

echo "============================================================"
echo "  Cross-Attention Forking — Full Reproduction"
echo "  Paper: Architectural Access Control via Cross-Attention"
echo "         Forking in Diffusion Transformers"
echo "============================================================"

# ── Phase 1–3: Routing and gradient isolation (Section 3.1–3.2) ──
echo ""
echo "[STEP 1/6] Phase 1-3: Auth routing + adversarial gradient isolation"
echo "  Expected: all PASS, max privileged grad = 0.0e+00"
python run_phase1_3.py \
    --num_forked_blocks 2 \
    --verify_passes 1000 \
    --adv_steps 100
echo "  → Results saved to phase1_3_report_fork2.json"

# ── RT-1 regression (Section 3.3, Table 1) ────────────────────
echo ""
echo "[STEP 2/6] RT-1 regression: privileged route bit-identity check"
echo "  Expected: fork auth=1 MSE == base MSE, peak abs diff = 0.00e+00"
python run_rt1_regression.py
echo "  → Table 1 numbers"

# ── G1 training, seed 0 (Section 4, Table 2) ──────────────────
echo ""
echo "[STEP 3/6] G1 humanoid training — seed 0 (5000 steps, ~60 min)"
python patch_g1_for_seeds.py run_fork2_g1.py
python run_fork2_g1.py \
    --num_forked_blocks 2 \
    --steps 5000 \
    --seed 0 \
    --output_tag seed0
echo "  → results_fork2_g1_seed0.json, model_fork2_g1_seed0.pt"

# ── G1 training, seeds 1 and 2 ────────────────────────────────
echo ""
echo "[STEP 4/6] G1 humanoid training — seeds 1 and 2"
python run_fork2_g1.py \
    --num_forked_blocks 2 \
    --steps 5000 \
    --seed 1 \
    --output_tag seed1
python run_fork2_g1.py \
    --num_forked_blocks 2 \
    --steps 5000 \
    --seed 2 \
    --output_tag seed2
echo "  → Table 2 seed sweep complete"

# ── Mechanistic analysis (Section 5) ──────────────────────────
echo ""
echo "[STEP 5/6] Mechanistic analysis: attention patterns, geometry, FFN"
echo "  Expected: 75 deg action subspace angle, 0/10 image token overlap"
python analyze_fork_mechanistic.py
echo "  → Figures 2, 3 and Table 2 FFN rows"

# ── LoRA comparison (Section 6, Table 3) ──────────────────────
echo ""
echo "[STEP 6/6] Resource comparison vs LoRA"
echo "  Expected: fork 17% less VRAM, 25% less latency vs full-DiT LoRA"
python run_lora_comparison.py
echo "  → Table 3"

echo ""
echo "============================================================"
echo "  Reproduction complete."
echo "  Key output files:"
echo "    phase1_3_report_fork2.json  — Sections 3.1-3.2"
echo "    results_fork2_g1_seed*.json — Table 2"
echo "    trajectory_comparison.npz   — Figure 2"
echo "    lora_comparison.json        — Table 3"
echo "============================================================"
