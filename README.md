# Architectural Access Control via Cross-Attention Forking in Diffusion Transformers

**Hasan Ahmed** · Independent Researcher  
📄 [Paper (Zenodo)](https://zenodo.org/19834989) · 🤖 Built on [RDT-170M](https://huggingface.co/robotics-diffusion-transformer/rdt-170m)

---

## Overview

This repository contains the full implementation for the paper. The core idea is **cross-attention forking**: a structural modification to a pretrained Diffusion Transformer (DiT) action head that creates two independently trainable branches sharing a frozen backbone.

- **Whitelisted branch** — always-on default, handles public social behaviors (waving, handshaking, greetings)
- **Privileged branch** — gated by an upstream auth signal, holds task capabilities (manipulation, household access)

Training one branch never updates the other. At inference, exactly one branch executes per forward pass. The non-selected branch consumes zero FLOPs.

**Results on RDT-170M:**

| Metric | Result |
|---|---|
| Action subspace angle | 75.0° ± 1.3° |
| Top-10 image token overlap | 0 / 10 |
| Frozen parameter drift | 0.00 (by construction) |
| VRAM vs matched-scope LoRA | −17% |
| Latency vs full-DiT LoRA | −25% |

---

## Repository Structure

```
.
├── setup.sh                          # one-shot environment setup
├── reproduce.sh                      # runs all paper experiments in order
│
├── run_crossattn_fork_170m_2.py      # core architecture: CrossAttnForkedRDT
├── auth_routing.py                   # auth scalar interface: AuthRoutedRDT
├── verification_harness.py           # forward hook counter (Section 3.1)
├── adversarial_gradient_test.py      # gradient isolation Tests A/B/C (Section 3.2)
│
├── run_phase1_3.py                   # orchestrates Sections 3.1–3.2
├── run_rt1_regression.py             # bit-identity check on RT-1 (Table 1)
├── run_fork2_g1.py                   # G1 humanoid training run (Table 2)
├── patch_g1_for_seeds.py             # adds --seed flag to run_fork2_g1.py
├── analyze_fork_mechanistic.py       # attention patterns, geometry, FFN (Section 5)
├── gen_trajectories.py               # trajectory comparison data (Figure 2)
├── run_lora_comparison.py            # VRAM/latency/FLOPs vs LoRA (Table 3)
│
└── test_shared_self_attention_interleaving.py  # shared block invariance check (unpublished)
```

---

## Requirements

- Python 3.12
- CUDA 12.4
- NVIDIA A100 (40GB) recommended — minimum ~24GB VRAM for bs=4 training

---

## Setup

```bash
git clone https://github.com/has9800/vlm-cross-attn-fork-code.git
cd crossattn-fork
chmod +x setup.sh reproduce.sh
./setup.sh
```

`setup.sh` will:
1. Clone the RDT backbone from `thu-ml/RoboticsDiffusionTransformer`
2. Install all Python dependencies
3. Copy experiment scripts to `/workspace`
4. Verify your GPU

---

## Reproducing the Paper

To run all experiments in sequence:

```bash
./reproduce.sh
```

To run individual experiments:

```bash
# Section 3.1-3.2: routing verification + adversarial gradient isolation
python run_phase1_3.py --num_forked_blocks 2 --verify_passes 1000 --adv_steps 100

# Section 3.3, Table 1: RT-1 regression / bit-identity check
python run_rt1_regression.py

# Section 4, Table 2: G1 training (run 3x with seeds 0, 1, 2)
python patch_g1_for_seeds.py run_fork2_g1.py
python run_fork2_g1.py --num_forked_blocks 2 --steps 5000 --seed 0 --output_tag seed0
python run_fork2_g1.py --num_forked_blocks 2 --steps 5000 --seed 1 --output_tag seed1
python run_fork2_g1.py --num_forked_blocks 2 --steps 5000 --seed 2 --output_tag seed2

# Section 5: mechanistic analysis (attention patterns, geometry, FFN amplification)
python analyze_fork_mechanistic.py

# Section 6, Table 3: LoRA comparison
python run_lora_comparison.py
```

---

## How the Fork Works

Each of the last N RDT blocks is modified as follows:

```python
def forward(self, x, c, mask=None, head="whitelisted"):
    x = x + self.attn(self.norm1(x))          # shared, frozen

    origin_x = x
    if head == "privileged":
        x = self.priv_cross_attn(self.priv_norm2(x), c, mask)
    else:
        x = self.wl_cross_attn(self.wl_norm2(x), c, mask)
    x = x + origin_x                          # forked, route-selected

    x = x + self.ffn(self.norm3(x))           # shared, frozen
    return x
```

The `if/else` is intentional — not a soft routing scalar. Only one branch executes per forward pass. The non-selected branch contributes zero activations to the residual stream and receives zero gradient during single-branch training.

The auth scalar interface:

```python
from auth_routing import AuthRoutedRDT

model = AuthRoutedRDT(forked_rdt)

# Public interaction (default)
action = model(x, freq, t, lang_c, img_c, auth=0)

# Authenticated operator
action = model(x, freq, t, lang_c, img_c, auth=1)
```

---

## Other Supporting Experiments

`test_shared_self_attention_interleaving.py` tests whether the shared self-attention blocks (upstream of the fork) behave differently depending on which route is selected. Result: 0.0000% relative difference across all 12 shared blocks, confirmed over 30 batches. This holds by construction for any architecture where the fork point is strictly downstream of the shared encoder. The code is included for completeness.

---

## Citation

```bibtex
@article{ahmed2026crossattnfork,
  title   = {Architectural Access Control via Cross-Attention Forking in Diffusion Transformers for Humanoid Robotics},
  author  = {Ahmed, Hasan},
  year    = {2026},
  url     = {https://zenodo.org/19834989}
}
```

---

## Related Work

- [RDT-1B](https://github.com/thu-ml/RoboticsDiffusionTransformer) — Liu et al., ICLR 2025
- [ChatVLA](https://arxiv.org/abs/2502.00746) — Zhou et al.
- [AEGIS](https://arxiv.org/abs/2504.02661) — Sinha et al.
