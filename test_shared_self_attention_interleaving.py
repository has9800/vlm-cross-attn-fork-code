#!/usr/bin/env python3
"""
test_shared_self_attention_interleaving.py

Tests whether the shared self-attention layers (before the fork point)
behave consistently or get disrupted when we alternate between
privileged and whitelisted routes.

This addresses the core worry: "Does the alternating vision/language
conditioning tokens in the shared self-attn cause problems?"
"""

import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np

# Import your existing classes (adjust path if needed)
from run_crossattn_fork_170m_2 import CrossAttnForkedRDT
# If using the AuthRouted wrapper:
from auth_routing import AuthRoutedRDT

# Optional: import RDTRunner if needed for adapt_conditions
from models.rdt_runner import RDTRunner   # adjust path


class SharedSelfAttnHook:
    """Capture self-attention patterns from shared blocks."""
    def __init__(self, model):
        self.model = model  # CrossAttnForkedRDT
        self.hooks = []
        self.attn_weights = defaultdict(list)  # block_idx -> list of tensors

    def attach(self):
        for i, block in enumerate(self.model.shared_blocks):
            # Register hook on the self-attention module
            handle = block.attn.register_forward_hook(
                self._make_hook(i)
            )
            self.hooks.append(handle)

    def _make_hook(self, block_idx):
        def hook(module, inputs, output):
            # output from self-attn is usually (B, N, C) after residual, but we want raw attn weights
            # For simplicity, we cache the output of attn (post-softmax weights if accessible)
            # Many DiT implementations store attn weights in module. We'll approximate with output stats
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            self.attn_weights[block_idx].append(out.detach().cpu())
        return hook

    def clear(self):
        for k in self.attn_weights:
            self.attn_weights[k].clear()

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_stats(self):
        stats = {}
        for blk, tensors in self.attn_weights.items():
            if not tensors:
                continue
            stacked = torch.cat(tensors, dim=0)  # (total_samples, seq_len, hidden)
            mean_norm = stacked.norm(dim=-1).mean().item()
            std_norm = stacked.norm(dim=-1).std().item()
            stats[blk] = {
                "mean_output_norm": mean_norm,
                "std_output_norm": std_norm,
                "num_samples": len(tensors)
            }
        return stats


def run_interleaving_test(checkpoint_path=None, num_batches=30, batch_size=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=== Shared Self-Attention Interleaving Test ===")
    print(f"Device: {device}, dtype: {dtype}\n")

    # Load base runner
    sys.path.insert(0, "/workspace/RoboticsDiffusionTransformer")  # adjust if needed
    from models.rdt_runner import RDTRunner
    runner = RDTRunner.from_pretrained("robotics-diffusion-transformer/rdt-170m", dtype=dtype)
    runner = runner.to(device)

    # Build forked model
    forked = CrossAttnForkedRDT(runner.model, num_forked_blocks=2, fork_final=True).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        forked.load_state_dict(state, strict=False)
        print("Checkpoint loaded.")
    else:
        print("Running on freshly forked (untrained) model.")

    # Wrap with auth routing for clean interface
    model = AuthRoutedRDT(forked).to(device)
    model.eval()

    # Attach hooks to shared self-attn
    hooker = SharedSelfAttnHook(forked)
    hooker.attach()

    stats_priv = []
    stats_wl = []

    for i in range(num_batches):
        # Build random but identical inputs for both routes
        B = batch_size
        lang = torch.randn(B, 32, 4096, device=device, dtype=dtype)
        img = torch.randn(B, 4374, 1152, device=device, dtype=dtype)
        state = torch.randn(B, 1, 128, device=device, dtype=dtype)
        action = torch.randn(B, 64, 128, device=device, dtype=dtype)
        mask = torch.ones(B, 1, 128, device=device, dtype=dtype)
        freq = torch.full((B,), 25.0, device=device, dtype=dtype)
        lm = torch.ones(B, 32, dtype=torch.bool, device=device)

        with torch.no_grad():
            # Prepare conditions the same way as training
            sa = torch.cat([state, action], dim=1)
            me = mask.expand(-1, sa.shape[1], -1)
            sa = torch.cat([sa, me], dim=2)
            lang_c, img_c, sa_c = runner.adapt_conditions(lang, img, sa)

            # --- Whitelisted route ---
            hooker.clear()
            out_wl = model(sa_c, freq, torch.zeros(B, device=device).long(),
                           lang_c, img_c, auth=0, lang_mask=lm)

            wl_stats = hooker.get_stats()
            stats_wl.append(wl_stats)

            # --- Privileged route ---
            hooker.clear()
            out_priv = model(sa_c, freq, torch.zeros(B, device=device).long(),
                             lang_c, img_c, auth=1, lang_mask=lm)

            priv_stats = hooker.get_stats()
            stats_priv.append(priv_stats)

        if i % 5 == 0:
            print(f"  Batch {i+1}/{num_batches} completed")

    hooker.detach()

    # Analyze consistency
    print("\n=== RESULTS: Shared Self-Attention Consistency ===\n")

    for blk in range(len(forked.shared_blocks)):
        wl_norms = [s.get(blk, {}).get("mean_output_norm", 0) for s in stats_wl if blk in s]
        priv_norms = [s.get(blk, {}).get("mean_output_norm", 0) for s in stats_priv if blk in s]

        if not wl_norms or not priv_norms:
            continue

        mean_wl = np.mean(wl_norms)
        mean_priv = np.mean(priv_norms)
        diff = abs(mean_wl - mean_priv)
        rel_diff = diff / (mean_wl + 1e-8)

        print(f"Shared Block {blk:2d}:")
        print(f"   Whitelisted mean norm : {mean_wl:.4f}")
        print(f"   Privileged mean norm  : {mean_priv:.4f}")
        print(f"   Absolute diff         : {diff:.4f}")
        print(f"   Relative diff         : {rel_diff:.4%}")
        
        if rel_diff > 0.05:
            print("   ⚠️  NOTICEABLE DIFFERENCE — possible interleaving sensitivity")
        elif rel_diff > 0.01:
            print("   ⚠️  Minor difference")
        else:
            print("   ✅ Very consistent")

    print("\nTest finished. If differences are small (<2-3%), the interleaving is likely not a major issue.")
    print("If differences are large, we should add a modality embedding or token type ID.")

    return {"wl_stats": stats_wl, "priv_stats": stats_priv}


if __name__ == "__main__":
    import sys, os
    checkpoint = "model_fork2_g1.pt" if os.path.exists("model_fork2_g1.pt") else None
    run_interleaving_test(checkpoint_path=checkpoint, num_batches=30)
