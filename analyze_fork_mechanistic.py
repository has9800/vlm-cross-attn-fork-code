#!/usr/bin/env python3
"""
analyze_fork_mechanistic.py — Mechanistic Analysis of Cross-Attention Fork
===========================================================================
Loads a trained forked RDT-170M checkpoint and runs:

1. Cross-Attention Pattern Divergence
   - Extract attention weight matrices from priv vs wl cross-attn
   - Visualize which conditioning tokens each head attends to
   - Measure KL divergence and JS divergence of attention distributions

2. Action Space Geometry
   - Collect action predictions from both heads across many inputs
   - PCA/SVD the action distributions
   - Show they occupy different principal subspaces
   - Measure subspace angle between privileged and whitelisted manifolds

3. FFN Input/Output Analysis
   - Cache FFN inputs from both heads (= cross-attn outputs)
   - Measure whether shared FFN compresses or preserves divergence
   - Check neuron activation overlap between heads

Usage:
    python analyze_fork_mechanistic.py --checkpoint model_fork2.pt --rdt_repo /workspace/RoboticsDiffusionTransformer
"""

import argparse
import copy
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────────────────────────
# Re-import the fork classes (self-contained)
# ─────────────────────────────────────────────────────────────
class ForkedFinalLayer(nn.Module):
    def __init__(self, original_final_layer):
        super().__init__()
        self.privileged = copy.deepcopy(original_final_layer)
        self.whitelisted = copy.deepcopy(original_final_layer)
        for p in self.privileged.parameters():
            p.requires_grad = False
        for p in self.whitelisted.parameters():
            p.requires_grad = True
    def forward(self, x, head="whitelisted"):
        if head == "privileged":
            return self.privileged(x)
        return self.whitelisted(x)


class ForkedRDTBlock(nn.Module):
    def __init__(self, original_block, freeze_privileged=True, freeze_shared=True):
        super().__init__()
        self.norm1 = original_block.norm1
        self.attn = original_block.attn
        self.norm3 = original_block.norm3
        self.ffn = original_block.ffn
        if freeze_shared:
            for p in self.norm1.parameters():
                p.requires_grad = False
            for p in self.attn.parameters():
                p.requires_grad = False
            for p in self.norm3.parameters():
                p.requires_grad = False
            for p in self.ffn.parameters():
                p.requires_grad = False
        self.priv_norm2 = original_block.norm2
        self.priv_cross_attn = original_block.cross_attn
        if freeze_privileged:
            for p in self.priv_norm2.parameters():
                p.requires_grad = False
            for p in self.priv_cross_attn.parameters():
                p.requires_grad = False
        self.wl_norm2 = copy.deepcopy(original_block.norm2)
        self.wl_cross_attn = copy.deepcopy(original_block.cross_attn)
        for p in self.wl_norm2.parameters():
            p.requires_grad = True
        for p in self.wl_cross_attn.parameters():
            p.requires_grad = True

    def forward(self, x, c, mask=None, head="whitelisted"):
        origin_x = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + origin_x
        origin_x = x
        if head == "privileged":
            x = self.priv_norm2(x)
            x = self.priv_cross_attn(x, c, mask)
        else:
            x = self.wl_norm2(x)
            x = self.wl_cross_attn(x, c, mask)
        x = x + origin_x
        origin_x = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = x + origin_x
        return x


class CrossAttnForkedRDT(nn.Module):
    def __init__(self, rdt_model, num_forked_blocks=2, fork_final=True):
        super().__init__()
        self.horizon = rdt_model.horizon
        self.hidden_size = rdt_model.hidden_size
        self.num_forked_blocks = num_forked_blocks
        self.fork_final = fork_final
        total_depth = len(rdt_model.blocks)
        self.split_idx = total_depth - num_forked_blocks
        self.shared_blocks = nn.ModuleList([
            rdt_model.blocks[i] for i in range(self.split_idx)])
        for b in self.shared_blocks:
            for p in b.parameters():
                p.requires_grad = False
        self.forked_blocks = nn.ModuleList([
            ForkedRDTBlock(rdt_model.blocks[i])
            for i in range(self.split_idx, total_depth)])
        if fork_final:
            self.forked_final = ForkedFinalLayer(rdt_model.final_layer)
        else:
            self.final_layer = rdt_model.final_layer
        self.t_embedder = rdt_model.t_embedder
        self.freq_embedder = rdt_model.freq_embedder
        self.x_pos_embed = rdt_model.x_pos_embed
        self.lang_cond_pos_embed = rdt_model.lang_cond_pos_embed
        self.img_cond_pos_embed = rdt_model.img_cond_pos_embed
        for p in self.t_embedder.parameters():
            p.requires_grad = False
        for p in self.freq_embedder.parameters():
            p.requires_grad = False
        self.x_pos_embed.requires_grad = False
        self.lang_cond_pos_embed.requires_grad = False
        self.img_cond_pos_embed.requires_grad = False

    def forward(self, x, freq, t, lang_c, img_c,
                head="whitelisted", lang_mask=None, img_mask=None):
        t_emb = self.t_embedder(t).unsqueeze(1)
        freq_emb = self.freq_embedder(freq).unsqueeze(1)
        if t_emb.shape[0] == 1:
            t_emb = t_emb.expand(x.shape[0], -1, -1)
        x = torch.cat([t_emb, freq_emb, x], dim=1)
        x = x + self.x_pos_embed
        lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
        img_c = img_c + self.img_cond_pos_embed
        conds = [lang_c, img_c]
        masks = [lang_mask, img_mask]
        for i, block in enumerate(self.shared_blocks):
            x = block(x, conds[i % 2], masks[i % 2])
        for local_i, block in enumerate(self.forked_blocks):
            global_i = self.split_idx + local_i
            x = block(x, conds[global_i % 2], masks[global_i % 2], head=head)
        if self.fork_final:
            x = self.forked_final(x, head=head)
        else:
            x = self.final_layer(x)
        return x[:, -self.horizon:]


# ─────────────────────────────────────────────────────────────
# Hooked forward for extracting cross-attn internals
# ─────────────────────────────────────────────────────────────
class ForkedBlockWithHooks(nn.Module):
    """
    Wraps a ForkedRDTBlock to capture:
    - Cross-attention weights (Q @ K^T after softmax)
    - FFN input (= post cross-attn + residual)
    - FFN output
    """
    def __init__(self, forked_block):
        super().__init__()
        self.block = forked_block
        self.hooks = {}

    def forward_with_hooks(self, x, c, mask=None, head="whitelisted"):
        hooks = {}

        # Shared self-attention
        origin_x = x
        x = self.block.norm1(x)
        x = self.block.attn(x)
        x = x + origin_x

        # Forked cross-attention — manual computation to extract attention weights
        origin_x = x
        if head == "privileged":
            norm = self.block.priv_norm2
            cross_attn = self.block.priv_cross_attn
        else:
            norm = self.block.wl_norm2
            cross_attn = self.block.wl_cross_attn

        x_normed = norm(x)

        # Manual cross-attention to capture weights
        B, N, C = x_normed.shape
        _, L, _ = c.shape
        num_heads = cross_attn.num_heads
        head_dim = cross_attn.head_dim

        q = cross_attn.q(x_normed).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        kv = cross_attn.kv(c).reshape(B, L, 2, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q = cross_attn.q_norm(q)
        k = cross_attn.k_norm(k)

        # Compute attention weights explicitly
        scale = head_dim ** -0.5
        attn_weights = (q * scale) @ k.transpose(-2, -1)  # (B, H, N, L)
        if mask is not None:
            attn_mask = mask.reshape(B, 1, 1, L).expand(-1, -1, N, -1)
            attn_weights = attn_weights.masked_fill_(attn_mask.logical_not(), float('-inf'))
        attn_weights = attn_weights.softmax(dim=-1)

        hooks["attn_weights"] = attn_weights.detach()  # (B, H, N, L)

        # Complete cross-attention
        attn_out = attn_weights @ v
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, C)
        attn_out = cross_attn.proj(attn_out)

        hooks["cross_attn_output"] = attn_out.detach()

        x = attn_out + origin_x

        # FFN
        hooks["ffn_input"] = x.detach()
        origin_x = x
        x = self.block.norm3(x)
        x = self.block.ffn(x)
        hooks["ffn_output_prenorm"] = x.detach()
        x = x + origin_x
        hooks["ffn_output"] = x.detach()

        return x, hooks


# ─────────────────────────────────────────────────────────────
# Analysis 1: Cross-Attention Pattern Divergence
# ─────────────────────────────────────────────────────────────
def analyze_crossattn_patterns(forked_rdt, runner, device, dtype, num_batches=50):
    """
    Extract attention weight matrices from both heads,
    measure divergence in what they attend to.
    """
    print("\n" + "=" * 60)
    print("  ANALYSIS 1: Cross-Attention Pattern Divergence")
    print("=" * 60)

    # Wrap forked blocks with hooks
    hooked_blocks = [ForkedBlockWithHooks(b) for b in forked_rdt.forked_blocks]

    all_attn_priv = defaultdict(list)  # block_idx -> list of attn weight tensors
    all_attn_wl = defaultdict(list)

    for batch_idx in range(num_batches):
        # Generate random input
        B = 4
        lang_tokens = torch.randn(B, 32, 4096, device=device, dtype=dtype)
        img_tokens = torch.randn(B, 4374, 1152, device=device, dtype=dtype)
        state_tokens = torch.randn(B, 1, 128, device=device, dtype=dtype)
        action_gt = torch.randn(B, 64, 128, device=device, dtype=dtype)
        action_mask = torch.ones(B, 1, 128, device=device, dtype=dtype)
        ctrl_freqs = torch.ones(B, device=device) * 25
        lang_mask = torch.ones(B, 32, dtype=torch.bool, device=device)

        with torch.no_grad():
            state_action = torch.cat([state_tokens, action_gt], dim=1)
            mask_exp = action_mask.expand(-1, state_action.shape[1], -1)
            state_action = torch.cat([state_action, mask_exp], dim=2)
            lang_cond, img_cond, sa_cond = runner.adapt_conditions(
                lang_tokens, img_tokens, state_action)

            # Forward through shared blocks
            t_emb = forked_rdt.t_embedder(torch.zeros(B, device=device)).unsqueeze(1)
            freq_emb = forked_rdt.freq_embedder(ctrl_freqs).unsqueeze(1)
            t_emb = t_emb.expand(B, -1, -1)
            x = torch.cat([t_emb, freq_emb, sa_cond], dim=1)
            x = x + forked_rdt.x_pos_embed
            lc = lang_cond + forked_rdt.lang_cond_pos_embed[:, :lang_cond.shape[1]]
            ic = img_cond + forked_rdt.img_cond_pos_embed

            conds = [lc, ic]
            masks = [lang_mask, None]

            for i, block in enumerate(forked_rdt.shared_blocks):
                x = block(x, conds[i % 2], masks[i % 2])

            # Through forked blocks with hooks
            x_priv = x.clone()
            x_wl = x.clone()

            for local_i, hooked in enumerate(hooked_blocks):
                global_i = forked_rdt.split_idx + local_i
                c = conds[global_i % 2]
                m = masks[global_i % 2]

                x_priv, hooks_p = hooked.forward_with_hooks(x_priv, c, m, head="privileged")
                x_wl, hooks_w = hooked.forward_with_hooks(x_wl, c, m, head="whitelisted")

                # Store attention weights averaged over batch and heads
                # Shape: (B, H, N, L) -> mean over B,H -> (N, L)
                all_attn_priv[local_i].append(hooks_p["attn_weights"].float().mean(dim=(0, 1)).cpu())
                all_attn_wl[local_i].append(hooks_w["attn_weights"].float().mean(dim=(0, 1)).cpu())

    # Aggregate and analyze
    results = {}
    for block_i in range(len(hooked_blocks)):
        global_i = forked_rdt.split_idx + block_i

        # Average attention patterns across batches: (N, L)
        avg_priv = torch.stack(all_attn_priv[block_i]).mean(dim=0)
        avg_wl = torch.stack(all_attn_wl[block_i]).mean(dim=0)

        # Marginal over query positions: which KV positions get most attention
        # (L,) - average attention received by each conditioning token
        marginal_priv = avg_priv.mean(dim=0)
        marginal_wl = avg_wl.mean(dim=0)

        # KL divergence of marginal attention distributions
        # Add epsilon for numerical stability
        eps = 1e-8
        mp = marginal_priv + eps
        mw = marginal_wl + eps
        mp = mp / mp.sum()
        mw = mw / mw.sum()

        kl_div = (mp * (mp / mw).log()).sum().item()

        # JS divergence (symmetric)
        m_avg = 0.5 * (mp + mw)
        js_div = 0.5 * (mp * (mp / m_avg).log()).sum().item() + \
                 0.5 * (mw * (mw / m_avg).log()).sum().item()

        # Cosine similarity of marginal distributions
        cos_sim = F.cosine_similarity(marginal_priv.unsqueeze(0),
                                       marginal_wl.unsqueeze(0)).item()

        # Top-K attended positions
        top_k = 10
        _, priv_top = marginal_priv.topk(top_k)
        _, wl_top = marginal_wl.topk(top_k)
        overlap = len(set(priv_top.tolist()) & set(wl_top.tolist()))

        results[f"block_{global_i}"] = {
            "kl_divergence": kl_div,
            "js_divergence": js_div,
            "marginal_cosine": cos_sim,
            "top10_overlap": overlap,
            "cond_length": marginal_priv.shape[0],
        }

        print(f"\n  Block {global_i} (cond={'lang' if global_i % 2 == 0 else 'img'}, L={marginal_priv.shape[0]}):")
        print(f"    KL divergence:        {kl_div:.6f}")
        print(f"    JS divergence:        {js_div:.6f}")
        print(f"    Marginal cosine sim:  {cos_sim:.6f}")
        print(f"    Top-10 position overlap: {overlap}/10")
        print(f"    Priv top-5 positions: {priv_top[:5].tolist()}")
        print(f"    WL top-5 positions:   {wl_top[:5].tolist()}")

    return results


# ─────────────────────────────────────────────────────────────
# Analysis 2: Action Space Geometry
# ─────────────────────────────────────────────────────────────
def analyze_action_geometry(forked_rdt, runner, device, dtype, num_batches=100):
    """
    Collect action predictions from both heads, PCA them,
    measure subspace angles.
    """
    print("\n" + "=" * 60)
    print("  ANALYSIS 2: Action Space Geometry")
    print("=" * 60)

    all_priv_actions = []
    all_wl_actions = []

    for batch_idx in range(num_batches):
        B = 4
        lang_tokens = torch.randn(B, 32, 4096, device=device, dtype=dtype)
        img_tokens = torch.randn(B, 4374, 1152, device=device, dtype=dtype)
        state_tokens = torch.randn(B, 1, 128, device=device, dtype=dtype)
        action_gt = torch.randn(B, 64, 128, device=device, dtype=dtype)
        action_mask = torch.ones(B, 1, 128, device=device, dtype=dtype)
        ctrl_freqs = torch.ones(B, device=device) * 25
        lang_mask = torch.ones(B, 32, dtype=torch.bool, device=device)

        with torch.no_grad():
            state_action = torch.cat([state_tokens, action_gt], dim=1)
            mask_exp = action_mask.expand(-1, state_action.shape[1], -1)
            state_action = torch.cat([state_action, mask_exp], dim=2)
            lang_cond, img_cond, sa_cond = runner.adapt_conditions(
                lang_tokens, img_tokens, state_action)

            out_priv = forked_rdt(sa_cond, ctrl_freqs,
                                  torch.zeros(B, device=device).long(),
                                  lang_cond, img_cond, head="privileged",
                                  lang_mask=lang_mask)
            out_wl = forked_rdt(sa_cond, ctrl_freqs,
                                torch.zeros(B, device=device).long(),
                                lang_cond, img_cond, head="whitelisted",
                                lang_mask=lang_mask)

        # Flatten: (B, 64, 128) -> (B, 8192)
        all_priv_actions.append(out_priv.reshape(B, -1).float().cpu())
        all_wl_actions.append(out_wl.reshape(B, -1).float().cpu())

    # Stack: (N, 8192)
    priv_actions = torch.cat(all_priv_actions, dim=0).numpy()
    wl_actions = torch.cat(all_wl_actions, dim=0).numpy()

    print(f"\n  Collected {priv_actions.shape[0]} samples per head")
    print(f"  Action dim: {priv_actions.shape[1]} (64 steps × 128 action dims)")

    # PCA on each
    from numpy.linalg import svd

    def compute_pca(X, n_components=10):
        X_centered = X - X.mean(axis=0)
        U, S, Vt = svd(X_centered, full_matrices=False)
        explained_var = (S ** 2) / (S ** 2).sum()
        return Vt[:n_components], S[:n_components], explained_var[:n_components]

    n_comp = 10
    priv_Vt, priv_S, priv_var = compute_pca(priv_actions, n_comp)
    wl_Vt, wl_S, wl_var = compute_pca(wl_actions, n_comp)

    print(f"\n  Privileged top-{n_comp} explained variance: {priv_var.sum():.4f}")
    print(f"  Whitelisted top-{n_comp} explained variance: {wl_var.sum():.4f}")
    print(f"  Priv var per PC: {[f'{v:.3f}' for v in priv_var[:5]]}")
    print(f"  WL var per PC:   {[f'{v:.3f}' for v in wl_var[:5]]}")

    # Subspace angle: principal angles between the two subspaces
    # cos(theta_i) = singular values of Vt_priv @ Vt_wl^T
    cross = priv_Vt @ wl_Vt.T  # (n_comp, n_comp)
    _, cross_S, _ = svd(cross)
    principal_angles_deg = np.degrees(np.arccos(np.clip(cross_S, -1, 1)))

    print(f"\n  Principal angles between subspaces (degrees):")
    for i, angle in enumerate(principal_angles_deg[:5]):
        print(f"    PC{i}: {angle:.2f}°")

    mean_angle = principal_angles_deg.mean()
    print(f"  Mean principal angle: {mean_angle:.2f}°")
    print(f"  (90° = orthogonal subspaces, 0° = identical)")

    # Action distribution statistics
    priv_mean_norm = np.linalg.norm(priv_actions, axis=1).mean()
    wl_mean_norm = np.linalg.norm(wl_actions, axis=1).mean()
    cross_cos = np.array([
        np.dot(priv_actions[i], wl_actions[i]) /
        (np.linalg.norm(priv_actions[i]) * np.linalg.norm(wl_actions[i]) + 1e-8)
        for i in range(min(len(priv_actions), len(wl_actions)))
    ])

    print(f"\n  Priv action mean norm:  {priv_mean_norm:.4f}")
    print(f"  WL action mean norm:    {wl_mean_norm:.4f}")
    print(f"  Pairwise cosine sim:    {cross_cos.mean():.4f} ± {cross_cos.std():.4f}")

    results = {
        "priv_explained_var": priv_var.tolist(),
        "wl_explained_var": wl_var.tolist(),
        "principal_angles_deg": principal_angles_deg.tolist(),
        "mean_principal_angle": float(mean_angle),
        "pairwise_cosine_mean": float(cross_cos.mean()),
        "pairwise_cosine_std": float(cross_cos.std()),
    }

    return results


# ─────────────────────────────────────────────────────────────
# Analysis 3: FFN Input/Output Divergence
# ─────────────────────────────────────────────────────────────
def analyze_ffn_divergence(forked_rdt, runner, device, dtype, num_batches=50):
    """
    Check whether the shared FFN compresses or preserves
    the cross-attention divergence. Also measure neuron
    activation overlap.
    """
    print("\n" + "=" * 60)
    print("  ANALYSIS 3: FFN Input/Output Divergence")
    print("=" * 60)

    hooked_blocks = [ForkedBlockWithHooks(b) for b in forked_rdt.forked_blocks]

    ffn_metrics = defaultdict(lambda: {
        "input_divergence": [],
        "output_divergence": [],
        "input_cosine": [],
        "output_cosine": [],
        "neuron_overlap": [],
    })

    for batch_idx in range(num_batches):
        B = 4
        lang_tokens = torch.randn(B, 32, 4096, device=device, dtype=dtype)
        img_tokens = torch.randn(B, 4374, 1152, device=device, dtype=dtype)
        state_tokens = torch.randn(B, 1, 128, device=device, dtype=dtype)
        action_gt = torch.randn(B, 64, 128, device=device, dtype=dtype)
        action_mask = torch.ones(B, 1, 128, device=device, dtype=dtype)
        ctrl_freqs = torch.ones(B, device=device) * 25
        lang_mask = torch.ones(B, 32, dtype=torch.bool, device=device)

        with torch.no_grad():
            state_action = torch.cat([state_tokens, action_gt], dim=1)
            mask_exp = action_mask.expand(-1, state_action.shape[1], -1)
            state_action = torch.cat([state_action, mask_exp], dim=2)
            lang_cond, img_cond, sa_cond = runner.adapt_conditions(
                lang_tokens, img_tokens, state_action)

            t_emb = forked_rdt.t_embedder(torch.zeros(B, device=device)).unsqueeze(1)
            freq_emb = forked_rdt.freq_embedder(ctrl_freqs).unsqueeze(1)
            t_emb = t_emb.expand(B, -1, -1)
            x = torch.cat([t_emb, freq_emb, sa_cond], dim=1)
            x = x + forked_rdt.x_pos_embed
            lc = lang_cond + forked_rdt.lang_cond_pos_embed[:, :lang_cond.shape[1]]
            ic = img_cond + forked_rdt.img_cond_pos_embed

            conds = [lc, ic]
            masks = [lang_mask, None]

            for i, block in enumerate(forked_rdt.shared_blocks):
                x = block(x, conds[i % 2], masks[i % 2])

            x_priv = x.clone()
            x_wl = x.clone()

            for local_i, hooked in enumerate(hooked_blocks):
                global_i = forked_rdt.split_idx + local_i
                c = conds[global_i % 2]
                m = masks[global_i % 2]

                x_priv, hooks_p = hooked.forward_with_hooks(x_priv, c, m, head="privileged")
                x_wl, hooks_w = hooked.forward_with_hooks(x_wl, c, m, head="whitelisted")

                # FFN input divergence (= post cross-attn + residual)
                ffn_in_p = hooks_p["ffn_input"].float()
                ffn_in_w = hooks_w["ffn_input"].float()
                ffn_out_p = hooks_p["ffn_output"].float()
                ffn_out_w = hooks_w["ffn_output"].float()

                in_div = (ffn_in_p - ffn_in_w).abs().mean().item()
                out_div = (ffn_out_p - ffn_out_w).abs().mean().item()
                in_cos = F.cosine_similarity(ffn_in_p.reshape(1, -1),
                                              ffn_in_w.reshape(1, -1)).item()
                out_cos = F.cosine_similarity(ffn_out_p.reshape(1, -1),
                                               ffn_out_w.reshape(1, -1)).item()

                ffn_metrics[local_i]["input_divergence"].append(in_div)
                ffn_metrics[local_i]["output_divergence"].append(out_div)
                ffn_metrics[local_i]["input_cosine"].append(in_cos)
                ffn_metrics[local_i]["output_cosine"].append(out_cos)

                # Neuron activation overlap
                # Which neurons (hidden dim) are active (> 0 after GELU) for each head
                # We check the pre-residual FFN output
                ffn_pre_p = hooks_p["ffn_output_prenorm"].float()
                ffn_pre_w = hooks_w["ffn_output_prenorm"].float()

                # Active = magnitude > threshold (mean activation level)
                thresh_p = ffn_pre_p.abs().mean()
                thresh_w = ffn_pre_w.abs().mean()
                active_p = (ffn_pre_p.abs() > thresh_p).float().mean(dim=(0, 1))  # (hidden,)
                active_w = (ffn_pre_w.abs() > thresh_w).float().mean(dim=(0, 1))  # (hidden,)

                # Overlap: fraction of neurons active in both
                both_active = ((active_p > 0.5) & (active_w > 0.5)).float().mean().item()
                ffn_metrics[local_i]["neuron_overlap"].append(both_active)

    # Report
    results = {}
    for block_i in range(len(hooked_blocks)):
        global_i = forked_rdt.split_idx + block_i
        m = ffn_metrics[block_i]

        avg_in_div = np.mean(m["input_divergence"])
        avg_out_div = np.mean(m["output_divergence"])
        avg_in_cos = np.mean(m["input_cosine"])
        avg_out_cos = np.mean(m["output_cosine"])
        avg_overlap = np.mean(m["neuron_overlap"])

        # Does FFN compress or amplify?
        ratio = avg_out_div / (avg_in_div + 1e-8)

        results[f"block_{global_i}"] = {
            "ffn_input_divergence": avg_in_div,
            "ffn_output_divergence": avg_out_div,
            "ffn_input_cosine": avg_in_cos,
            "ffn_output_cosine": avg_out_cos,
            "divergence_ratio": ratio,
            "neuron_overlap": avg_overlap,
        }

        print(f"\n  Block {global_i}:")
        print(f"    FFN input divergence:    {avg_in_div:.6f}")
        print(f"    FFN output divergence:   {avg_out_div:.6f}")
        print(f"    Divergence ratio:        {ratio:.4f}x {'(AMPLIFIES)' if ratio > 1 else '(COMPRESSES)' if ratio < 0.95 else '(PRESERVES)'}")
        print(f"    FFN input cosine:        {avg_in_cos:.6f}")
        print(f"    FFN output cosine:       {avg_out_cos:.6f}")
        print(f"    Neuron overlap:          {avg_overlap:.4f} ({avg_overlap*100:.1f}%)")

        if avg_overlap > 0.8:
            print(f"    ⚠️  High neuron overlap — shared FFN circuits active for both heads")
        elif avg_overlap < 0.3:
            print(f"    ✅ Low neuron overlap — heads use distinct FFN circuits")
        else:
            print(f"    ℹ️  Moderate neuron overlap — partial circuit sharing")

    return results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"{'='*60}")
    print(f"  Mechanistic Analysis of Cross-Attention Fork")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Load RDT repo
    sys.path.insert(0, args.rdt_repo)
    os.chdir(args.rdt_repo)
    from models.rdt_runner import RDTRunner

    # Load base model
    print("\n[1/4] Loading RDT-170M base model...")
    runner = RDTRunner.from_pretrained("robotics-diffusion-transformer/rdt-170m", dtype=dtype)
    runner = runner.to(device=device, dtype=dtype)

    # Build forked model structure
    print(f"[2/4] Building fork structure ({args.num_forked_blocks} blocks)...")
    forked_rdt = CrossAttnForkedRDT(
        runner.model, num_forked_blocks=args.num_forked_blocks, fork_final=True
    ).to(device)

    # Load trained checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[3/4] Loading trained checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        forked_rdt.load_state_dict(state_dict)
        print("  Checkpoint loaded ✓")
    else:
        print(f"[3/4] No checkpoint found at {args.checkpoint}")
        print("  Running analysis on freshly forked model (pre-training)")
        print("  Results will show baseline divergence before training")

    forked_rdt.eval()
    runner.model = None  # detach
    for p in runner.parameters():
        p.requires_grad = False

    # Run analyses
    print(f"[4/4] Running analyses...")

    results = {}

    results["crossattn_patterns"] = analyze_crossattn_patterns(
        forked_rdt, runner, device, dtype, num_batches=args.num_batches)

    results["action_geometry"] = analyze_action_geometry(
        forked_rdt, runner, device, dtype, num_batches=args.num_batches)

    results["ffn_divergence"] = analyze_ffn_divergence(
        forked_rdt, runner, device, dtype, num_batches=args.num_batches)

    # Save results
    out_path = f"mechanistic_analysis_fork{args.num_forked_blocks}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  All results saved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    for block_key, data in results["ffn_divergence"].items():
        ratio = data["divergence_ratio"]
        overlap = data["neuron_overlap"]
        if ratio > 1 and overlap < 0.5:
            print(f"  {block_key}: FFN AMPLIFIES divergence, LOW neuron overlap")
            print(f"    → Shared FFN is safe — cross-attn separation is sufficient")
        elif ratio < 0.95 and overlap > 0.7:
            print(f"  {block_key}: FFN COMPRESSES divergence, HIGH neuron overlap")
            print(f"    → Consider forking FFN — shared circuits may leak")
        else:
            print(f"  {block_key}: FFN ratio={ratio:.3f}, overlap={overlap:.1%}")

    geom = results["action_geometry"]
    print(f"\n  Action subspace mean angle: {geom['mean_principal_angle']:.1f}°")
    if geom['mean_principal_angle'] > 60:
        print(f"  ✅ Heads occupy substantially different action subspaces")
    elif geom['mean_principal_angle'] > 30:
        print(f"  ℹ️  Moderate subspace separation")
    else:
        print(f"  ⚠️  Heads share similar action subspaces")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="model_fork2.pt")
    parser.add_argument("--num_forked_blocks", type=int, default=2)
    parser.add_argument("--num_batches", type=int, default=50,
                        help="Number of batches for analysis (more = smoother stats)")
    parser.add_argument("--rdt_repo", type=str,
                        default="/workspace/RoboticsDiffusionTransformer")
    args = parser.parse_args()
    main(args)
