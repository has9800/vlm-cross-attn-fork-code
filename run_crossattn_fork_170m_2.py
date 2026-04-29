#!/usr/bin/env python3
"""
run_crossattn_fork_170m.py — A100 Experiment
=============================================
Loads real RDT-170M (170M params, hidden=1024, depth=14, 16 heads),
applies cross-attention-only fork on last N blocks, trains the
whitelisted branch on synthetic gesture data, monitors corruption
of the frozen privileged branch, generates paper figures.

Usage (on Vast.ai A100):
    # 1. Clone RDT repo and install deps
    git clone https://github.com/thu-ml/RoboticsDiffusionTransformer.git
    cd RoboticsDiffusionTransformer
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
    pip install packaging==24.0 timm==0.9.12 diffusers==0.24.0 transformers==4.36.0
    pip install safetensors sentencepiece protobuf matplotlib
    pip install flash-attn --no-build-isolation 2>/dev/null || true

    # 2. Run experiment
    python run_crossattn_fork_170m.py --num_forked_blocks 6 --steps 5000

    # 3. Sweep fork depth for ablation
    for n in 2 4 6 8 10; do
        python run_crossattn_fork_170m.py --num_forked_blocks $n --steps 5000
    done

RDT-170M specs:
    hidden_size: 1024
    depth: 14
    num_heads: 16 (1024/64 head_dim, inferred from config)
    action_dim: 128 (unified action space)
    horizon: 64 (action chunk size)
    lang_token_dim: 4096 (T5-XXL)
    img_token_dim: 1152 (SigLIP)
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
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────────────────────────────────────────
# Self-contained ForkedFinalLayer + CrossAttnForkedRDT
# (inlined so this script has zero external dependencies
#  beyond the RDT repo itself)
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
    """Shares self-attn + FFN, forks cross-attn + its norm."""
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
    """Applies cross-attn fork to last N blocks of an RDT model."""
    def __init__(self, rdt_model, num_forked_blocks=6, fork_final=True):
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
            for p in self.final_layer.parameters():
                p.requires_grad = False
        
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
# Corruption Monitor
# ─────────────────────────────────────────────────────────────
class CorruptionMonitor:
    def __init__(self, model: CrossAttnForkedRDT):
        self.snapshots = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                self.snapshots[name] = param.data.detach().clone().cpu()
    
    def check(self, model: CrossAttnForkedRDT) -> dict:
        max_drift = 0.0
        corrupted = []
        for name, param in model.named_parameters():
            if name in self.snapshots:
                drift = (param.data.cpu() - self.snapshots[name]).abs().max().item()
                if drift > 0:
                    corrupted.append(name)
                max_drift = max(max_drift, drift)
        return {"max_drift": max_drift, "corrupted": corrupted}


# ─────────────────────────────────────────────────────────────
# Synthetic Gesture Dataset
# ─────────────────────────────────────────────────────────────
class SyntheticGestureDataset(Dataset):
    """
    Generates synthetic social gesture trajectories.
    RDT unified action space is 128-dim.
    We target right arm joints (64-73) for waving,
    base locomotion (118-127) for stepping aside.
    """
    def __init__(self, num_samples=50000, horizon=64, state_dim=128,
                 lang_dim=4096, img_dim=1152):
        self.num_samples = num_samples
        self.horizon = horizon
        self.state_dim = state_dim
        self.lang_dim = lang_dim
        self.img_dim = img_dim
        
        # SigLIP: 3 views * 196 patches = 588 tokens (img_history=2 → 1176)
        # But RDT pads to a fixed img_cond_len from config
        # For 170M: img_cond_len = num_cameras * img_history * num_patches
        # = 3 * 2 * 196 = 1176
        self.num_img_tokens = 4374
        
        t = torch.linspace(0, 2 * 3.14159, horizon)
        self.wave = torch.zeros(horizon, state_dim)
        self.wave[:, 64] = 0.3 * torch.sin(t * 3)
        self.wave[:, 65] = 0.2 * torch.cos(t * 3)
        self.wave[:, 66] = 0.1 * torch.sin(t * 6)
        
        self.step_aside = torch.zeros(horizon, state_dim)
        self.step_aside[:, 118] = torch.linspace(0, 0.5, horizon)
        self.step_aside[:, 119] = 0.1 * torch.sin(t * 2)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        gesture = idx % 2
        if gesture == 0:
            action_gt = self.wave + 0.02 * torch.randn_like(self.wave)
            mask = torch.zeros(1, self.state_dim)
            mask[0, 64:70] = 1.0
        else:
            action_gt = self.step_aside + 0.02 * torch.randn_like(self.step_aside)
            mask = torch.zeros(1, self.state_dim)
            mask[0, 118:122] = 1.0
        
        return {
            "lang_tokens": torch.randn(32, self.lang_dim),
            "lang_attn_mask": torch.ones(32, dtype=torch.bool),
            "img_tokens": torch.randn(self.num_img_tokens, self.img_dim),
            "state_tokens": torch.randn(1, self.state_dim),
            "action_gt": action_gt,
            "action_mask": mask,
            "ctrl_freqs": torch.tensor(25.0),
        }


# ─────────────────────────────────────────────────────────────
# Main Experiment
# ─────────────────────────────────────────────────────────────
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    print(f"{'='*60}")
    print(f"  Cross-Attention Fork on RDT-170M")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Forked blocks: {args.num_forked_blocks}/14")
    print(f"  Steps: {args.steps}")
    print(f"{'='*60}\n")
    
    # ── Load RDT-170M ──
    print("[1/5] Loading RDT-170M...")
    sys.path.insert(0, args.rdt_repo)
    os.chdir(args.rdt_repo)
    
    from models.rdt_runner import RDTRunner
    
    runner = RDTRunner.from_pretrained(
        "robotics-diffusion-transformer/rdt-170m",
        dtype=dtype
    )
    
    rdt_model = runner.model
    total_params = sum(p.numel() for p in runner.parameters())
    print(f"  Loaded: {total_params:,} params")
    print(f"  DiT: depth={len(rdt_model.blocks)}, hidden={rdt_model.hidden_size}")
    
    # ── Apply cross-attn fork ──
    print(f"\n[2/5] Applying cross-attention fork (last {args.num_forked_blocks} blocks)...")
    
    forked_rdt = CrossAttnForkedRDT(
        rdt_model, 
        num_forked_blocks=args.num_forked_blocks,
        fork_final=True
    ).to(device)
    
    # Replace the runner's model reference
    # We need to keep the runner for its adaptors and noise scheduler
    runner.model = None  # detach old reference
    
    # Freeze runner's adaptors
    for p in runner.lang_adaptor.parameters():
        p.requires_grad = False
    for p in runner.img_adaptor.parameters():
        p.requires_grad = False
    for p in runner.state_adaptor.parameters():
        p.requires_grad = False
    
    runner = runner.to(device=device, dtype=dtype)
    
    trainable = sum(p.numel() for p in forked_rdt.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in forked_rdt.parameters() if not p.requires_grad)
    frozen += sum(p.numel() for p in runner.parameters() if not p.requires_grad)
    print(f"  Trainable: {trainable:,} ({trainable/(trainable+frozen)*100:.1f}%)")
    print(f"  Frozen:    {frozen:,}")
    
    # ── Setup monitoring ──
    print(f"\n[3/5] Setting up corruption monitor...")
    monitor = CorruptionMonitor(forked_rdt)
    print(f"  Snapshotted {len(monitor.snapshots)} frozen param tensors")
    
    # ── Dataset + optimizer ──
    print(f"\n[4/5] Preparing data and optimizer...")
    dataset = SyntheticGestureDataset(num_samples=args.steps * args.batch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)
    
    trainable_params = [p for p in forked_rdt.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    
    # ── Fixed eval batch for consistent divergence measurement ──
    eval_batch = next(iter(dataloader))
    eval_batch = {k: v.to(device=device, dtype=dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in eval_batch.items()}
    
    # ── Training loop ──
    print(f"\n[5/5] Training whitelisted cross-attention...")
    print(f"  {'Step':>6s} | {'Loss':>8s} | {'OutDiff':>8s} | {'CosSim':>8s} | {'Drift':>10s} | {'Time':>6s}")
    print(f"  {'-'*60}")
    
    history = {
        "steps": [], "loss": [], "output_diff": [], "output_cosine": [],
        "max_drift": [],
        "block_q_div": {i: [] for i in range(args.num_forked_blocks)},
        "block_kv_div": {i: [] for i in range(args.num_forked_blocks)},
        "block_proj_div": {i: [] for i in range(args.num_forked_blocks)},
    }
    
    forked_rdt.train()
    t_start = time.time()
    step = 0
    
    for batch in dataloader:
        if step >= args.steps:
            break
        
        batch = {k: v.to(device=device, dtype=dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        
        # ── Forward: adapt conditions through runner, then forward through forked DiT ──
        with torch.no_grad():
            # State tokens need action_mask appended before adapting
            state_action = torch.cat([batch["state_tokens"], batch["action_gt"]], dim=1)
            action_mask_exp = batch["action_mask"].expand(-1, state_action.shape[1], -1)
            state_action = torch.cat([state_action, action_mask_exp], dim=2)
            
            lang_cond, img_cond, state_action_cond = runner.adapt_conditions(
                batch["lang_tokens"], batch["img_tokens"], state_action)
        
        # Diffusion: add noise to actions
        noise = torch.randn_like(batch["action_gt"])
        timesteps = torch.randint(0, runner.num_train_timesteps,
                                  (batch["action_gt"].shape[0],), device=device).long()
        noisy_action = runner.noise_scheduler.add_noise(batch["action_gt"], noise, timesteps)
        
        # Build noisy state-action trajectory and adapt
        noisy_sa = torch.cat([batch["state_tokens"], noisy_action], dim=1)
        noisy_mask = batch["action_mask"].expand(-1, noisy_sa.shape[1], -1)
        noisy_sa = torch.cat([noisy_sa, noisy_mask], dim=2)
        _, _, noisy_sa_cond = runner.adapt_conditions(
            batch["lang_tokens"], batch["img_tokens"], noisy_sa)
        
        # Detach conditioning (no grad through frozen adaptors)
        lang_cond = lang_cond.detach()
        img_cond = img_cond.detach()
        noisy_sa_cond = noisy_sa_cond  # this one needs grad for the forked blocks
        
        # Forward through forked DiT
        pred = forked_rdt(noisy_sa_cond, batch["ctrl_freqs"], timesteps,
                          lang_cond, img_cond, head="whitelisted")
        
        # Loss (prediction_type=sample → target is clean action)
        target = batch["action_gt"]
        loss = F.mse_loss(pred, target)
        
        optimizer.zero_grad()
        loss.backward()
        
        # First step: verify no gradient leakage
        if step == 0:
            leaked = []
            for name, p in forked_rdt.named_parameters():
                if not p.requires_grad and p.grad is not None:
                    leaked.append(name)
            if leaked:
                print(f"  !! GRADIENT LEAKAGE: {leaked[:5]}")
            else:
                print(f"  ✓ No gradient leakage verified")
        
        optimizer.step()
        
        # ── Log metrics ──
        if step % args.log_interval == 0:
            history["steps"].append(step)
            history["loss"].append(loss.item())
            
            # Output divergence on fixed eval batch
            with torch.no_grad():
                eval_sa = torch.cat([eval_batch["state_tokens"], eval_batch["action_gt"]], dim=1)
                eval_mask = eval_batch["action_mask"].expand(-1, eval_sa.shape[1], -1)
                eval_sa = torch.cat([eval_sa, eval_mask], dim=2)
                el, ei, es = runner.adapt_conditions(
                    eval_batch["lang_tokens"], eval_batch["img_tokens"], eval_sa)
                
                out_p = forked_rdt(es, eval_batch["ctrl_freqs"],
                                   torch.zeros(args.batch_size, device=device).long(),
                                   el, ei, head="privileged")
                out_w = forked_rdt(es, eval_batch["ctrl_freqs"],
                                   torch.zeros(args.batch_size, device=device).long(),
                                   el, ei, head="whitelisted")
            
            diff = (out_p - out_w).abs().mean().item()
            cos = F.cosine_similarity(out_p.reshape(1,-1).float(), 
                                       out_w.reshape(1,-1).float()).item()
            history["output_diff"].append(diff)
            history["output_cosine"].append(cos)
            
            # Corruption check
            m = monitor.check(forked_rdt)
            history["max_drift"].append(m["max_drift"])
            
            # Per-block weight divergence
            for i, block in enumerate(forked_rdt.forked_blocks):
                history["block_q_div"][i].append(
                    (block.priv_cross_attn.q.weight.data - block.wl_cross_attn.q.weight.data).abs().mean().item())
                history["block_kv_div"][i].append(
                    (block.priv_cross_attn.kv.weight.data - block.wl_cross_attn.kv.weight.data).abs().mean().item())
                history["block_proj_div"][i].append(
                    (block.priv_cross_attn.proj.weight.data - block.wl_cross_attn.proj.weight.data).abs().mean().item())
            
            elapsed = time.time() - t_start
            print(f"  {step:6d} | {loss.item():8.4f} | {diff:8.4f} | {cos:8.4f} | {m['max_drift']:10.2e} | {elapsed:5.0f}s")
        
        step += 1
    
    # ── Final report ──
    elapsed = time.time() - t_start
    final_m = monitor.check(forked_rdt)
    
    print(f"\n{'='*60}")
    print(f"  RESULTS (fork={args.num_forked_blocks} blocks, {step} steps, {elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  Final loss:         {history['loss'][-1]:.6f}")
    print(f"  Output divergence:  {history['output_diff'][-1]:.4f}")
    print(f"  Output cosine:      {history['output_cosine'][-1]:.4f}")
    print(f"  Max frozen drift:   {final_m['max_drift']:.2e}")
    
    if final_m['max_drift'] == 0.0:
        print(f"\n  ✅ ZERO DRIFT — privileged policy completely untouched")
    else:
        print(f"\n  ❌ CORRUPTION DETECTED — {len(final_m['corrupted'])} params drifted")
    
    if len(final_m['corrupted']) > 0:
        for c in final_m['corrupted'][:5]:
            print(f"     {c}")
    
    # Per-block final divergence
    print(f"\n  Per-block cross-attn weight divergence:")
    for i, block in enumerate(forked_rdt.forked_blocks):
        gi = forked_rdt.split_idx + i
        q = (block.priv_cross_attn.q.weight.data - block.wl_cross_attn.q.weight.data).abs().mean().item()
        kv = (block.priv_cross_attn.kv.weight.data - block.wl_cross_attn.kv.weight.data).abs().mean().item()
        proj = (block.priv_cross_attn.proj.weight.data - block.wl_cross_attn.proj.weight.data).abs().mean().item()
        print(f"    Block {gi:2d}: Q={q:.6f}  KV={kv:.6f}  proj={proj:.6f}")
    
    # ── Save metrics ──
    tag = f"fork{args.num_forked_blocks}"
    metrics_path = f"results_{tag}.json"
    save_hist = {k: v for k, v in history.items() if not k.startswith("block_")}
    for key in ["block_q_div", "block_kv_div", "block_proj_div"]:
        save_hist[key] = {str(k): v for k, v in history[key].items()}
    save_hist["config"] = {
        "num_forked_blocks": args.num_forked_blocks,
        "steps": step,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "elapsed_s": elapsed,
    }
    with open(metrics_path, "w") as f:
        json.dump(save_hist, f, indent=2)
    torch.save(forked_rdt.state_dict(), f"model_fork{args.num_forked_blocks}.pt")
    print(f"  Model saved to model_fork{args.num_forked_blocks}.pt")
    print(f"\n  Metrics saved to {metrics_path}")
    
    # ── Generate figures ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        steps = history["steps"]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Cross-Attention Fork on RDT-170M (last {args.num_forked_blocks}/14 blocks)",
                     fontsize=14, fontweight='bold')
        
        # Fig 1: Per-block weight divergence
        ax = axes[0, 0]
        for i in range(args.num_forked_blocks):
            gi = forked_rdt.split_idx + i
            ax.plot(steps, history["block_q_div"][i], '-', label=f'Blk {gi} Q', alpha=0.8)
            ax.plot(steps, history["block_kv_div"][i], '--', label=f'Blk {gi} KV', alpha=0.6)
            ax.plot(steps, history["block_proj_div"][i], ':', label=f'Blk {gi} Proj', alpha=0.6)
        ax.set_xlabel("Step"); ax.set_ylabel("Mean |W_priv - W_wl|")
        ax.set_title("Cross-Attn Weight Divergence"); ax.legend(fontsize=5, ncol=3); ax.grid(alpha=0.3)
        
        # Fig 2: Output divergence
        ax = axes[0, 1]
        ax.plot(steps, history["output_diff"], 'b-', lw=2, label='Mean |Δ|')
        ax2 = ax.twinx()
        ax2.plot(steps, history["output_cosine"], 'r--', lw=2, label='Cosine sim')
        ax2.set_ylabel("Cosine Similarity", color='r')
        ax.set_xlabel("Step"); ax.set_ylabel("Mean Abs Diff", color='b')
        ax.set_title("Head Output Divergence")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, fontsize=8); ax.grid(alpha=0.3)
        
        # Fig 3: Corruption (should be flat zero)
        ax = axes[1, 0]
        ax.plot(steps, history["max_drift"], 'r-o', markersize=2, lw=2)
        ax.set_xlabel("Step"); ax.set_ylabel("Max Weight Drift")
        ax.set_title("Corruption Monitor (all frozen params)")
        ax.grid(alpha=0.3)
        if all(d == 0 for d in history["max_drift"]):
            ax.annotate("✅ Exactly 0.0 drift across all steps",
                        xy=(0.5, 0.5), xycoords='axes fraction', ha='center',
                        fontsize=13, color='green', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Fig 4: Loss
        ax = axes[1, 1]
        ax.plot(steps, history["loss"], 'k-', alpha=0.5, lw=0.5)
        w = max(1, len(steps) // 20)
        smoothed = [sum(history["loss"][max(0,i-w):i+1])/min(i+1,w+1) for i in range(len(steps))]
        ax.plot(steps, smoothed, 'orange', lw=2, label='Smoothed')
        ax.set_xlabel("Step"); ax.set_ylabel("MSE Loss")
        ax.set_title("Whitelisted Head Training Loss"); ax.legend(); ax.grid(alpha=0.3)
        
        plt.tight_layout()
        fig_path = f"results_{tag}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved to {fig_path}")
        plt.close()
        
    except ImportError:
        print("  [WARN] matplotlib not installed, skipping figures")
    
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_forked_blocks", type=int, default=6)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--rdt_repo", type=str, 
                        default="/workspace/RoboticsDiffusionTransformer",
                        help="Path to cloned RDT repo")
    args = parser.parse_args()
    run(args)
