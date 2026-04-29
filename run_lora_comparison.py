#!/usr/bin/env python3
"""
run_lora_comparison.py — Resource footprint: fork vs LoRA baselines

Measures VRAM and FLOPs (and param counts) for three configurations on
RDT-170M:
  1. Cross-attention fork (ours, fork=2)
  2. LoRA on cross-attn q/kv/proj in last 2 blocks (matched-scope baseline)
  3. LoRA on all attention in all 14 blocks (standard PEFT baseline)

For each config we measure:
  - trainable parameters
  - total model parameters (frozen + trainable + LoRA adapters)
  - peak VRAM during a single forward+backward pass at batch size 1
  - peak VRAM during inference-only forward at batch sizes 1, 4, 16
  - forward latency (ms) at batch sizes 1, 4, 16
  - approximate FLOPs per forward pass

No training. No convergence. Just resource numbers.

Usage:
    python run_lora_comparison.py --rdt_repo /workspace/RoboticsDiffusionTransformer
"""

import argparse
import copy
import gc
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from run_crossattn_fork_170m_2 import CrossAttnForkedRDT


# ─────────────────────────────────────────────────────────────
# Minimal LoRA implementation
# ─────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a LoRA adapter: W + BA where B is
    rank r x in, A is out x r. Original W is frozen; only A and B train.
    
    No fancy initialization — zeros for B (so initial output matches the
    frozen layer exactly) and kaiming for A.
    """
    def __init__(self, base_linear, rank=8, alpha=16):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        
        # A: (in -> r), B: (r -> out). Output added to base(x).
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        
        self.scaling = alpha / rank
        self.rank = rank
    
    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(x))


def replace_linear_with_lora(module, rank=8, alpha=16):
    """Walks a module and replaces every nn.Linear with LoRALinear in place.
    Returns the module. Mutates in place.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
        else:
            replace_linear_with_lora(child, rank=rank, alpha=alpha)
    return module


def apply_lora_cross_attn_last_n(rdt_model, num_blocks=2, rank=8, alpha=16):
    """LoRA on cross-attn q/kv/proj in last N blocks only.
    
    Freezes everything in the model first, then replaces the nn.Linear
    modules inside block.cross_attn for the last N blocks with LoRALinear.
    """
    for p in rdt_model.parameters():
        p.requires_grad = False
    
    total = len(rdt_model.blocks)
    for i in range(total - num_blocks, total):
        block = rdt_model.blocks[i]
        replace_linear_with_lora(block.cross_attn, rank=rank, alpha=alpha)
    
    return rdt_model


def apply_lora_all_attn_full(rdt_model, rank=8, alpha=16):
    """LoRA on all self-attn AND cross-attn in every block."""
    for p in rdt_model.parameters():
        p.requires_grad = False
    
    for block in rdt_model.blocks:
        replace_linear_with_lora(block.attn, rank=rank, alpha=alpha)
        replace_linear_with_lora(block.cross_attn, rank=rank, alpha=alpha)
    
    return rdt_model


# ─────────────────────────────────────────────────────────────
# Measurement helpers
# ─────────────────────────────────────────────────────────────

def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def measure_peak_vram_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e6


def build_inputs(runner, device, batch_size):
    rdt = runner.model
    img_len = rdt.img_cond_pos_embed.shape[1]
    lang_len = rdt.lang_cond_pos_embed.shape[1]
    model_dtype = next(rdt.parameters()).dtype
    
    lang_dim = 4096
    img_dim = 1152
    state_dim = 128
    horizon = 64
    
    lang_tokens = torch.randn(batch_size, lang_len, lang_dim, device=device, dtype=model_dtype)
    img_tokens = torch.randn(batch_size, img_len, img_dim, device=device, dtype=model_dtype)
    state_tokens = torch.randn(batch_size, 1, state_dim, device=device, dtype=model_dtype)
    action_gt = torch.randn(batch_size, horizon, state_dim, device=device, dtype=model_dtype)
    action_mask = torch.zeros(batch_size, 1, state_dim, device=device, dtype=model_dtype)
    action_mask[:, :, 64:70] = 1.0
    ctrl_freqs = torch.full((batch_size,), 25.0, device=device)
    
    state_action = torch.cat([state_tokens, action_gt], dim=1)
    mask_exp = action_mask.expand(-1, state_action.shape[1], -1)
    state_action = torch.cat([state_action, mask_exp], dim=2)
    
    with torch.no_grad():
        lang_c, img_c, sa_cond = runner.adapt_conditions(
            lang_tokens, img_tokens, state_action)
    
    timesteps = torch.zeros(batch_size, device=device).long()
    
    return {
        "x": sa_cond, "freq": ctrl_freqs, "t": timesteps,
        "lang_c": lang_c, "img_c": img_c,
    }


def measure_forward_latency(forward_fn, inputs, num_warmup=5, num_trials=20):
    """Returns mean forward latency in ms."""
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = forward_fn(**inputs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    times = []
    for _ in range(num_trials):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = forward_fn(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000.0)
    
    # Drop top/bottom 10% as outlier filter
    times.sort()
    n_drop = max(1, len(times) // 10)
    trimmed = times[n_drop:-n_drop] if len(times) > 2 * n_drop else times
    return sum(trimmed) / len(trimmed)


def measure_flops_approx(forward_fn, inputs):
    """Approximate FLOPs via torch.profiler. Returns total GFLOPs per forward.
    
    Note: this is approximate. We're not going to claim 2 significant
    figures of accuracy here. It's for order-of-magnitude comparison
    between configurations.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    
    counter = FlopCounterMode(display=False, depth=None)
    with counter:
        with torch.no_grad():
            _ = forward_fn(**inputs)
    
    total_flops = counter.get_total_flops()
    return total_flops / 1e9  # GFLOPs


def measure_training_vram(forward_fn, inputs, target_shape, dtype, device):
    """Forward + loss + backward at batch 1; returns peak VRAM during the
    full train step.
    """
    reset_memory()
    
    # Build a random target matching the output shape expected by the model.
    # We run one forward to get the real shape.
    with torch.no_grad():
        sample_out = forward_fn(**inputs)
    target = torch.randn_like(sample_out)
    
    reset_memory()
    
    out = forward_fn(**inputs)
    loss = F.mse_loss(out.float(), target.float())
    loss.backward()
    
    peak = measure_peak_vram_mb()
    
    # Clean up
    del out, loss, target
    reset_memory()
    
    return peak


# ─────────────────────────────────────────────────────────────
# Per-configuration runners
# ─────────────────────────────────────────────────────────────

def load_runner(rdt_repo, device, dtype):
    sys.path.insert(0, rdt_repo)
    if os.getcwd() != rdt_repo:
        os.chdir(rdt_repo)
    from models.rdt_runner import RDTRunner
    
    runner = RDTRunner.from_pretrained(
        "robotics-diffusion-transformer/rdt-170m", dtype=dtype)
    runner = runner.to(dtype=dtype).to(device)
    
    for p in runner.lang_adaptor.parameters():
        p.requires_grad = False
    for p in runner.img_adaptor.parameters():
        p.requires_grad = False
    for p in runner.state_adaptor.parameters():
        p.requires_grad = False
    
    return runner


def config_fork(rdt_repo, device, dtype, num_forked=2):
    runner = load_runner(rdt_repo, device, dtype)
    forked = CrossAttnForkedRDT(
        runner.model, num_forked_blocks=num_forked, fork_final=True
    ).to(device)
    runner.model = forked  # so runner still holds adaptors but model is forked
    
    def fwd(x, freq, t, lang_c, img_c):
        return forked(x, freq, t, lang_c, img_c, head="whitelisted")
    
    return runner, forked, fwd


def config_lora_cross_attn(rdt_repo, device, dtype, num_blocks=2, rank=8):
    runner = load_runner(rdt_repo, device, dtype)
    model = apply_lora_cross_attn_last_n(
        runner.model, num_blocks=num_blocks, rank=rank, alpha=rank*2)
    model = model.to(device).to(dtype=dtype)
    
    def fwd(x, freq, t, lang_c, img_c):
        # Match RDT's native forward signature (no head argument).
        # RDT model forward takes: x, freq, t, lang_c, img_c
        return model(x, freq, t, lang_c, img_c)
    
    return runner, model, fwd


def config_lora_full(rdt_repo, device, dtype, rank=8):
    runner = load_runner(rdt_repo, device, dtype)
    model = apply_lora_all_attn_full(runner.model, rank=rank, alpha=rank*2)
    model = model.to(device).to(dtype=dtype)
    
    def fwd(x, freq, t, lang_c, img_c):
        return model(x, freq, t, lang_c, img_c)
    
    return runner, model, fwd


# ─────────────────────────────────────────────────────────────
# Measurement driver
# ─────────────────────────────────────────────────────────────

def measure_config(name, setup_fn, args, device, dtype):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    
    reset_memory()
    runner, model, fwd = setup_fn()
    
    trainable, total = count_params(model)
    print(f"  Trainable params: {trainable:,} ({trainable/total*100:.2f}%)")
    print(f"  Total params:     {total:,}")
    
    results = {
        "name": name,
        "trainable_params": trainable,
        "total_params": total,
        "batch_results": {},
    }
    
    # Inference VRAM and latency at each batch size
    for bs in args.batch_sizes:
        reset_memory()
        try:
            inputs = build_inputs(runner, device, batch_size=bs)
            
            # Peak VRAM during a single inference forward
            reset_memory()
            with torch.no_grad():
                _ = fwd(**inputs)
            infer_vram = measure_peak_vram_mb()
            
            # Latency
            latency_ms = measure_forward_latency(
                fwd, inputs, num_warmup=args.warmup, num_trials=args.trials)
            
            # FLOPs (only need to measure once but we do it per-batch to see scaling)
            flops_gf = measure_flops_approx(fwd, inputs)
            
            print(f"  bs={bs:3d}: VRAM={infer_vram:7.1f}MB  "
                  f"latency={latency_ms:7.2f}ms  "
                  f"FLOPs={flops_gf:.2f}GF" if flops_gf else
                  f"  bs={bs:3d}: VRAM={infer_vram:7.1f}MB  "
                  f"latency={latency_ms:7.2f}ms")
            
            results["batch_results"][str(bs)] = {
                "inference_vram_mb": infer_vram,
                "latency_ms": latency_ms,
                "gflops": flops_gf,
            }
            
            del inputs
            reset_memory()
        except torch.cuda.OutOfMemoryError:
            print(f"  bs={bs:3d}: OOM")
            results["batch_results"][str(bs)] = {"oom": True}
            reset_memory()
    
    # Training VRAM at batch size 1 (forward + backward)
    try:
        reset_memory()
        inputs = build_inputs(runner, device, batch_size=1)
        train_vram = measure_training_vram(fwd, inputs, None, dtype, device)
        print(f"  Training VRAM (bs=1, fwd+bwd): {train_vram:.1f}MB")
        results["training_vram_mb_bs1"] = train_vram
        del inputs
        reset_memory()
    except torch.cuda.OutOfMemoryError:
        print(f"  Training VRAM: OOM")
        results["training_vram_mb_bs1"] = None
    except Exception as e:
        print(f"  Training VRAM: error ({e})")
        results["training_vram_mb_bs1"] = None
    
    # Clean up
    del runner, model, fwd
    reset_memory()
    
    return results


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    print(f"{'='*70}")
    print(f"  Resource Comparison: Fork vs LoRA")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"{'='*70}")
    
    all_results = []
    
    # 1. Cross-attention fork (ours)
    all_results.append(measure_config(
        f"Cross-Attention Fork (ours, last {args.num_forked} blocks)",
        lambda: config_fork(args.rdt_repo, device, dtype, num_forked=args.num_forked),
        args, device, dtype,
    ))
    
    # 2. LoRA on cross-attn in last N blocks (matched scope)
    all_results.append(measure_config(
        f"LoRA cross-attn only (last {args.num_forked} blocks, r={args.lora_rank})",
        lambda: config_lora_cross_attn(
            args.rdt_repo, device, dtype,
            num_blocks=args.num_forked, rank=args.lora_rank),
        args, device, dtype,
    ))
    
    # 3. LoRA on all attention (full model, standard PEFT)
    all_results.append(measure_config(
        f"LoRA all attn, full DiT (14 blocks, r={args.lora_rank})",
        lambda: config_lora_full(args.rdt_repo, device, dtype, rank=args.lora_rank),
        args, device, dtype,
    ))
    
    # ── Summary table ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*70}\n")
    
    # Header
    print(f"  {'Config':<50s} {'Trainable':>12s} {'Total':>12s}")
    for r in all_results:
        print(f"  {r['name']:<50s} {r['trainable_params']:>12,} {r['total_params']:>12,}")
    
    print(f"\n  Inference VRAM (MB):")
    print(f"  {'Config':<50s}", end="")
    for bs in args.batch_sizes:
        print(f" {'bs='+str(bs):>10s}", end="")
    print()
    for r in all_results:
        print(f"  {r['name']:<50s}", end="")
        for bs in args.batch_sizes:
            v = r["batch_results"].get(str(bs), {})
            if v.get("oom"):
                print(f" {'OOM':>10s}", end="")
            else:
                print(f" {v.get('inference_vram_mb', 0):>10.1f}", end="")
        print()
    
    print(f"\n  Forward Latency (ms):")
    print(f"  {'Config':<50s}", end="")
    for bs in args.batch_sizes:
        print(f" {'bs='+str(bs):>10s}", end="")
    print()
    for r in all_results:
        print(f"  {r['name']:<50s}", end="")
        for bs in args.batch_sizes:
            v = r["batch_results"].get(str(bs), {})
            if v.get("oom"):
                print(f" {'OOM':>10s}", end="")
            else:
                print(f" {v.get('latency_ms', 0):>10.2f}", end="")
        print()
    
    print(f"\n  Forward GFLOPs:")
    print(f"  {'Config':<50s}", end="")
    for bs in args.batch_sizes:
        print(f" {'bs='+str(bs):>10s}", end="")
    print()
    for r in all_results:
        print(f"  {r['name']:<50s}", end="")
        for bs in args.batch_sizes:
            v = r["batch_results"].get(str(bs), {})
            if v.get("oom"):
                print(f" {'OOM':>10s}", end="")
            else:
                g = v.get("gflops")
                print(f" {g:>10.2f}" if g else f" {'N/A':>10s}", end="")
        print()
    
    print(f"\n  Training VRAM bs=1 (fwd+bwd, MB):")
    for r in all_results:
        v = r["training_vram_mb_bs1"]
        print(f"  {r['name']:<50s} {v:>10.1f}" if v else
              f"  {r['name']:<50s} {'N/A':>10s}")
    
    out = {
        "config": {
            "num_forked": args.num_forked,
            "lora_rank": args.lora_rank,
            "batch_sizes": args.batch_sizes,
            "dtype": str(dtype),
        },
        "results": all_results,
    }
    out_path = "lora_comparison_report.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Report written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_forked", type=int, default=2)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--rdt_repo", type=str,
                        default="/workspace/RoboticsDiffusionTransformer")
    args = parser.parse_args()
    main(args)
