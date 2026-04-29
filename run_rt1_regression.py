#!/usr/bin/env python3
"""
run_rt1_regression.py — Task-level regression test for the fork.

Question: does applying the fork (and training the whitelisted branch)
cause the privileged route's outputs on RDT-170M's pretraining distribution
to degrade compared to the unmodified base model?

Test:
  1. Load RDT-170M twice: once untouched (baseline), once to be forked.
  2. Download a held-out slice of RT-1 (fractal20220817_data_lerobot),
     which is in RDT's pretraining set.
  3. Evaluate baseline and forked-auth=1 on the slice. Report:
       - MSE of base predictions vs ground truth actions
       - MSE of forked-auth=1 predictions vs ground truth actions
       - Max abs diff between base and forked-auth=1 outputs
  4. Train whitelisted branch on synthetic gestures for N steps.
  5. Re-evaluate. Forked-auth=1 should still be bit-identical to baseline.

Expected result:
  - base_mse and forked_auth1_mse_pre identical
  - base_mse and forked_auth1_mse_post identical
  - max_abs_diff exactly 0.0 at all three checkpoints

Notes:
  We use random tensors for language and image conditioning. Both models
  see the same randoms, so the regression claim (forked-auth=1 == base on
  identical inputs) is unaffected. The claim is about forward-pass
  equivalence, not about real-image task performance. If base MSE on
  ground-truth actions comes out high, that's fine: what matters is that
  both models produce the same high number, confirming equivalence.

Usage on Vast.ai A100:
    # Download RT-1 episodes once (about 20MB for 5 episodes)
    python -c "
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id='IPEC-COMMUNITY/fractal20220817_data_lerobot',
        repo_type='dataset',
        allow_patterns='data/chunk-000/episode_00000[0-4].parquet',
        local_dir='./rt1_data',
    )
    "

    # Run regression test
    python run_rt1_regression.py \\
        --rt1_dir ./rt1_data/data/chunk-000 \\
        --rdt_repo /workspace/RoboticsDiffusionTransformer \\
        --train_steps 500
"""

import argparse
import glob
import json
import os
import sys
import time
import random as _stdlib_random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from run_crossattn_fork_170m_2 import (
    CrossAttnForkedRDT, SyntheticGestureDataset,
)


# ─────────────────────────────────────────────────────────────
# RT-1 dataset slice (lightweight, actions only)
# ─────────────────────────────────────────────────────────────

class RT1HeldOut(Dataset):
    """RT-1 (fractal20220817_data_lerobot) held-out slice.

    Each RT-1 parquet file is one episode: list of frames with
    'action' (7-dim) and 'observation.state' (8-dim).

    We map:
      - 7-dim RT-1 action -> 128-dim RDT unified action space.
        RT-1 dims: [x, y, z, roll, pitch, yaw, gripper]
        RDT reserves dims 0-9 for end-effector position/orientation and
        dim 10 for gripper in the unified space. We place:
          dims 0-2: xyz translation
          dims 3-5: roll, pitch, yaw
          dim 10:   gripper
      - 8-dim state -> same mapping plus one extra quaternion component.
        We use dims 0-6 for x,y,z,rx,ry,rz,rw and dim 10 for gripper.

    Random tensors for lang/image tokens (same shapes as G1 pipeline).
    Both baseline and forked models receive the same randoms on the same
    step, so the regression comparison is valid.
    """

    def __init__(self, data_dir, horizon=64, stride=8, max_episodes=5,
                 lang_len=1024, img_len=4374):
        self.horizon = horizon
        self.lang_len = lang_len
        self.img_len = img_len
        files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))[:max_episodes]
        if not files:
            raise RuntimeError(f"no parquet files in {data_dir}")

        print(f"  [rt1] loading {len(files)} episodes")

        all_actions_128 = []
        all_states_128 = []
        total_frames = 0
        for f in files:
            df = pd.read_parquet(f)
            actions_7 = np.stack(df['action'].values).astype(np.float32)
            states_8 = np.stack(df['observation.state'].values).astype(np.float32)
            n = len(actions_7)
            total_frames += n

            a128 = np.zeros((n, 128), dtype=np.float32)
            s128 = np.zeros((n, 128), dtype=np.float32)

            # Actions: xyz, rpy, gripper
            a128[:, 0:3] = actions_7[:, 0:3]   # xyz
            a128[:, 3:6] = actions_7[:, 3:6]   # roll, pitch, yaw
            a128[:, 10] = actions_7[:, 6]      # gripper

            # States: x, y, z, rx, ry, rz, rw (quaternion), gripper
            s128[:, 0:7] = states_8[:, 0:7]
            s128[:, 10] = states_8[:, 7]

            all_actions_128.append(a128)
            all_states_128.append(s128)

        self.actions_128 = np.concatenate(all_actions_128)
        self.states_128 = np.concatenate(all_states_128)
        print(f"  [rt1] {total_frames} frames total")

        N = self.actions_128.shape[0]
        self.chunks = []
        self.state_inits = []
        for i in range(0, N - horizon, stride):
            self.chunks.append(self.actions_128[i:i+horizon])
            self.state_inits.append(self.states_128[i:i+1])
        print(f"  [rt1] {len(self.chunks)} chunks (horizon={horizon}, stride={stride})")

        var = np.var(self.actions_128, axis=0)
        self.active = np.where(var > 1e-6)[0]
        self.mask = np.zeros((1, 128), dtype=np.float32)
        self.mask[0, self.active] = 1.0
        print(f"  [rt1] active dims: {self.active.tolist()}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return {
            "lang_tokens": torch.randn(self.lang_len, 4096, dtype=torch.bfloat16),
            "img_tokens": torch.randn(self.img_len, 1152, dtype=torch.bfloat16),
            "state_tokens": torch.from_numpy(self.state_inits[idx]).to(torch.bfloat16),
            "action_gt": torch.from_numpy(self.chunks[idx]).to(torch.bfloat16),
            "action_mask": torch.from_numpy(self.mask).to(torch.bfloat16),
            "ctrl_freqs": torch.tensor(3.0),
        }


# ─────────────────────────────────────────────────────────────
# Forward pass helpers
# ─────────────────────────────────────────────────────────────

def prepare_forward_inputs(batch, runner, device):
    """Build the DiT inputs from a dataloader batch, matching how
    the diffusion training loop does it. Uses fixed t=0 for deterministic
    evaluation (no noise added).
    """
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}

    state_action = torch.cat([batch["state_tokens"], batch["action_gt"]], dim=1)
    mask_exp = batch["action_mask"].expand(-1, state_action.shape[1], -1)
    state_action = torch.cat([state_action, mask_exp], dim=2)

    with torch.no_grad():
        lang_c, img_c, sa_cond = runner.adapt_conditions(
            batch["lang_tokens"], batch["img_tokens"], state_action)

    timesteps = torch.zeros(batch["action_gt"].shape[0], device=device).long()
    return {
        "x": sa_cond, "freq": batch["ctrl_freqs"], "t": timesteps,
        "lang_c": lang_c, "img_c": img_c,
        "action_gt": batch["action_gt"],
        "mask": batch["action_mask"],
    }


def evaluate(baseline_model, forked_model, runner, loader, device, active_dims):
    """Run baseline and forked auth=1 on the loader. Returns a dict of metrics."""
    baseline_model.eval(); forked_model.eval()

    n = 0
    base_mse_sum = 0.0
    fork_mse_sum = 0.0
    abs_diff_sum = 0.0
    abs_diff_peak = 0.0
    cos_sum = 0.0

    active_dims_t = torch.tensor(active_dims, device=device)

    for batch in loader:
        inp = prepare_forward_inputs(batch, runner, device)

        with torch.no_grad():
            out_base = baseline_model(
                inp["x"], inp["freq"], inp["t"], inp["lang_c"], inp["img_c"])
            out_fork = forked_model(
                inp["x"], inp["freq"], inp["t"], inp["lang_c"], inp["img_c"],
                head="privileged")

        # MSE against ground truth on active dims only
        gt = inp["action_gt"].index_select(-1, active_dims_t).float()
        ob = out_base.index_select(-1, active_dims_t).float()
        of = out_fork.index_select(-1, active_dims_t).float()

        base_mse_sum += F.mse_loss(ob, gt, reduction='mean').item()
        fork_mse_sum += F.mse_loss(of, gt, reduction='mean').item()

        # Equivalence between base and fork-auth=1
        diff = (out_base - out_fork).abs()
        abs_diff_sum += diff.mean().item()
        abs_diff_peak = max(abs_diff_peak, diff.max().item())

        cos = F.cosine_similarity(
            out_base.reshape(1, -1).float(),
            out_fork.reshape(1, -1).float()
        ).item()
        cos_sum += cos

        n += 1

    return {
        "n_batches": n,
        "base_mse_vs_gt": base_mse_sum / n,
        "fork_auth1_mse_vs_gt": fork_mse_sum / n,
        "base_vs_fork_auth1_mean_abs_diff": abs_diff_sum / n,
        "base_vs_fork_auth1_peak_abs_diff": abs_diff_peak,
        "base_vs_fork_auth1_mean_cosine": cos_sum / n,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"{'='*70}")
    print(f"  RT-1 Regression Test: does forking regress base capability?")
    print(f"  Device: {device}, dtype: {dtype}")
    print(f"  Training steps (synthetic gestures): {args.train_steps}")
    print(f"{'='*70}")

    # Determinism
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _stdlib_random.seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Load RDT twice ──
    print("\n[1/5] Loading RDT-170M (baseline)...")
    sys.path.insert(0, args.rdt_repo)
    os.chdir(args.rdt_repo)
    from models.rdt_runner import RDTRunner

    baseline_runner = RDTRunner.from_pretrained(
        "robotics-diffusion-transformer/rdt-170m", dtype=dtype
    ).to(device=device, dtype=dtype)
    for p in baseline_runner.parameters():
        p.requires_grad = False
    baseline_model = baseline_runner.model
    baseline_model.eval()

    print("[2/5] Loading RDT-170M (to be forked)...")
    fork_runner = RDTRunner.from_pretrained(
        "robotics-diffusion-transformer/rdt-170m", dtype=dtype
    ).to(device=device, dtype=dtype)
    for p in fork_runner.lang_adaptor.parameters():
        p.requires_grad = False
    for p in fork_runner.img_adaptor.parameters():
        p.requires_grad = False
    for p in fork_runner.state_adaptor.parameters():
        p.requires_grad = False

    forked = CrossAttnForkedRDT(
        fork_runner.model, num_forked_blocks=args.num_forked, fork_final=True
    ).to(device)
    forked.eval()

    # ── RT-1 held-out ──
    print("\n[3/5] Loading RT-1 held-out slice...")
    rt1 = RT1HeldOut(
        args.rt1_dir, horizon=64, stride=8, max_episodes=args.max_episodes,
        lang_len=baseline_runner.model.lang_cond_pos_embed.shape[1],
        img_len=baseline_runner.model.img_cond_pos_embed.shape[1],
    )
    rt1_loader = DataLoader(rt1, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, drop_last=True)

    # ── Evaluation: pre-training ──
    print("\n[4/5] Evaluating PRE-training...")
    pre = evaluate(baseline_model, forked, baseline_runner, rt1_loader,
                   device, rt1.active.tolist())

    print(f"\n  Pre-training results:")
    print(f"    Base MSE vs GT:                   {pre['base_mse_vs_gt']:.6f}")
    print(f"    Fork auth=1 MSE vs GT:            {pre['fork_auth1_mse_vs_gt']:.6f}")
    print(f"    Base vs Fork auth=1 mean |diff|:  {pre['base_vs_fork_auth1_mean_abs_diff']:.2e}")
    print(f"    Base vs Fork auth=1 peak |diff|:  {pre['base_vs_fork_auth1_peak_abs_diff']:.2e}")
    print(f"    Base vs Fork auth=1 mean cosine:  {pre['base_vs_fork_auth1_mean_cosine']:.6f}")

    # ── Train whitelisted branch on synthetic gestures ──
    print(f"\n[5/5] Training whitelisted branch on synthetic gestures "
          f"({args.train_steps} steps)...")

    synth = SyntheticGestureDataset(
        num_samples=args.train_steps * args.batch_size,
        horizon=64, state_dim=128, lang_dim=4096, img_dim=1152,
    )
    synth_loader = DataLoader(synth, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, drop_last=True)

    forked.train()
    trainable = [p for p in forked.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    step = 0
    t0 = time.time()
    print(f"  {'Step':>6s} | {'Loss':>10s} | {'Time':>6s}")

    while step < args.train_steps:
        for batch in synth_loader:
            if step >= args.train_steps:
                break
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            # Cast float tensors (not ctrl_freqs, not int timesteps) to bfloat16
            for k in ("lang_tokens", "img_tokens", "state_tokens",
                      "action_gt", "action_mask"):
                batch[k] = batch[k].to(torch.bfloat16)

            with torch.no_grad():
                sa = torch.cat([batch["state_tokens"], batch["action_gt"]], dim=1)
                mask_exp = batch["action_mask"].expand(-1, sa.shape[1], -1)
                sa = torch.cat([sa, mask_exp], dim=2)
                _, _, _ = fork_runner.adapt_conditions(
                    batch["lang_tokens"], batch["img_tokens"], sa)

            noise = torch.randn_like(batch["action_gt"])
            timesteps = torch.randint(
                0, fork_runner.num_train_timesteps,
                (batch["action_gt"].shape[0],), device=device
            ).long()
            noisy_action = fork_runner.noise_scheduler.add_noise(
                batch["action_gt"], noise, timesteps)
            noisy_sa = torch.cat([batch["state_tokens"], noisy_action], dim=1)
            mask_exp2 = batch["action_mask"].expand(-1, noisy_sa.shape[1], -1)
            noisy_sa = torch.cat([noisy_sa, mask_exp2], dim=2)
            lang_c, img_c, noisy_sa_cond = fork_runner.adapt_conditions(
                batch["lang_tokens"], batch["img_tokens"], noisy_sa)

            pred = forked(
                noisy_sa_cond, batch["ctrl_freqs"], timesteps,
                lang_c.detach(), img_c.detach(), head="whitelisted",
            )
            loss = F.mse_loss(pred, batch["action_gt"])

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 100 == 0:
                elapsed = time.time() - t0
                print(f"  {step:>6d} | {loss.item():>10.4f} | {elapsed:>5.0f}s")
            step += 1

    # ── Evaluation: post-training ──
    print(f"\n[post] Evaluating POST-training on RT-1 held-out...")
    post = evaluate(baseline_model, forked, baseline_runner, rt1_loader,
                    device, rt1.active.tolist())

    print(f"\n  Post-training results:")
    print(f"    Base MSE vs GT:                   {post['base_mse_vs_gt']:.6f}")
    print(f"    Fork auth=1 MSE vs GT:            {post['fork_auth1_mse_vs_gt']:.6f}")
    print(f"    Base vs Fork auth=1 mean |diff|:  {post['base_vs_fork_auth1_mean_abs_diff']:.2e}")
    print(f"    Base vs Fork auth=1 peak |diff|:  {post['base_vs_fork_auth1_peak_abs_diff']:.2e}")
    print(f"    Base vs Fork auth=1 mean cosine:  {post['base_vs_fork_auth1_mean_cosine']:.6f}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  REGRESSION TEST SUMMARY")
    print(f"{'='*70}")
    print()
    print(f"  Base model MSE on RT-1 held-out:     {post['base_mse_vs_gt']:.6f}")
    print(f"  Fork auth=1 MSE on RT-1 held-out:    {post['fork_auth1_mse_vs_gt']:.6f}")
    mse_delta = abs(post['fork_auth1_mse_vs_gt'] - post['base_mse_vs_gt'])
    print(f"  MSE delta:                           {mse_delta:.2e}")
    print()
    print(f"  Peak output diff PRE training:       {pre['base_vs_fork_auth1_peak_abs_diff']:.2e}")
    print(f"  Peak output diff POST training:      {post['base_vs_fork_auth1_peak_abs_diff']:.2e}")

    if post['base_vs_fork_auth1_peak_abs_diff'] == 0.0:
        print(f"\n  PASS: fork does not regress base model on RT-1 held-out data.")
        print(f"        Privileged route output is bit-identical to baseline,")
        print(f"        before and after training the whitelisted branch.")
    else:
        print(f"\n  DRIFT DETECTED: {post['base_vs_fork_auth1_peak_abs_diff']:.2e}")

    # ── Save report ──
    report = {
        "config": {
            "num_forked": args.num_forked,
            "train_steps": args.train_steps,
            "max_episodes": args.max_episodes,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "pre_training": pre,
        "post_training": post,
        "summary": {
            "base_mse_rt1": post['base_mse_vs_gt'],
            "fork_auth1_mse_rt1": post['fork_auth1_mse_vs_gt'],
            "mse_delta": mse_delta,
            "peak_diff_pre": pre['base_vs_fork_auth1_peak_abs_diff'],
            "peak_diff_post": post['base_vs_fork_auth1_peak_abs_diff'],
            "regression_detected": post['base_vs_fork_auth1_peak_abs_diff'] > 0.0,
        },
    }
    with open("rt1_regression_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report written to rt1_regression_report.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_forked", type=int, default=2)
    parser.add_argument("--train_steps", type=int, default=500)
    parser.add_argument("--max_episodes", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rt1_dir", type=str,
                        default="/workspace/rt1_data/data/chunk-000")
    parser.add_argument("--rdt_repo", type=str,
                        default="/workspace/RoboticsDiffusionTransformer")
    args = parser.parse_args()
    main(args)
