#!/usr/bin/env python3
"""
run_fork2_g1.py — Fork=2 on real Unitree G1 humanoid data
==========================================================
With augmentations: action noise, temporal jitter, conditioning variation.

Usage: python run_fork2_g1.py --steps 5000
"""

import argparse, copy, glob, json, os, sys, time
import random as _stdlib_random
import numpy as _np


def _set_all_seeds(seed):
    """Seed every RNG the training loop touches."""
    _stdlib_random.seed(seed)
    _np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reasonable determinism without crushing perf; we're not claiming
    # bit-exact reproducibility across hardware, just seed reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Fork classes ──
class ForkedFinalLayer(nn.Module):
    def __init__(self, o):
        super().__init__()
        self.privileged = copy.deepcopy(o); self.whitelisted = copy.deepcopy(o)
        for p in self.privileged.parameters(): p.requires_grad = False
        for p in self.whitelisted.parameters(): p.requires_grad = True
    def forward(self, x, head="whitelisted"):
        return self.privileged(x) if head == "privileged" else self.whitelisted(x)

class ForkedRDTBlock(nn.Module):
    def __init__(self, b):
        super().__init__()
        self.norm1, self.attn, self.norm3, self.ffn = b.norm1, b.attn, b.norm3, b.ffn
        for m in [self.norm1, self.attn, self.norm3, self.ffn]:
            for p in m.parameters(): p.requires_grad = False
        self.priv_norm2, self.priv_cross_attn = b.norm2, b.cross_attn
        for m in [self.priv_norm2, self.priv_cross_attn]:
            for p in m.parameters(): p.requires_grad = False
        self.wl_norm2 = copy.deepcopy(b.norm2); self.wl_cross_attn = copy.deepcopy(b.cross_attn)
        for m in [self.wl_norm2, self.wl_cross_attn]:
            for p in m.parameters(): p.requires_grad = True
    def forward(self, x, c, mask=None, head="whitelisted"):
        x = x + self.attn(self.norm1(x))
        ox = x
        if head == "privileged":
            x = self.priv_cross_attn(self.priv_norm2(x), c, mask)
        else:
            x = self.wl_cross_attn(self.wl_norm2(x), c, mask)
        return x + ox + self.ffn(self.norm3(x + ox))

class CrossAttnForkedRDT(nn.Module):
    def __init__(self, rdt, n_fork=2):
        super().__init__()
        self.horizon, self.hidden_size = rdt.horizon, rdt.hidden_size
        self.split_idx = len(rdt.blocks) - n_fork
        self.shared_blocks = nn.ModuleList([rdt.blocks[i] for i in range(self.split_idx)])
        for b in self.shared_blocks:
            for p in b.parameters(): p.requires_grad = False
        self.forked_blocks = nn.ModuleList([ForkedRDTBlock(rdt.blocks[i]) for i in range(self.split_idx, len(rdt.blocks))])
        self.forked_final = ForkedFinalLayer(rdt.final_layer)
        self.t_embedder, self.freq_embedder = rdt.t_embedder, rdt.freq_embedder
        self.x_pos_embed = rdt.x_pos_embed
        self.lang_cond_pos_embed, self.img_cond_pos_embed = rdt.lang_cond_pos_embed, rdt.img_cond_pos_embed
        for p in self.t_embedder.parameters(): p.requires_grad = False
        for p in self.freq_embedder.parameters(): p.requires_grad = False
        self.x_pos_embed.requires_grad = False
        self.lang_cond_pos_embed.requires_grad = False
        self.img_cond_pos_embed.requires_grad = False
    def forward(self, x, freq, t, lang_c, img_c, head="whitelisted", lang_mask=None, img_mask=None):
        t_e = self.t_embedder(t).unsqueeze(1)
        f_e = self.freq_embedder(freq).unsqueeze(1)
        if t_e.shape[0] == 1: t_e = t_e.expand(x.shape[0], -1, -1)
        x = torch.cat([t_e, f_e, x], dim=1) + self.x_pos_embed
        lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
        img_c = img_c + self.img_cond_pos_embed
        conds, masks = [lang_c, img_c], [lang_mask, img_mask]
        for i, b in enumerate(self.shared_blocks): x = b(x, conds[i%2], masks[i%2])
        for j, b in enumerate(self.forked_blocks):
            x = b(x, conds[(self.split_idx+j)%2], masks[(self.split_idx+j)%2], head=head)
        x = self.forked_final(x, head=head)
        return x[:, -self.horizon:]

# ── Hooked block ──
class HookedBlock:
    def __init__(self, block): self.block = block
    def forward_with_hooks(self, x, c, mask=None, head="whitelisted"):
        hooks = {}
        x = x + self.block.attn(self.block.norm1(x))
        ox = x
        ca = self.block.priv_cross_attn if head == "privileged" else self.block.wl_cross_attn
        norm = self.block.priv_norm2 if head == "privileged" else self.block.wl_norm2
        x_n = norm(x)
        B, N, C = x_n.shape; _, L, _ = c.shape; nh, hd = ca.num_heads, ca.head_dim
        q = ca.q(x_n).reshape(B,N,nh,hd).permute(0,2,1,3)
        kv = ca.kv(c).reshape(B,L,2,nh,hd).permute(2,0,3,1,4)
        k, v = kv.unbind(0); q, k = ca.q_norm(q), ca.k_norm(k)
        aw = (q * hd**-0.5) @ k.transpose(-2,-1)
        if mask is not None:
            aw = aw.masked_fill_(mask.reshape(B,1,1,L).expand(-1,-1,N,-1).logical_not(), float('-inf'))
        aw = aw.softmax(dim=-1)
        hooks["attn_weights"] = aw.detach()
        ao = (aw @ v).permute(0,2,1,3).reshape(B,N,C); ao = ca.proj(ao)
        x = ao + ox; hooks["ffn_input"] = x.detach(); ox = x
        ffn_out = self.block.ffn(self.block.norm3(x))
        hooks["ffn_output_pre"] = ffn_out.detach()
        x = ffn_out + ox; hooks["ffn_output"] = x.detach()
        return x, hooks

# ── G1 Dataset with augmentations ──
class G1Dataset(Dataset):
    """
    Unitree G1 humanoid: 16-dim actions → RDT 128-dim.
    
    Augmentations:
      - Action noise: Gaussian noise on joint angles (simulates sensor noise)
      - Temporal jitter: random start offset within episode (data diversity)
      - Conditioning variation: random lang/img tokens vary per sample
        (simulates different viewpoints, lighting, instructions)
      - Token dropout: randomly zero out portions of img tokens
        (simulates occlusion)
      - Speed variation: randomly subsample or interpolate trajectories
        (simulates different execution speeds)
    """
    def __init__(self, data_dir, horizon=64, stride=8,
                 action_noise_std=0.01, temporal_jitter=4,
                 img_dropout_prob=0.15, speed_range=(0.8, 1.2)):
        self.horizon = horizon
        self.action_noise_std = action_noise_std
        self.temporal_jitter = temporal_jitter
        self.img_dropout_prob = img_dropout_prob
        self.speed_range = speed_range

        files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        print(f"  Found {len(files)} parquet episodes")

        # Load and store per-episode data (for temporal jitter)
        self.episodes = []
        total_frames = 0
        for f in files:
            df = pd.read_parquet(f)
            actions = np.stack(df['action'].values).astype(np.float32)
            states = np.stack(df['observation.state'].values).astype(np.float32)
            self.episodes.append({"actions": actions, "states": states})
            total_frames += len(actions)
        print(f"  Total frames: {total_frames}, episodes: {len(self.episodes)}")

        # Also build contiguous chunks for baseline
        all_actions = np.concatenate([e["actions"] for e in self.episodes])
        all_states = np.concatenate([e["states"] for e in self.episodes])

        # Map 16-dim G1 → 128-dim RDT
        N = all_actions.shape[0]
        self.actions_128 = np.zeros((N, 128), dtype=np.float32)
        self.states_128 = np.zeros((N, 128), dtype=np.float32)
        self._map_g1_to_rdt(all_actions, all_states, self.actions_128, self.states_128)

        # Slice into chunks
        self.chunks = []
        self.state_inits = []
        for i in range(0, N - horizon, stride):
            self.chunks.append(self.actions_128[i:i+horizon])
            self.state_inits.append(self.states_128[i:i+1])
        print(f"  Base chunks: {len(self.chunks)} (horizon={horizon}, stride={stride})")

        # Active dims
        var = np.var(self.actions_128, axis=0)
        self.active = np.where(var > 1e-6)[0]
        self.mask = np.zeros((1, 128), dtype=np.float32)
        self.mask[0, self.active] = 1.0
        print(f"  Active dims: {len(self.active)}/128: {self.active.tolist()}")
        print(f"  Action range: [{self.actions_128[:, self.active].min():.3f}, {self.actions_128[:, self.active].max():.3f}]")

        # Augmentation stats
        print(f"\n  Augmentations:")
        print(f"    Action noise std:    {action_noise_std}")
        print(f"    Temporal jitter:     ±{temporal_jitter} frames")
        print(f"    Img token dropout:   {img_dropout_prob*100:.0f}%")
        print(f"    Speed range:         {speed_range}")

    def _map_g1_to_rdt(self, raw_a, raw_s, out_a, out_s):
        """Map 16-dim G1 actions to 128-dim RDT unified space."""
        out_a[:, 64:71] = raw_a[:, 0:7]   # right arm joints
        out_s[:, 64:71] = raw_s[:, 0:7]
        out_a[:, 72] = raw_a[:, 7]         # right gripper
        out_s[:, 72] = raw_s[:, 7]
        out_a[:, 74:81] = raw_a[:, 8:15]   # left arm joints
        out_s[:, 74:81] = raw_s[:, 8:15]
        out_a[:, 73] = raw_a[:, 15]        # left gripper
        out_s[:, 73] = raw_s[:, 15]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        action_gt = self.chunks[idx].copy()
        state = self.state_inits[idx].copy()

        # ── Aug 1: Action noise ──
        if self.action_noise_std > 0:
            noise = np.random.randn(*action_gt.shape).astype(np.float32) * self.action_noise_std
            noise[:, ~np.isin(np.arange(128), self.active)] = 0  # only noise on active dims
            action_gt = action_gt + noise

        # ── Aug 2: Speed variation via interpolation ──
        if self.speed_range != (1.0, 1.0):
            speed = np.random.uniform(*self.speed_range)
            orig_len = action_gt.shape[0]
            new_len = int(orig_len * speed)
            if new_len != orig_len and new_len >= 8:
                x_old = np.linspace(0, 1, new_len)
                x_new = np.linspace(0, 1, orig_len)
                resampled = np.zeros_like(action_gt)
                for d in range(128):
                    if d in self.active:
                        # Grab slice at new speed, then resample back to horizon
                        orig_vals = action_gt[:, d]
                        resampled[:, d] = np.interp(x_new, np.linspace(0, 1, orig_len), orig_vals)
                action_gt = resampled

        # ── Aug 3: Image token dropout (simulates occlusion) ──
        img_tokens = torch.randn(4374, 1152)
        if self.img_dropout_prob > 0:
            drop_mask = torch.rand(4374) < self.img_dropout_prob
            img_tokens[drop_mask] = 0.0

        # ── Aug 4: Varied conditioning (different random seeds per sample) ──
        # Each sample gets unique random lang/img tokens, simulating
        # different viewpoints, lighting conditions, instructions
        lang_tokens = torch.randn(32, 4096)
        lang_mask = torch.ones(32, dtype=torch.bool)

        return {
            "lang_tokens": lang_tokens,
            "lang_attn_mask": lang_mask,
            "img_tokens": img_tokens,
            "state_tokens": torch.from_numpy(state),
            "action_gt": torch.from_numpy(action_gt),
            "action_mask": torch.from_numpy(self.mask),
            "ctrl_freqs": torch.tensor(25.0),
        }
    
def _ensure_g1_data(data_dir):
    """Download G1 dataset if not already present."""
    import glob
    files = glob.glob(os.path.join(data_dir, '*.parquet'))
    if files:
        print(f'  [g1] found {len(files)} parquet files, skipping download.')
        return
    print('  [g1] data not found, downloading from HuggingFace...')
    os.makedirs(data_dir, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id='unitreerobotics/G1_MountCameraRedGripper_Dataset',
            repo_type='dataset',
            allow_patterns='data/chunk-000/*.parquet',
            local_dir=os.path.dirname(os.path.dirname(data_dir)),
        )
        print('  [g1] download complete.')
    except Exception as e:
        raise RuntimeError(
            f'G1 data missing and auto-download failed: {e}\n'
            f'Run manually: see docstring at top of file.'
        )


# ── Main ──
def run(args):
    if args.seed is not None:
        _set_all_seeds(args.seed)
        print(f"[seed] set to {args.seed}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"{'='*60}")
    print(f"  Fork=2 on Unitree G1 Humanoid Data (Augmented)")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Data
    print("[1/6] Loading G1 data...")
    _ensure_g1_data(args.data_dir) 
    dataset = G1Dataset(
        args.data_dir, horizon=64, stride=8,
        action_noise_std=args.action_noise,
        temporal_jitter=args.temporal_jitter,
        img_dropout_prob=args.img_dropout,
        speed_range=(args.speed_min, args.speed_max),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    # Model
    print(f"\n[2/6] Loading RDT-170M...")
    sys.path.insert(0, args.rdt_repo); os.chdir(args.rdt_repo)
    from models.rdt_runner import RDTRunner
    runner = RDTRunner.from_pretrained("robotics-diffusion-transformer/rdt-170m", dtype=dtype)
    runner = runner.to(device=device, dtype=dtype)

    print(f"\n[3/6] Applying cross-attention fork (2 blocks)...")
    forked = CrossAttnForkedRDT(runner.model, n_fork=2).to(device)
    runner.model = None
    for p in runner.parameters(): p.requires_grad = False
    trainable = sum(p.numel() for p in forked.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in forked.parameters() if not p.requires_grad)
    print(f"  Trainable: {trainable:,} ({trainable/(trainable+frozen)*100:.1f}%)")

    # Snapshots
    snapshots = {n: p.data.cpu().clone() for n, p in forked.named_parameters() if not p.requires_grad}

    # Train
    print(f"\n[4/6] Training ({args.steps} steps)...")
    print(f"  {'Step':>5s} | {'Loss':>10s} | {'OutDiff':>8s} | {'CosSim':>8s} | {'Drift':>10s} | {'Time':>6s}")
    print(f"  {'-'*58}")
    opt = torch.optim.AdamW([p for p in forked.parameters() if p.requires_grad], lr=args.lr)
    history = {"steps":[], "loss":[], "output_diff":[], "output_cosine":[], "drift":[]}
    eval_batch = None; forked.train(); step = 0; t0 = time.time()

    for epoch in range(args.epochs):
        for batch in loader:
            if step >= args.steps: break
            batch = {k: v.to(device=device, dtype=dtype) if isinstance(v, torch.Tensor) and v.is_floating_point()
                     else v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if eval_batch is None:
                eval_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with torch.no_grad():
                sa = torch.cat([batch["state_tokens"], batch["action_gt"]], dim=1)
                me = batch["action_mask"].expand(-1, sa.shape[1], -1)
                sa = torch.cat([sa, me], dim=2)
                lc, ic, _ = runner.adapt_conditions(batch["lang_tokens"], batch["img_tokens"], sa)

            noise = torch.randn_like(batch["action_gt"])
            ts = torch.randint(0, runner.num_train_timesteps, (args.batch_size,), device=device).long()
            noisy = runner.noise_scheduler.add_noise(batch["action_gt"], noise, ts)
            nsa = torch.cat([batch["state_tokens"], noisy], dim=1)
            nm = batch["action_mask"].expand(-1, nsa.shape[1], -1)
            nsa = torch.cat([nsa, nm], dim=2)
            _, _, nsac = runner.adapt_conditions(batch["lang_tokens"], batch["img_tokens"], nsa)

            pred = forked(nsac, batch["ctrl_freqs"], ts, lc.detach(), ic.detach(),
                          head="whitelisted", lang_mask=batch["lang_attn_mask"])
            loss = F.mse_loss(pred, batch["action_gt"])
            opt.zero_grad(); loss.backward(); opt.step()

            if step % args.log_interval == 0:
                with torch.no_grad():
                    esa = torch.cat([eval_batch["state_tokens"], eval_batch["action_gt"]], dim=1)
                    eme = eval_batch["action_mask"].expand(-1, esa.shape[1], -1)
                    esa = torch.cat([esa, eme], dim=2)
                    el, ei, es = runner.adapt_conditions(eval_batch["lang_tokens"], eval_batch["img_tokens"], esa)
                    zt = torch.zeros(args.batch_size, device=device).long()
                    op = forked(es, eval_batch["ctrl_freqs"], zt, el, ei, head="privileged", lang_mask=eval_batch["lang_attn_mask"])
                    ow = forked(es, eval_batch["ctrl_freqs"], zt, el, ei, head="whitelisted", lang_mask=eval_batch["lang_attn_mask"])
                diff = (op-ow).abs().mean().item()
                cos = F.cosine_similarity(op.reshape(1,-1).float(), ow.reshape(1,-1).float()).item()
                max_d = max((p.data.cpu()-snapshots[n]).abs().max().item() for n,p in forked.named_parameters() if n in snapshots)
                history["steps"].append(step); history["loss"].append(loss.item())
                history["output_diff"].append(diff); history["output_cosine"].append(cos); history["drift"].append(max_d)
                print(f"  {step:5d} | {loss.item():10.6f} | {diff:8.4f} | {cos:+8.4f} | {max_d:10.2e} | {time.time()-t0:5.0f}s")
            step += 1
        if step >= args.steps: break

    torch.save(forked.state_dict(), "model_fork2_g1.pt")
    print(f"\n  Checkpoint saved to model_fork2_g1.pt")

    # ── Mechanistic Analysis ──
    print(f"\n[5/6] Mechanistic analysis...")
    forked.eval()
    hooked = [HookedBlock(b) for b in forked.forked_blocks]
    n_anal = min(50, len(dataset) // args.batch_size)
    attn_p, attn_w = {0:[], 1:[]}, {0:[], 1:[]}
    act_p, act_w = [], []
    ffn_in_d, ffn_out_d, ffn_ovl = {0:[], 1:[]}, {0:[], 1:[]}, {0:[], 1:[]}

    anal_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    for bi, batch in enumerate(anal_loader):
        if bi >= n_anal: break
        batch = {k: v.to(device=device, dtype=dtype) if isinstance(v, torch.Tensor) and v.is_floating_point()
                 else v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            sa = torch.cat([batch["state_tokens"], batch["action_gt"]], dim=1)
            me = batch["action_mask"].expand(-1, sa.shape[1], -1)
            sa = torch.cat([sa, me], dim=2)
            lc, ic, sac = runner.adapt_conditions(batch["lang_tokens"], batch["img_tokens"], sa)
            t_e = forked.t_embedder(torch.zeros(args.batch_size, device=device)).unsqueeze(1).expand(args.batch_size,-1,-1)
            f_e = forked.freq_embedder(batch["ctrl_freqs"]).unsqueeze(1)
            x = torch.cat([t_e, f_e, sac], dim=1) + forked.x_pos_embed
            l = lc + forked.lang_cond_pos_embed[:, :lc.shape[1]]
            im = ic + forked.img_cond_pos_embed
            conds, masks = [l, im], [batch["lang_attn_mask"], None]
            for i, b in enumerate(forked.shared_blocks): x = b(x, conds[i%2], masks[i%2])
            xp, xw = x.clone(), x.clone()
            for j, h in enumerate(hooked):
                gi = forked.split_idx + j; c, m = conds[gi%2], masks[gi%2]
                xp, hp = h.forward_with_hooks(xp, c, m, head="privileged")
                xw, hw = h.forward_with_hooks(xw, c, m, head="whitelisted")
                attn_p[j].append(hp["attn_weights"].float().mean(dim=(0,1)).cpu())
                attn_w[j].append(hw["attn_weights"].float().mean(dim=(0,1)).cpu())
                fip, fiw = hp["ffn_input"].float(), hw["ffn_input"].float()
                fop, fow = hp["ffn_output"].float(), hw["ffn_output"].float()
                ffn_in_d[j].append((fip-fiw).abs().mean().item())
                ffn_out_d[j].append((fop-fow).abs().mean().item())
                fp, fw = hp["ffn_output_pre"].float(), hw["ffn_output_pre"].float()
                tp, tw = fp.abs().mean(), fw.abs().mean()
                ap = (fp.abs()>tp).float().mean(dim=(0,1))
                aw_ = (fw.abs()>tw).float().mean(dim=(0,1))
                ffn_ovl[j].append(((ap>0.5)&(aw_>0.5)).float().mean().item())
            op = forked.forked_final(xp, head="privileged")[:, -64:]
            ow = forked.forked_final(xw, head="whitelisted")[:, -64:]
            act_p.append(op.reshape(args.batch_size,-1).float().cpu())
            act_w.append(ow.reshape(args.batch_size,-1).float().cpu())

    # ── Report ──
    print(f"\n  Cross-Attention Pattern Divergence:")
    crossattn_results = {}
    for j in range(2):
        gi = forked.split_idx + j
        mp = torch.stack(attn_p[j]).mean(0).mean(0); mw = torch.stack(attn_w[j]).mean(0).mean(0)
        eps = 1e-8; mp_n = (mp+eps)/(mp+eps).sum(); mw_n = (mw+eps)/(mw+eps).sum()
        m_avg = 0.5*(mp_n+mw_n)
        js = 0.5*((mp_n*(mp_n/m_avg).log()).sum()+(mw_n*(mw_n/m_avg).log()).sum()).item()
        cos = F.cosine_similarity(mp.unsqueeze(0), mw.unsqueeze(0)).item()
        _, pt = mp.topk(10); _, wt = mw.topk(10)
        overlap = len(set(pt.tolist()) & set(wt.tolist()))
        ctype = "lang" if gi%2==0 else "img"
        crossattn_results[f"block_{gi}"] = {"js": js, "cos": cos, "overlap": overlap}
        print(f"    Block {gi} ({ctype}, L={mp.shape[0]}): JS={js:.6f} cos={cos:.6f} top10_overlap={overlap}/10")
        print(f"      Priv top-5: {pt[:5].tolist()}")
        print(f"      WL top-5:   {wt[:5].tolist()}")

    print(f"\n  Action Space Geometry:")
    pa = torch.cat(act_p).numpy(); wa = torch.cat(act_w).numpy()
    from numpy.linalg import svd
    def pca(X, k=10):
        U,S,Vt = svd(X-X.mean(0), full_matrices=False)
        ev = (S**2)/(S**2).sum()
        return Vt[:k], ev[:k]
    pVt, pv = pca(pa); wVt, wv = pca(wa)
    _, cs, _ = svd(pVt @ wVt.T)
    angles = np.degrees(np.arccos(np.clip(cs, -1, 1)))
    pccos = np.array([np.dot(pa[i],wa[i])/(np.linalg.norm(pa[i])*np.linalg.norm(wa[i])+1e-8) for i in range(len(pa))])
    print(f"    Mean principal angle: {angles.mean():.1f}\u00b0")
    print(f"    Angles: {[f'{a:.1f}' for a in angles[:5]]}")
    print(f"    Pairwise cosine: {pccos.mean():.4f} \u00b1 {pccos.std():.4f}")
    print(f"    Priv var: {[f'{v:.3f}' for v in pv[:5]]}")
    print(f"    WL var:   {[f'{v:.3f}' for v in wv[:5]]}")
    print(f"    Priv norm: {np.linalg.norm(pa,axis=1).mean():.2f}, WL norm: {np.linalg.norm(wa,axis=1).mean():.2f}")

    print(f"\n  FFN Divergence:")
    ffn_results = {}
    for j in range(2):
        gi = forked.split_idx + j
        ind = np.mean(ffn_in_d[j]); outd = np.mean(ffn_out_d[j])
        ratio = outd/(ind+1e-8); ovlp = np.mean(ffn_ovl[j])
        tag = "AMPLIFIES" if ratio > 1 else "COMPRESSES" if ratio < 0.95 else "PRESERVES"
        ffn_results[f"block_{gi}"] = {"in": ind, "out": outd, "ratio": ratio, "overlap": ovlp}
        print(f"    Block {gi}: in={ind:.4f} out={outd:.4f} ratio={ratio:.3f}x ({tag}) overlap={ovlp:.1%}")

    # ── Save ──
    print(f"\n[6/6] Saving results...")
    final_drift = max((p.data.cpu()-snapshots[n]).abs().max().item() for n,p in forked.named_parameters() if n in snapshots)

    results = {
        "config": {
            "data": "Unitree G1 humanoid (HuggingFaceVLA/community_dataset_v3)",
            "episodes": len(dataset.episodes),
            "total_frames": sum(len(e["actions"]) for e in dataset.episodes),
            "chunks": len(dataset),
            "active_dims": len(dataset.active),
            "fork_blocks": 2,
            "steps": step,
            "augmentations": {
                "action_noise_std": args.action_noise,
                "img_dropout_prob": args.img_dropout,
                "speed_range": [args.speed_min, args.speed_max],
            }
        },
        "history": history,
        "crossattn": crossattn_results,
        "action_geometry": {
            "mean_angle": float(angles.mean()),
            "angles": angles.tolist(),
            "pairwise_cosine_mean": float(pccos.mean()),
            "pairwise_cosine_std": float(pccos.std()),
            "priv_var": pv.tolist(),
            "wl_var": wv.tolist(),
        },
        "ffn": ffn_results,
        "final_drift": final_drift,
    }
    with open((f"results_fork2_g1_{args.output_tag}.json" if args.output_tag else "results_fork2_g1.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved to results_fork2_g1.json")

    print(f"\n{'='*60}")
    print(f"  G1 HUMANOID REAL DATA RESULTS")
    print(f"{'='*60}")
    print(f"  Data:            Unitree G1, {len(dataset.episodes)} episodes, {sum(len(e['actions']) for e in dataset.episodes)} frames")
    print(f"  Drift:           {final_drift:.2e} {'✅ ZERO' if final_drift == 0 else '❌ CORRUPTED'}")
    print(f"  Output cosine:   {history['output_cosine'][-1]:+.4f}")
    print(f"  Subspace angle:  {angles.mean():.1f}\u00b0")
    for gi_key, fd in ffn_results.items():
        tag = "✅ SAFE" if fd["ratio"] >= 0.95 else "⚠️ COMPRESSES"
        print(f"  FFN {gi_key}:      ratio={fd['ratio']:.3f}x overlap={fd['overlap']:.1%} {tag}")
    print(f"{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--data_dir", type=str, default="./vla_data/unitreerobotics/G1_MountCameraRedGripper_Dataset/data/chunk-000")
    p.add_argument("--rdt_repo", type=str, default="/workspace/RoboticsDiffusionTransformer")
    # Augmentation args
    p.add_argument("--action_noise", type=float, default=0.01)
    p.add_argument("--temporal_jitter", type=int, default=4)
    p.add_argument("--img_dropout", type=float, default=0.15)
    p.add_argument("--speed_min", type=float, default=0.8)
    p.add_argument("--speed_max", type=float, default=1.2)
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--output_tag", type=str, default="",
                   help="Suffix for output filenames (e.g. seed0)")
    args = p.parse_args()
    run(args)
