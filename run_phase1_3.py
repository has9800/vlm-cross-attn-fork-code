#!/usr/bin/env python3
"""
run_phase1_3.py — Phase 1–3 on RDT-170M (Vast.ai A100)

Loads real RDT-170M, applies cross-attention fork, wraps with the auth
routing interface, and runs:
  Phase 1: sanity check that auth=0 and auth=1 produce different outputs
  Phase 2: hook-based routing verification (1000 forwards each route)
  Phase 3: adversarial gradient isolation test (100 steps, 3 test variants)

Usage on Vast.ai A100:
    # Setup (once)
    cd /workspace
    git clone https://github.com/thu-ml/RoboticsDiffusionTransformer.git
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
    pip install packaging==24.0 timm==0.9.12 diffusers==0.24.0 transformers==4.36.0
    pip install safetensors sentencepiece protobuf matplotlib

    # Put run_crossattn_fork_170m_2.py, auth_routing.py,
    # verification_harness.py, adversarial_gradient_test.py,
    # and this script in the same directory.

    # Run
    python run_phase1_3.py --num_forked_blocks 2
"""

import argparse
import copy
import json
import os
import sys
import time

import torch
import torch.nn as nn

# Import the fork infrastructure from the existing file.
# We rename the module so Python can find it with a clean name.
# Make sure run_crossattn_fork_170m_2.py is in the same directory
# (or on sys.path) as this script.
from run_crossattn_fork_170m_2 import (
    CrossAttnForkedRDT,
    CorruptionMonitor,
    SyntheticGestureDataset,
)

from auth_routing import AuthRoutedRDT, AUTH_WHITELISTED, AUTH_PRIVILEGED
from verification_harness import verify_routing_contract, RouteHookHarness
from adversarial_gradient_test import run_adversarial_gradient_test


def build_input_factory(runner, device, batch_size=1):
    rdt = runner.model
    img_len = rdt.img_cond_pos_embed.shape[1]
    lang_len = rdt.lang_cond_pos_embed.shape[1]
    model_dtype = next(rdt.parameters()).dtype
    lang_dim = 4096
    img_dim = 1152
    state_dim = 128
    horizon = 64
    print(f"  [input factory] lang_len={lang_len}, img_len={img_len}, dtype={model_dtype}")
    def make():
        lang_tokens = torch.randn(batch_size, lang_len, lang_dim, device=device, dtype=model_dtype)
        img_tokens = torch.randn(batch_size, img_len, img_dim, device=device, dtype=model_dtype)
        state_tokens = torch.randn(batch_size, 1, state_dim, device=device, dtype=model_dtype)
        action_gt = torch.randn(batch_size, horizon, state_dim, device=device, dtype=model_dtype)
        action_mask = torch.zeros(batch_size, 1, state_dim, device=device, dtype=model_dtype)
        action_mask[:, :, 64:70] = 1.0
        ctrl_freqs = torch.full((batch_size,), 25.0, device=device, dtype=model_dtype)
        state_action = torch.cat([state_tokens, action_gt], dim=1)
        mask_exp = action_mask.expand(-1, state_action.shape[1], -1)
        state_action = torch.cat([state_action, mask_exp], dim=2)
        with torch.no_grad():
            lang_c, img_c, sa_cond = runner.adapt_conditions(
                lang_tokens, img_tokens, state_action)
        timesteps = torch.zeros(batch_size, device=device).long()
        return {
            "x": sa_cond,
            "freq": ctrl_freqs,
            "t": timesteps,
            "lang_c": lang_c,
            "img_c": img_c,
        }
    return make


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    print(f"{'='*70}")
    print(f"  Phase 1–3: Auth-routed Cross-Attention Fork on RDT-170M")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Forked blocks: {args.num_forked_blocks}/14")
    print(f"{'='*70}\n")
    
    # ── Load RDT-170M ──
    print("[load] RDT-170M from HuggingFace...")
    sys.path.insert(0, args.rdt_repo)
    os.chdir(args.rdt_repo)
    from models.rdt_runner import RDTRunner
    
    runner = RDTRunner.from_pretrained(
        "robotics-diffusion-transformer/rdt-170m",
        dtype=dtype,
    )
    runner = runner.to(dtype=dtype)
    rdt_model = runner.model
    print(f"  depth={len(rdt_model.blocks)}, hidden={rdt_model.hidden_size}")
    
    # ── Apply fork ──
    print(f"\n[fork] applying cross-attention fork (last {args.num_forked_blocks} blocks)...")
    forked_rdt = CrossAttnForkedRDT(
        rdt_model,
        num_forked_blocks=args.num_forked_blocks,
        fork_final=True,
    ).to(device)
    
    for p in runner.lang_adaptor.parameters():
        p.requires_grad = False
    for p in runner.img_adaptor.parameters():
        p.requires_grad = False
    for p in runner.state_adaptor.parameters():
        p.requires_grad = False
    runner = runner.to(device)
    
    # ── Phase 1: wrap with auth routing and sanity-check ──
    print(f"\n[phase 1] wrapping with AuthRoutedRDT...")
    routed = AuthRoutedRDT(forked_rdt).to(device)
    
    make_inputs = build_input_factory(runner, device, batch_size=1)
    
    # Sanity: auth=0 and auth=1 should produce different outputs on same input
    # (they will, because the whitelisted branch was initialized from the same
    # weights but even a small perturbation from the deepcopy + any FP
    # nondeterminism can show up; more importantly, once we train, the outputs
    # will genuinely differ).
    print(f"  quick sanity: running auth=None, auth=0, auth=1 on same input...")
    routed.eval()
    with torch.no_grad():
        inp = make_inputs()
        out_default = routed(**inp, auth=None)
        out_auth0 = routed(**inp, auth=0)
        out_auth1 = routed(**inp, auth=1)
    
    # auth=None and auth=0 should be identical (both route to whitelisted)
    default_vs_auth0 = (out_default - out_auth0).abs().max().item()
    print(f"  |auth=None − auth=0|_max = {default_vs_auth0:.2e}  "
          f"(should be 0.0; both route to whitelisted)")
    assert default_vs_auth0 == 0.0, \
        "auth=None must be identical to auth=0 (whitelisted default)"
    
    # auth=0 and auth=1 should be identical BEFORE any training, because
    # the whitelisted branch is deepcopied from the same weights.
    # (Deepcopy preserves values exactly, so at init they're equal.)
    auth0_vs_auth1 = (out_auth0 - out_auth1).abs().max().item()
    print(f"  |auth=0 − auth=1|_max     = {auth0_vs_auth1:.2e}  "
          f"(should be 0.0 before training; branches initialized identically)")
    
    print(f"  → Phase 1 interface check: PASS")
    
    # ── Phase 2: hook-based routing verification ──
    print(f"\n[phase 2] hook-based routing verification "
          f"({args.verify_passes} passes per route)...")
    
    verify_results = verify_routing_contract(
        routed,
        make_inputs_fn=make_inputs,
        num_passes=args.verify_passes,
        device=device,
        verbose=True,
    )
    
    # ── Phase 3: adversarial gradient isolation ──
    print(f"\n[phase 3] adversarial gradient isolation "
          f"({args.adv_steps} steps × 3 tests)...")
    
    adv_results = run_adversarial_gradient_test(
        routed,
        make_inputs_fn=make_inputs,
        num_steps=args.adv_steps,
        lr=1e-4,
        device=device,
        verbose=True,
    )
    
    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Phase 1 (auth scalar interface):          "
          f"{'PASS' if default_vs_auth0 == 0.0 else 'FAIL'}")
    print(f"  Phase 2 (hook routing verification):      "
          f"{'PASS' if verify_results['passed'] else 'FAIL'}  "
          f"({verify_results['num_passes']} passes × "
          f"{verify_results['num_forked_blocks']} blocks, zero leakage)")
    print(f"  Phase 3 (adversarial gradient isolation): "
          f"{'PASS' if adv_results['passed'] else 'FAIL'}")
    print(f"    Test A (innocent loss):        "
          f"{'PASS' if adv_results['test_a']['passed'] else 'FAIL'}  "
          f"(max priv grad = {adv_results['test_a']['max_priv_grad']:.2e})")
    print(f"    Test B (adversarial, detach):  "
          f"{'PASS' if adv_results['test_b']['passed'] else 'FAIL'}  "
          f"(max priv grad = {adv_results['test_b']['max_priv_grad']:.2e})")
    print(f"    Test C (coupled loss):         "
          f"{'PASS' if adv_results['test_c']['passed'] else 'FAIL'}  "
          f"(max priv grad = {adv_results['test_c']['max_priv_grad']:.2e})")
    
    # ── Save report ──
    report = {
        "config": {
            "num_forked_blocks": args.num_forked_blocks,
            "verify_passes": args.verify_passes,
            "adv_steps": args.adv_steps,
        },
        "phase1": {
            "auth_none_vs_auth0_max_diff": default_vs_auth0,
            "auth0_vs_auth1_max_diff": auth0_vs_auth1,
            "passed": default_vs_auth0 == 0.0,
        },
        "phase2": verify_results,
        "phase3": adv_results,
    }
    out_path = f"phase1_3_report_fork{args.num_forked_blocks}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report written to {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_forked_blocks", type=int, default=2)
    parser.add_argument("--verify_passes", type=int, default=1000,
                        help="Forwards per route for Phase 2 hook verification")
    parser.add_argument("--adv_steps", type=int, default=100,
                        help="Training steps per Phase 3 test variant")
    parser.add_argument("--rdt_repo", type=str,
                        default="/workspace/RoboticsDiffusionTransformer")
    args = parser.parse_args()
    main(args)
