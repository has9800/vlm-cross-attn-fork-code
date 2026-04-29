#!/usr/bin/env python3
"""
patch_g1_for_seeds.py — Adds --seed and --output_tag args to run_fork2_g1.py

One-shot patch. Run this once on your Vast.ai instance before running the
multi-seed sweep. Idempotent — safe to run twice.

Changes:
  - Adds --seed and --output_tag CLI args
  - Seeds torch, numpy, python random, and CUDA deterministically
  - Renames results_fork2_g1.json to results_fork2_g1_{output_tag}.json

Usage:
    python patch_g1_for_seeds.py /workspace/run_fork2_g1.py
"""

import re
import sys


PATCH_SEED_IMPORT = '''import argparse, copy, glob, json, os, sys, time
import random as _stdlib_random
import numpy as _np
'''

PATCH_SEED_FN = '''

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

'''


def apply_patch(path):
    with open(path) as f:
        src = f.read()
    
    # 1. Add seed-supporting imports at top
    old_import = "import argparse, copy, glob, json, os, sys, time"
    if "_stdlib_random" in src:
        print(f"[patch] {path} already has seed imports, skipping import patch")
    else:
        src = src.replace(old_import, PATCH_SEED_IMPORT.strip(), 1)
        # Add the _set_all_seeds helper right after imports
        # Find a good anchor: the first blank line after the imports
        src = src.replace(
            "import numpy as _np\n",
            "import numpy as _np\n" + PATCH_SEED_FN,
            1,
        )
    
    # 2. Add --seed and --output_tag args to argparse
    if "--seed" in src:
        print(f"[patch] {path} already has --seed arg, skipping argparse patch")
    else:
        old_arg_block = 'p.add_argument("--speed_max", type=float, default=1.2)'
        new_arg_block = (
            'p.add_argument("--speed_max", type=float, default=1.2)\n'
            '    p.add_argument("--seed", type=int, default=None,\n'
            '                   help="Random seed for reproducibility")\n'
            '    p.add_argument("--output_tag", type=str, default="",\n'
            '                   help="Suffix for output filenames (e.g. seed0)")'
        )
        src = src.replace(old_arg_block, new_arg_block)
    
    # 3. Call _set_all_seeds at the top of run()
    if "_set_all_seeds(args.seed)" in src:
        print(f"[patch] {path} already seeds in run(), skipping run() patch")
    else:
        old_run_header = 'def run(args):\n    device = "cuda"'
        new_run_header = (
            'def run(args):\n'
            '    if args.seed is not None:\n'
            '        _set_all_seeds(args.seed)\n'
            '        print(f"[seed] set to {args.seed}")\n'
            '    device = "cuda"'
        )
        src = src.replace(old_run_header, new_run_header)
    
    # 4. Use output_tag for the JSON filename
    if 'results_fork2_g1_{' in src or "results_fork2_g1_{args.output_tag" in src:
        print(f"[patch] {path} already uses output_tag for filename, skipping")
    else:
        src = src.replace(
            '"results_fork2_g1.json"',
            '(f"results_fork2_g1_{args.output_tag}.json" if args.output_tag '
            'else "results_fork2_g1.json")',
        )
    
    with open(path, "w") as f:
        f.write(src)
    print(f"[patch] wrote {path}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/run_fork2_g1.py"
    apply_patch(path)
