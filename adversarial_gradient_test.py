"""
adversarial_gradient_test.py — Phase 3

Constructs a loss that *explicitly depends* on privileged cross-attention
outputs, then runs training on the WHITELISTED route only. The contract:
every privileged parameter's .grad must be None or exactly zero, because
the whitelisted route never touches privileged modules, so autograd has
no path from the loss to privileged weights.

Why this is stronger than the existing step-0 leakage check:
  - Adversarial construction: we deliberately build a loss that, if
    anything could leak, would make it leak. We call the privileged
    branch, detach, compute a target from it, then train the whitelisted
    branch to match. Then we flip: we train with a loss that uses both
    branches' outputs but routes gradient only through whitelisted.
  - Multi-step: we run 100 steps, checking after each one, not just
    at step 0. Weight-update side effects could in principle compound.
  - Explicit assertion: any nonzero privileged gradient is a test failure.

What we test:
  Test A: "innocent" loss — train whitelisted on synthetic target,
          verify privileged grads are None/zero (sanity baseline).
  Test B: "adversarial" loss — loss depends on privileged outputs via a
          detached copy; we verify autograd graph still can't reach
          privileged weights.
  Test C: "coupled" loss — loss = L(wl_out) + L(priv_out.detach()).
          This is the case where a careless implementation might leak
          because both branches appear in the loss expression.
"""

import torch
import torch.nn.functional as F


def _collect_privileged_params(forked_rdt):
    """Return (name, param) pairs for every privileged parameter in the model.
    
    Privileged parameters are those in:
      - block.priv_norm2 and block.priv_cross_attn for each forked block
      - forked_final.privileged (if fork_final is True)
    These should all have requires_grad=False already.
    """
    # Reach through wrapper if given
    if hasattr(forked_rdt, "model") and hasattr(forked_rdt.model, "forked_blocks"):
        forked_rdt = forked_rdt.model
    
    priv = []
    for local_i, block in enumerate(forked_rdt.forked_blocks):
        gi = forked_rdt.split_idx + local_i
        for pname, p in block.priv_norm2.named_parameters():
            priv.append((f"block{gi}.priv_norm2.{pname}", p))
        for pname, p in block.priv_cross_attn.named_parameters():
            priv.append((f"block{gi}.priv_cross_attn.{pname}", p))
    
    if forked_rdt.fork_final:
        for pname, p in forked_rdt.forked_final.privileged.named_parameters():
            priv.append((f"forked_final.privileged.{pname}", p))
    
    return priv


def _collect_all_frozen_params(forked_rdt):
    """Every param with requires_grad=False. Used for a broader sanity check."""
    if hasattr(forked_rdt, "model") and hasattr(forked_rdt.model, "forked_blocks"):
        forked_rdt = forked_rdt.model
    
    frozen = []
    for name, p in forked_rdt.named_parameters():
        if not p.requires_grad:
            frozen.append((name, p))
    return frozen


def _max_grad_magnitude(named_params):
    """Return (max |grad|, name of offender). None-grads count as zero."""
    worst = 0.0
    worst_name = None
    for name, p in named_params:
        if p.grad is None:
            continue
        m = p.grad.abs().max().item()
        if m > worst:
            worst = m
            worst_name = name
    return worst, worst_name


def _zero_grads_everywhere(forked_rdt):
    """Zero .grad on all parameters, including frozen ones (belt and braces)."""
    if hasattr(forked_rdt, "model") and hasattr(forked_rdt.model, "forked_blocks"):
        forked_rdt = forked_rdt.model
    for _, p in forked_rdt.named_parameters():
        if p.grad is not None:
            p.grad = None


def run_adversarial_gradient_test(
    forked_rdt, make_inputs_fn, num_steps=100, lr=1e-4,
    device="cuda", verbose=True,
):
    """Run three gradient-isolation tests and return a pass/fail report.
    
    Args:
        forked_rdt: CrossAttnForkedRDT or AuthRoutedRDT. The caller is
            responsible for having set it up with forked blocks and moved
            it to device.
        make_inputs_fn: zero-arg callable returning dict of forward inputs
            (x, freq, t, lang_c, img_c), same as verification_harness.
        num_steps: steps per test.
        lr: learning rate for AdamW. Low is fine; we only care about
            where gradients go, not about convergence.
    
    Returns:
        dict with keys passed, test_a, test_b, test_c, and details.
    """
    is_wrapper = hasattr(forked_rdt, "model") and hasattr(forked_rdt.model, "forked_blocks")
    raw = forked_rdt.model if is_wrapper else forked_rdt
    
    def fwd(inp, auth):
        if is_wrapper:
            return forked_rdt(**inp, auth=auth)
        head = "privileged" if auth == 1 else "whitelisted"
        return forked_rdt(**inp, head=head)
    
    priv_params = _collect_privileged_params(raw)
    frozen_params = _collect_all_frozen_params(raw)
    trainable_params = [p for p in raw.parameters() if p.requires_grad]
    
    if verbose:
        print(f"[adv-grad] privileged params: {len(priv_params)} tensors "
              f"({sum(p.numel() for _,p in priv_params):,} weights)")
        print(f"[adv-grad] all frozen params: {len(frozen_params)} tensors")
        print(f"[adv-grad] trainable params:  {len(trainable_params)} tensors "
              f"({sum(p.numel() for p in trainable_params):,} weights)")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
    results = {"passed": True, "test_a": {}, "test_b": {}, "test_c": {}}
    
    # ─────────────────────────────────────────────────────────────
    # Test A: innocent loss — whitelisted route only
    # ─────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n[adv-grad] Test A: innocent whitelisted-only loss ({num_steps} steps)")
    
    max_priv_grad_A = 0.0
    max_frozen_grad_A = 0.0
    worst_name_A = None
    
    forked_rdt.train()
    for step in range(num_steps):
        inp = make_inputs_fn()
        inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
               for k, v in inp.items()}
        
        _zero_grads_everywhere(forked_rdt)
        optimizer.zero_grad()
        
        out = fwd(inp, auth=0)
        target = torch.randn_like(out)
        loss = F.mse_loss(out, target)
        loss.backward()
        
        m_priv, n_priv = _max_grad_magnitude(priv_params)
        m_frozen, n_frozen = _max_grad_magnitude(frozen_params)
        if m_priv > max_priv_grad_A:
            max_priv_grad_A = m_priv
            worst_name_A = n_priv
        if m_frozen > max_frozen_grad_A:
            max_frozen_grad_A = m_frozen
        
        optimizer.step()
    
    passed_A = max_priv_grad_A == 0.0 and max_frozen_grad_A == 0.0
    results["test_a"] = {
        "passed": passed_A,
        "max_priv_grad": max_priv_grad_A,
        "max_frozen_grad": max_frozen_grad_A,
        "worst_name": worst_name_A,
    }
    if verbose:
        print(f"  max privileged grad: {max_priv_grad_A:.2e}  (worst: {worst_name_A})")
        print(f"  max frozen grad:     {max_frozen_grad_A:.2e}")
        print(f"  → {'PASS' if passed_A else 'FAIL'}")
    
    # ─────────────────────────────────────────────────────────────
    # Test B: adversarial — target comes from privileged forward (detached)
    #
    # This is the paranoid version: we deliberately introduce the privileged
    # branch into the compute graph, then detach, and verify that the
    # whitelisted training pass still produces zero privileged gradients.
    # ─────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n[adv-grad] Test B: adversarial — privileged forward then detach "
              f"({num_steps} steps)")
    
    max_priv_grad_B = 0.0
    max_frozen_grad_B = 0.0
    worst_name_B = None
    
    forked_rdt.train()
    for step in range(num_steps):
        inp = make_inputs_fn()
        inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
               for k, v in inp.items()}
        
        # Privileged forward, detached — establishes target values.
        # If isolation is correct, calling this does not create
        # any gradient path from later losses into privileged weights,
        # because (a) detach kills the graph and (b) privileged weights
        # have requires_grad=False.
        with torch.no_grad():
            priv_target = fwd(inp, auth=1)
        
        _zero_grads_everywhere(forked_rdt)
        optimizer.zero_grad()
        
        wl_out = fwd(inp, auth=0)
        loss = F.mse_loss(wl_out, priv_target)
        loss.backward()
        
        m_priv, n_priv = _max_grad_magnitude(priv_params)
        m_frozen, n_frozen = _max_grad_magnitude(frozen_params)
        if m_priv > max_priv_grad_B:
            max_priv_grad_B = m_priv
            worst_name_B = n_priv
        if m_frozen > max_frozen_grad_B:
            max_frozen_grad_B = m_frozen
        
        optimizer.step()
    
    passed_B = max_priv_grad_B == 0.0 and max_frozen_grad_B == 0.0
    results["test_b"] = {
        "passed": passed_B,
        "max_priv_grad": max_priv_grad_B,
        "max_frozen_grad": max_frozen_grad_B,
        "worst_name": worst_name_B,
    }
    if verbose:
        print(f"  max privileged grad: {max_priv_grad_B:.2e}  (worst: {worst_name_B})")
        print(f"  max frozen grad:     {max_frozen_grad_B:.2e}")
        print(f"  → {'PASS' if passed_B else 'FAIL'}")
    
    # ─────────────────────────────────────────────────────────────
    # Test C: coupled loss — both branches in same loss expression,
    # privileged detached. Checks that even when privileged outputs
    # appear in the graph, their weights see no gradient.
    # ─────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n[adv-grad] Test C: coupled loss L(wl) + L(priv.detach()) "
              f"({num_steps} steps)")
    
    max_priv_grad_C = 0.0
    max_frozen_grad_C = 0.0
    worst_name_C = None
    
    forked_rdt.train()
    for step in range(num_steps):
        inp = make_inputs_fn()
        inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
               for k, v in inp.items()}
        
        _zero_grads_everywhere(forked_rdt)
        optimizer.zero_grad()
        
        wl_out = fwd(inp, auth=0)
        # Privileged forward WITHOUT no_grad, but we detach its output.
        # This is more aggressive than Test B: we allow the privileged
        # forward to build its own graph, then sever the connection
        # via detach. The graph still doesn't reach privileged weights
        # because they have requires_grad=False and the detach cuts
        # the activation path anyway — but this tests both safeguards
        # simultaneously.
        priv_out = fwd(inp, auth=1).detach()
        
        target = torch.randn_like(wl_out)
        loss = F.mse_loss(wl_out, target) + F.mse_loss(priv_out, target)
        # The priv term is a constant w.r.t. anything requiring grad,
        # so its gradient contribution should be exactly zero.
        loss.backward()
        
        m_priv, n_priv = _max_grad_magnitude(priv_params)
        m_frozen, n_frozen = _max_grad_magnitude(frozen_params)
        if m_priv > max_priv_grad_C:
            max_priv_grad_C = m_priv
            worst_name_C = n_priv
        if m_frozen > max_frozen_grad_C:
            max_frozen_grad_C = m_frozen
        
        optimizer.step()
    
    passed_C = max_priv_grad_C == 0.0 and max_frozen_grad_C == 0.0
    results["test_c"] = {
        "passed": passed_C,
        "max_priv_grad": max_priv_grad_C,
        "max_frozen_grad": max_frozen_grad_C,
        "worst_name": worst_name_C,
    }
    if verbose:
        print(f"  max privileged grad: {max_priv_grad_C:.2e}  (worst: {worst_name_C})")
        print(f"  max frozen grad:     {max_frozen_grad_C:.2e}")
        print(f"  → {'PASS' if passed_C else 'FAIL'}")
    
    results["passed"] = passed_A and passed_B and passed_C
    
    if verbose:
        print(f"\n[adv-grad] {'ALL TESTS PASSED' if results['passed'] else 'SOME TESTS FAILED'}")
    
    return results
