"""
verification_harness.py — Phase 2

Registers forward hooks on all privileged and whitelisted cross-attention
modules in the forked blocks. Hooks count the number of times each module
is actually executed.

Contract we verify:
  - auth=0 (whitelisted) forward:
      privileged cross-attention: 0 calls, per block
      whitelisted cross-attention: 1 call, per block
  - auth=1 (privileged) forward:
      privileged cross-attention: 1 call, per block
      whitelisted cross-attention: 0 calls, per block
  - Forked FinalLayer: same contract.

Why this matters:
  The paper claims "the non-selected branch's cross-attention is not
  executed." That claim has to be executable. A forward hook fires when
  and only when the module's forward() method runs. Zero hook calls on
  the non-selected branch proves the isolation holds at runtime, not just
  at weight level.
"""

import torch


class RouteHookHarness:
    """Attach counters to every forked cross-attn and final-layer module.
    
    Usage:
        harness = RouteHookHarness(forked_rdt)
        harness.attach()
        # ... run forwards ...
        counts = harness.counts()
        harness.reset()       # zero counters between tests
        harness.detach()      # remove hooks when done
    """
    
    def __init__(self, forked_rdt):
        # forked_rdt is a CrossAttnForkedRDT (unwrapped). If user passes
        # the AuthRoutedRDT wrapper, reach through to the underlying model.
        if hasattr(forked_rdt, "model") and hasattr(forked_rdt.model, "forked_blocks"):
            forked_rdt = forked_rdt.model
        
        self.model = forked_rdt
        self._handles = []
        self._counts = {}
        self._registered = False
    
    def _make_hook(self, key):
        def hook(module, inputs, output):
            self._counts[key] = self._counts.get(key, 0) + 1
        return hook
    
    def attach(self):
        if self._registered:
            return
        
        for local_i, block in enumerate(self.model.forked_blocks):
            gi = self.model.split_idx + local_i
            
            priv_key = f"block{gi}.priv_cross_attn"
            wl_key = f"block{gi}.wl_cross_attn"
            self._counts[priv_key] = 0
            self._counts[wl_key] = 0
            
            self._handles.append(
                block.priv_cross_attn.register_forward_hook(self._make_hook(priv_key))
            )
            self._handles.append(
                block.wl_cross_attn.register_forward_hook(self._make_hook(wl_key))
            )
        
        # Forked final layer
        if self.model.fork_final:
            final = self.model.forked_final
            self._counts["final.privileged"] = 0
            self._counts["final.whitelisted"] = 0
            self._handles.append(
                final.privileged.register_forward_hook(
                    self._make_hook("final.privileged"))
            )
            self._handles.append(
                final.whitelisted.register_forward_hook(
                    self._make_hook("final.whitelisted"))
            )
        
        self._registered = True
    
    def reset(self):
        for k in self._counts:
            self._counts[k] = 0
    
    def counts(self):
        return dict(self._counts)
    
    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._registered = False
    
    # Context manager sugar
    def __enter__(self):
        self.attach()
        return self
    
    def __exit__(self, *args):
        self.detach()


def verify_routing_contract(forked_rdt, make_inputs_fn, num_passes=1000,
                            device="cuda", verbose=True):
    """Run `num_passes` forwards under each auth setting and verify hook counts.
    
    Args:
        forked_rdt: either CrossAttnForkedRDT or AuthRoutedRDT. If wrapper,
            we call with auth=..., otherwise with head=... (string).
        make_inputs_fn: zero-arg callable returning a dict with keys
            x, freq, t, lang_c, img_c — the positional inputs to forward().
            Called fresh each iteration so input tensors don't alias.
        num_passes: how many forwards to run per auth setting.
        device: device for inputs.
        verbose: print progress.
    
    Returns:
        dict with keys:
          passed: bool
          errors: list of assertion-failure strings (empty if passed)
          counts_auth0: hook counts after auth=0 phase
          counts_auth1: hook counts after auth=1 phase
          num_forked_blocks: int
    """
    # Detect wrapper vs raw model
    is_wrapper = hasattr(forked_rdt, "model") and hasattr(forked_rdt.model, "forked_blocks")
    
    raw = forked_rdt.model if is_wrapper else forked_rdt
    num_forked = raw.num_forked_blocks
    
    harness = RouteHookHarness(raw)
    errors = []
    
    try:
        harness.attach()
        forked_rdt.eval()
        
        # Phase A: auth=0 (whitelisted) for num_passes
        if verbose:
            print(f"[verify] running {num_passes} forwards with auth=0 (whitelisted)...")
        harness.reset()
        
        with torch.no_grad():
            for i in range(num_passes):
                inp = make_inputs_fn()
                inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                       for k, v in inp.items()}
                if is_wrapper:
                    _ = forked_rdt(**inp, auth=0)
                else:
                    _ = forked_rdt(**inp, head="whitelisted")
        
        counts_auth0 = harness.counts()
        
        # Verify auth=0 contract
        for local_i, block in enumerate(raw.forked_blocks):
            gi = raw.split_idx + local_i
            priv_key = f"block{gi}.priv_cross_attn"
            wl_key = f"block{gi}.wl_cross_attn"
            
            if counts_auth0[priv_key] != 0:
                errors.append(
                    f"[auth=0] privileged {priv_key} called "
                    f"{counts_auth0[priv_key]} times, expected 0"
                )
            if counts_auth0[wl_key] != num_passes:
                errors.append(
                    f"[auth=0] whitelisted {wl_key} called "
                    f"{counts_auth0[wl_key]} times, expected {num_passes}"
                )
        
        if raw.fork_final:
            if counts_auth0["final.privileged"] != 0:
                errors.append(
                    f"[auth=0] final.privileged called "
                    f"{counts_auth0['final.privileged']} times, expected 0"
                )
            if counts_auth0["final.whitelisted"] != num_passes:
                errors.append(
                    f"[auth=0] final.whitelisted called "
                    f"{counts_auth0['final.whitelisted']} times, "
                    f"expected {num_passes}"
                )
        
        # Phase B: auth=1 (privileged) for num_passes
        if verbose:
            print(f"[verify] running {num_passes} forwards with auth=1 (privileged)...")
        harness.reset()
        
        with torch.no_grad():
            for i in range(num_passes):
                inp = make_inputs_fn()
                inp = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                       for k, v in inp.items()}
                if is_wrapper:
                    _ = forked_rdt(**inp, auth=1)
                else:
                    _ = forked_rdt(**inp, head="privileged")
        
        counts_auth1 = harness.counts()
        
        # Verify auth=1 contract
        for local_i, block in enumerate(raw.forked_blocks):
            gi = raw.split_idx + local_i
            priv_key = f"block{gi}.priv_cross_attn"
            wl_key = f"block{gi}.wl_cross_attn"
            
            if counts_auth1[priv_key] != num_passes:
                errors.append(
                    f"[auth=1] privileged {priv_key} called "
                    f"{counts_auth1[priv_key]} times, expected {num_passes}"
                )
            if counts_auth1[wl_key] != 0:
                errors.append(
                    f"[auth=1] whitelisted {wl_key} called "
                    f"{counts_auth1[wl_key]} times, expected 0"
                )
        
        if raw.fork_final:
            if counts_auth1["final.privileged"] != num_passes:
                errors.append(
                    f"[auth=1] final.privileged called "
                    f"{counts_auth1['final.privileged']} times, "
                    f"expected {num_passes}"
                )
            if counts_auth1["final.whitelisted"] != 0:
                errors.append(
                    f"[auth=1] final.whitelisted called "
                    f"{counts_auth1['final.whitelisted']} times, expected 0"
                )
    
    finally:
        harness.detach()
    
    passed = len(errors) == 0
    
    if verbose:
        if passed:
            print(f"[verify] PASSED: {num_forked} forked blocks, "
                  f"{num_passes} passes each, zero leakage")
        else:
            print(f"[verify] FAILED with {len(errors)} errors:")
            for e in errors[:10]:
                print(f"  - {e}")
    
    return {
        "passed": passed,
        "errors": errors,
        "counts_auth0": counts_auth0,
        "counts_auth1": counts_auth1,
        "num_forked_blocks": num_forked,
        "num_passes": num_passes,
    }
