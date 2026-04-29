"""
auth_routing.py — Phase 1

Wraps the existing CrossAttnForkedRDT with an auth-scalar interface.

Design:
  - auth=0  → whitelisted route (the always-on default, public policy)
  - auth=1  → privileged route  (gated, task-capable policy)
  - auth=None → defaults to whitelisted (asymmetric framing: public is default)

The wrapper threads the scalar through to the existing head="..." string
interface. We keep the internal representation as a string because the
ForkedRDTBlock.forward signature already uses it; the external interface
is now a scalar, matching what System 2 would actually produce.

IMPORTANT: the route selection is a Python `if` on the scalar at the block
level, so the non-selected cross-attention module is genuinely not called.
No multiplicative mask, no wasted FLOPs.
"""

import torch
import torch.nn as nn


AUTH_WHITELISTED = 0
AUTH_PRIVILEGED = 1


def auth_to_head(auth):
    """Map auth scalar to the string head name the underlying model expects.
    
    auth can be:
      - None           → whitelisted (default)
      - 0 / False      → whitelisted
      - 1 / True       → privileged
      - 0-d tensor     → same as above via .item()
    
    We reject batched auth because routing is per-forward-pass, not per-sample.
    A batched auth would mean running both branches and gathering, which
    defeats the point of the architectural isolation.
    """
    if auth is None:
        return "whitelisted"
    
    if isinstance(auth, torch.Tensor):
        if auth.numel() != 1:
            raise ValueError(
                f"auth must be a scalar (per forward pass), got shape {auth.shape}. "
                "If you need per-sample routing, run two separate forwards and "
                "concatenate outputs."
            )
        auth = auth.item()
    
    if auth in (0, False):
        return "whitelisted"
    if auth in (1, True):
        return "privileged"
    
    raise ValueError(f"auth must be 0 or 1 (or None), got {auth!r}")


class AuthRoutedRDT(nn.Module):
    """Thin wrapper exposing an `auth` scalar interface over CrossAttnForkedRDT.
    
    This is the interface that System 2 would call after making its
    authentication decision. The auth scalar is the downstream boolean
    from whatever upstream mechanism (ECAPA voice, FaceID, etc.) decided
    the operator is verified.
    """
    
    def __init__(self, forked_rdt):
        super().__init__()
        self.model = forked_rdt
        
        # Expose underlying attributes so the wrapper is a drop-in replacement.
        self.horizon = forked_rdt.horizon
        self.hidden_size = forked_rdt.hidden_size
        self.num_forked_blocks = forked_rdt.num_forked_blocks
    
    def forward(self, x, freq, t, lang_c, img_c,
                auth=None, lang_mask=None, img_mask=None):
        head = auth_to_head(auth)
        return self.model(
            x, freq, t, lang_c, img_c,
            head=head,
            lang_mask=lang_mask, img_mask=img_mask,
        )
    
    # Pass-through for named_parameters, parameters, etc.
    # nn.Module handles these via child module registration already,
    # since self.model is a submodule.
