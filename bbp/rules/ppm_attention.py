"""PPM ATTENTION: Prediction by Partial Matching via attention + grandmother cells.

grandmother_cells gets 1.899 by compiling the top 2048 n-gram patterns into
MLP AND-gate neurons. But it misses patterns beyond its budget. This rule
keeps the MLP cells AND wires the attention in blocks 1-3 as a context-window
n-gram lookup table:

  Q/K: match on copied-char dims (previous 1-4 chars)
  V:   read the current char code (the char that FOLLOWS the matching context)
  proj: decode char code to logit-slot increments via code^T

The attention computes: for each position t, find past positions t' < t where
the previous n chars match, then average what char follows there. This is
in-context Prediction by Partial Matching, a nearest-neighbor n-gram model
using the current 128-char window.

The MLP provides globally-learned statistics (from millions of chars).
The attention provides local context-window statistics.
They SUM in the logit slots.

Key insight: Q/K read from COPIED-char dims (chars t-1, t-2, etc., set by
block 0's copy heads), while V reads the CURRENT char at t' (dim 0:b).
So when contexts match (same previous n chars), V gives the char that
followed that context — exactly the n-gram prediction.

Autograd level: "none" — no .backward() anywhere.
"""
import math

import numpy as np
import torch

from . import Rule, register
from .grandmother_cells import CellModel, count_strings, hire


@register
class PPMAttention(Rule):
    name = "ppm_attention"
    autograd = "none"
    blurb = "Grandmother cells + attention-based context-window n-gram lookup."
    paper = "Novel: PPM (Cleary & Witten 1984) implemented via transformer attention Q/K matching, combined with hand-wired AND-gate MLP neurons."
    defaults = dict(
        kappa=4.0,
        count_steps=2000,
        max_len=9,
        attn_gain=100.0,    # sharpness of Q/K matching (higher = more exact)
        attn_scale=1.0,     # logit contribution per head for a perfect match
        n_attn_blocks=3,    # how many blocks (1-3) get context-matching heads
    )
    tune = "attn_scale"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.cm = CellModel(model)
        self.N = min(self.cm.N, int(self.hp["max_len"]))
        self.xs, self.ys, self.k, self.cells = [], [], 0, []
        self._wire_context_heads(model)

    @torch.no_grad()
    def _wire_context_heads(self, model):
        """Wire attention in blocks 1+ for context matching.

        For each head h in blocks 1-3:
          Q reads copied-char dims for context of length (h+1)
          K reads the same dims
          V reads current char code (dims 0:b)
          proj writes to logit-slot dims via code^T decoding

        Scaling:
          Q includes sd*gain (cancel LayerNorm + sharp matching)
          K includes sd (cancel LayerNorm)
          V includes sd (cancel LayerNorm)
          proj includes attn_scale/(U*sd*b) so final logit = attn_scale per match
        """
        cm = self.cm
        b, V, C, H = cm.b, cm.V, cm.C, cm.H
        hd = cm.hd
        sd = cm.sd
        dev = cm.dev
        code = cm.code  # (V, b)
        l0 = cm.l0
        U = cm.U
        np_ = cm.np_  # number of copy positions (chars t-1..t-np_)

        attn_gain = self.hp["attn_gain"]
        attn_scale = self.hp["attn_scale"]
        n_blocks = min(int(self.hp["n_attn_blocks"]), 3)

        # proj_coeff: chosen so that lm_head amplification gives exactly attn_scale
        # lm_head[c, l0+c] = U * sd.  Attention writes proj_coeff * code to logit slots.
        # logit[c] = U*sd * proj_coeff * code[c] . V_output
        # For perfect match: V_output = code(x[t']), code[c].code(x[t']) = b
        # logit = U*sd * proj_coeff * b = attn_scale
        # => proj_coeff = attn_scale / (U * sd * b)
        proj_coeff = attn_scale / (U * sd * b)

        for bi in range(1, 1 + n_blocks):
            if bi >= len(model.blocks):
                break
            attn = model.blocks[bi].attn
            W = attn.qkv.weight  # (3*C, C)

            for h in range(H):
                # Context length for this head: 1-gram through min(H, np_)-gram
                n_ctx = min(h + 1, np_)
                if n_ctx == 0:
                    continue

                # Q projection: read copied-char dims for chars t-1 to t-n_ctx
                q_start = h * hd
                for ci in range(n_ctx):
                    src_start = b * (ci + 1)  # char at t-(ci+1)
                    dst_start = b * ci        # within head's hd dims
                    for j in range(b):
                        W[q_start + dst_start + j, src_start + j] = sd * attn_gain

                # K projection: same dims, sd only (gain goes in Q)
                k_start = C + h * hd
                for ci in range(n_ctx):
                    src_start = b * (ci + 1)
                    dst_start = b * ci
                    for j in range(b):
                        W[k_start + dst_start + j, src_start + j] = sd

                # V projection: read current char code (dims 0:b)
                v_start = 2 * C + h * hd
                for j in range(b):
                    W[v_start + j, j] = sd  # cancel LN

            # Output projection: decode char codes to logit-slot predictions
            for h in range(H):
                n_ctx = min(h + 1, np_)
                if n_ctx == 0:
                    continue
                for c in range(V):
                    for j in range(b):
                        attn.proj.weight[l0 + c, h * hd + j] = (
                            proj_coeff * code[c, j].item()
                        )

    def compile(self):
        V = self.cm.V
        t = count_strings(np.concatenate(self.xs), np.concatenate(self.ys), V, self.N)
        self.cells, outs, base = hire(t, V, self.N, self.hp["kappa"], self.cm.n_cells)
        self.cm.keys(self.cells, outs, base)

    @torch.no_grad()
    def step(self, x, y):
        self.k += 1
        n = int(self.hp["count_steps"])
        if self.k <= n:
            self.xs.append(x.cpu().numpy())
            self.ys.append(y.cpu().numpy())
            if self.k in (1, n // 10, n // 3, n):
                self.compile()
            if self.k == n:
                self.xs, self.ys = [], []
        self.forwards += 1
        return self.model(x, y)[1].item()
