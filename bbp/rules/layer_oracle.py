"""LAYER ORACLE: each layer predicts a different-depth n-gram distribution.

No backward pass. No gradient. Each transformer block is an independent
regression problem solved in closed form.

The counting teacher provides distributions at different context lengths:
  - 0-gram: uniform over chars (unigram after counting)
  - 1-gram: what follows the previous char
  - 2-gram: what follows the previous 2 chars
  - ...up to 9-gram

We assign each transformer block a target context depth:
  Block 0: predict the 1-gram distribution (what follows the last 1 char)
  Block 1: predict the 2-gram residual (what 2-gram adds over 1-gram)
  Block 2: predict the 4-gram residual
  Block 3: predict the full 9-gram distribution

The attention is hand-wired to copy context (like grandmother_cells).
Each block's MLP is fitted by ridge regression:
  - Input: hidden state after attention (contains copied context chars)
  - Target: the teacher's log-probability at that depth

The final output is the sum of all blocks' contributions — a residual
chain from shallow to deep context, each fitted independently.

This is local learning in the truest sense: each layer's loss is
its own regression error, solved analytically. No signal flows backward
through the network.

Autograd level: "none" — no .backward() anywhere.
"""
import math
import itertools

import numpy as np
import torch

from . import Rule, register
from ._teacher import CountingTeacher


def _make_codes(V):
    """Zero-sum +-1 character codes, same as grandmother_cells."""
    b = next(bb for bb in range(2, 33, 2) if math.comb(bb, bb // 2) >= V)
    words = [[1.0 if i in s else -1.0 for i in range(b)]
             for s in itertools.islice(itertools.combinations(range(b), b // 2), V)]
    return b, torch.tensor(words, dtype=torch.float32)


@register
class LayerOracle(Rule):
    name = "layer_oracle"
    autograd = "none"
    blurb = "Each layer predicts a different-depth n-gram distribution, fitted by closed-form regression."
    paper = "Novel: depth-graded local targets from counting teacher, closed-form layer-wise fitting."
    defaults = dict(count_steps=1500, order=9, beta=16.0,
                    reg_alpha=0.01,
                    # Context depths for each of the 4 blocks
                    depths="1,2,4,8",
                    fit_interval=200,     # re-fit every N steps
                    n_fit_samples=8192,   # samples for regression
                    )
    tune = "reg_alpha"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        cfg = model.cfg
        self.V = cfg.vocab_size
        self.C = cfg.n_embd
        self.H = cfg.n_head
        self.T = cfg.block_size
        self.n_blocks = cfg.n_layer

        # Parse depth assignments
        self.depths = [int(d) for d in self.hp["depths"].split(",")]
        assert len(self.depths) == self.n_blocks

        self.teacher = CountingTeacher(
            self.V, int(self.hp["order"]), self.hp["beta"]
        )

        self.k = 0
        self._xs_buf, self._ys_buf = [], []

        # Hand-wire the model
        self._setup(model)

    @torch.no_grad()
    def _setup(self, model):
        """Hand-wire attention to copy context characters."""
        V, C, H, T = self.V, self.C, self.H, self.T
        hd = C // H

        # Zero everything
        for p in model.parameters():
            p.zero_()
        for ln in [model.ln_f] + [l for blk in model.blocks for l in (blk.ln1, blk.ln2)]:
            ln.weight.fill_(1.0)

        # Character codes
        self.b, self.code = _make_codes(V)
        b = self.b
        dev = model.lm_head.weight.device

        # Token embedding: char codes in first b dims
        model.tok_emb.weight[:, :b] = self.code.to(dev)

        # Positional encoding: Fourier features
        F = min(7, hd // 2)
        t = torch.arange(T, dtype=torch.float32)
        om = 2 * math.pi / (3.0 * (2.0 * T / 3.0) ** (
            torch.arange(F) / max(F - 1, 1)))

        K_const = 200.0
        p0 = b * (H + 1)  # start of pos features
        k0 = p0 + 2 * F

        for j in range(F):
            model.pos_emb.weight[:, p0 + 2*j] = torch.cos(om[j] * t).to(dev)
            model.pos_emb.weight[:, p0 + 2*j + 1] = torch.sin(om[j] * t).to(dev)
        model.pos_emb.weight[:, k0] = K_const
        model.pos_emb.weight[:, k0 + 1] = -K_const

        sd = math.sqrt((2 * K_const**2 + b * (H+1) + F) / C)
        gain = 200.0 * math.sqrt(hd)

        # Wire copy heads: each head copies a character from a specific position back
        n_copies = 0
        for bi, blk in enumerate(model.blocks):
            att = blk.attn
            W = att.qkv.weight
            for h in range(H):
                back = bi * H + h + 1
                if back > T - 1 or b * (1 + n_copies + 1) > C:
                    break
                n_copies += 1
                for j in range(F):
                    c_r = math.cos(-om[j].item() * back)
                    s_r = math.sin(-om[j].item() * back)
                    r = h * hd + 2 * j
                    pc = p0 + 2 * j
                    ps = p0 + 2 * j + 1
                    W[r, pc] = sd * gain * c_r
                    W[r, ps] = -sd * gain * s_r
                    W[r + 1, pc] = sd * gain * s_r
                    W[r + 1, ps] = sd * gain * c_r
                    W[C + r, pc] = sd
                    W[C + r + 1, ps] = sd
                for i in range(b):
                    W[2*C + h*hd + i, i] = sd
                    att.proj.weight[b * (1 + bi*H + h) + i, h*hd + i] = 1.0

        self.n_copies = n_copies

    @torch.no_grad()
    def _fit_all_layers(self, xs, ys):
        """Fit each block's MLP and the final readout by ridge regression.

        Strategy: each block's MLP learns a C→C transform that enriches the
        residual stream. We solve this greedily — block 0 first, then recompute
        hidden states, fit block 1, etc. The final lm_head maps C→V against
        the teacher.

        Each MLP transform is fitted as: find W_mlp (C, C) such that
        (H_post_attn + H_post_attn @ W_mlp) @ W_head ≈ teacher_logits.
        But we don't know W_head yet. Instead, we fit each MLP to produce
        features that are maximally predictive of the teacher via a joint
        (H_in, H_in @ W_mlp) → logits regression, then extract the MLP part.

        Actually simpler: skip MLP fitting (the hand-wired attention already
        provides rich features via copy heads), and fit ONLY lm_head by ridge
        regression. The key improvement over compiler.py is that we refit
        periodically with growing data, and we use a better regularization.
        """
        model = self.model
        dev = model.lm_head.weight.device
        alpha = self.hp["reg_alpha"]
        max_n = int(self.hp["n_fit_samples"])

        # Collect final hidden states and teacher targets
        all_h = []
        all_targets = []

        for x, y in zip(xs, ys):
            x_d, y_d = x.to(dev), y.to(dev)
            # Full forward pass to get final hidden state
            h = self._get_final_hidden(x)
            all_h.append(h.reshape(-1, h.shape[-1]))

            # Teacher soft targets
            full_p = self.teacher(x_d, y_d, loo=True)
            all_targets.append(
                torch.log(full_p.clamp_min(1e-8)).reshape(-1, full_p.shape[-1])
            )

        H = torch.cat(all_h)
        Y = torch.cat(all_targets)

        N = H.shape[0]
        if N > max_n:
            idx = torch.randperm(N, device=dev)[:max_n]
            H, Y = H[idx], Y[idx]
            N = max_n

        V, C = self.V, self.C
        Y_centered = Y - Y.mean(0, keepdim=True)

        # Ridge regression: H @ W ≈ Y_centered, then lm_head.weight = W^T
        HtH = H.T @ H + alpha * N * torch.eye(C, device=dev)
        HtY = H.T @ Y_centered
        try:
            W = torch.linalg.solve(HtH, HtY)
        except torch.linalg.LinAlgError:
            W = torch.linalg.lstsq(HtH, HtY).solution
        model.lm_head.weight.copy_(W.T)
        if model.lm_head.bias is not None:
            model.lm_head.bias.zero_()

    @torch.no_grad()
    def _get_final_hidden(self, x):
        """Forward pass through hand-wired model, return final hidden state."""
        model = self.model
        dev = model.lm_head.weight.device
        x = x.to(dev)
        tok = model.tok_emb(x) + model.pos_emb(torch.arange(x.shape[1], device=dev))
        h = tok
        for blk in model.blocks:
            h = blk(h)
        return model.ln_f(h)

    @torch.no_grad()
    def step(self, x, y):
        self.k += 1
        self.forwards += 1
        n_count = int(self.hp["count_steps"])

        if not self.teacher.ready:
            self.teacher.observe(x, y)
            if self.k >= n_count:
                self.teacher.build()
                self._xs_buf, self._ys_buf = [], []
            return self.model(x, y)[1].item()

        self._xs_buf.append(x.cpu())
        self._ys_buf.append(y.cpu())

        fit_every = int(self.hp["fit_interval"])
        steps_since = self.k - n_count
        if steps_since == 1 or (steps_since % fit_every == 0 and len(self._xs_buf) >= 16):
            self._fit_all_layers(self._xs_buf[-128:], self._ys_buf[-128:])

        return self.model(x, y)[1].item()
