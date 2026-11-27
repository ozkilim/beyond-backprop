"""COMPILER: grandmother_cells + ridge-regressed readout against counting oracle.

grandmother_cells gets 1.900 by compiling n-gram patterns into MLP neurons.
The bottleneck: lm_head is hand-wired to read logit slots with a fixed identity
mapping. But the hidden states contain more information than those slots capture.

This rule does grandmother_cells first, then fits lm_head by ridge regression
against the counting teacher's soft targets. The regression is optimal for the
last layer given the hidden representation — no gradient needed, just linear
algebra on (128×128) matrices.

Autograd level: "none" — no .backward() anywhere.
"""
import math

import numpy as np
import torch

from . import Rule, register
from ._teacher import CountingTeacher
from .grandmother_cells import CellModel, count_strings, hire


@register
class Compiler(Rule):
    name = "compiler"
    autograd = "none"
    blurb = "Grandmother cells + ridge-regressed readout against counting oracle."
    paper = "Novel: hand-wired n-gram base + closed-form readout distilled from counting table."
    defaults = dict(count_steps=1500, order=9, beta=16.0,
                    regress_steps=100,    # how often to re-fit the readout
                    reg_alpha=0.1,        # L2 regularisation for regression
                    kappa=4.0, max_len=9)
    tune = "reg_alpha"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.cm = CellModel(model)
        self.N = min(self.cm.N, int(self.hp.get("max_len", 9)))
        self.teacher = CountingTeacher(
            model.cfg.vocab_size,
            int(self.hp["order"]),
            self.hp["beta"],
        )
        self.xs, self.ys = [], []
        self.k = 0
        self.cells_wired = False

    @torch.no_grad()
    def _wire_cells(self):
        V = self.cm.V
        t = count_strings(np.concatenate(self.xs), np.concatenate(self.ys), V, self.N)
        cells, outs, base = hire(t, V, self.N, self.hp["kappa"], self.cm.n_cells)
        self.cm.keys(cells, outs, base)
        self.cells_wired = True

    @torch.no_grad()
    def _collect_hidden_states(self, x):
        """Run forward pass, extract hidden states before lm_head."""
        model = self.model
        dev = model.lm_head.weight.device
        tok = model.tok_emb(x.to(dev))
        pos = model.pos_emb(torch.arange(x.shape[1], device=dev))
        h = tok + pos
        for blk in model.blocks:
            h = blk(h)
        h = model.ln_f(h)
        return h  # (B, T, C)

    @torch.no_grad()
    def _fit_readout(self, xs, ys):
        """Fit lm_head by ridge regression against teacher log-probs."""
        model = self.model
        dev = model.lm_head.weight.device

        all_h, all_targets = [], []
        for x, y in zip(xs, ys):
            x_d, y_d = x.to(dev), y.to(dev)
            h = self._collect_hidden_states(x)
            p = self.teacher(x_d, y_d, loo=True)
            targets = torch.log(p.clamp_min(1e-8))
            all_h.append(h.reshape(-1, h.shape[-1]))
            all_targets.append(targets.reshape(-1, targets.shape[-1]))

        H = torch.cat(all_h)       # (N, C)
        Y = torch.cat(all_targets)  # (N, V)

        N = H.shape[0]
        max_n = 16384
        if N > max_n:
            idx = torch.randperm(N, device=dev)[:max_n]
            H, Y = H[idx], Y[idx]
            N = max_n

        Y = Y - Y.mean(0, keepdim=True)
        alpha = self.hp["reg_alpha"]
        C = H.shape[1]

        HtH = H.T @ H + alpha * N * torch.eye(C, device=dev)
        HtY = H.T @ Y
        try:
            W = torch.linalg.solve(HtH, HtY)
        except torch.linalg.LinAlgError:
            W = torch.linalg.lstsq(HtH, HtY).solution

        model.lm_head.weight.copy_(W.T)
        if model.lm_head.bias is not None:
            model.lm_head.bias.zero_()

    @torch.no_grad()
    def step(self, x, y):
        self.k += 1
        self.forwards += 1
        n_count = int(self.hp["count_steps"])

        if not self.teacher.ready:
            self.xs.append(x.cpu().numpy())
            self.ys.append(y.cpu().numpy())
            self.teacher.observe(x, y)
            if self.k >= n_count:
                self.teacher.build()
                self._wire_cells()
                self.xs = []
                self.ys = []
                self._xs_buf = []
                self._ys_buf = []
            elif self.k in (1, n_count // 10, n_count // 3):
                self._wire_cells()
            return self.model(x, y)[1].item()

        # Phase 2: collect and periodically re-fit readout
        self._xs_buf.append(x.cpu())
        self._ys_buf.append(y.cpu())

        regress_every = int(self.hp["regress_steps"])
        steps_since = self.k - n_count
        if steps_since == 1 or (steps_since % regress_every == 0 and len(self._xs_buf) >= 8):
            self._fit_readout(self._xs_buf[-128:], self._ys_buf[-128:])

        return self.model(x, y)[1].item()
