"""GC+: grandmother_cells + orthogonal lm_head correction.

grandmother_cells wires 2048 AND-gate neurons that write to 65 logit-slot
dimensions. lm_head reads those with a fixed identity. But the hidden state
has 63 OTHER dimensions (char codes, position, K pair) that go unused.

This rule does grandmother_cells first, then adds an ORTHOGONAL correction
to lm_head: ridge regression from ONLY the non-logit-slot dims to the
teacher's residuals. The logit-slot readout is preserved EXACTLY.

The correction is a linear decoder: for each pair (context chars, target char),
it adds a small logit adjustment. This captures marginal bigram/trigram signal
that didn't make the top 2048 AND-gates.

Previous attempts (compiler, boosted_cells) failed because they replaced the
ENTIRE lm_head, disturbing the logit-slot readout. This version is orthogonal:
it adds to lm_head without touching logit-slot weights.

Autograd level: "none" — no .backward() anywhere.
"""
import math

import numpy as np
import torch

from . import Rule, register
from ._teacher import CountingTeacher
from .grandmother_cells import CellModel, count_strings, hire


@register
class GCPlus(Rule):
    name = "gc_plus"
    autograd = "none"
    blurb = "Grandmother cells + orthogonal lm_head correction from char-code dims."
    paper = "Novel: hand-wired n-gram base + orthogonal closed-form readout correction."
    defaults = dict(
        kappa=8.0,         # best from HP sweep
        count_steps=2500,  # best from HP sweep
        max_len=9,
        reg_alpha=1.0,     # L2 regularization for correction
        corr_steps=200,    # re-fit correction every N steps
        n_samples=16384,   # samples for regression
    )
    tune = "reg_alpha"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.cm = CellModel(model)
        self.N = min(self.cm.N, int(self.hp["max_len"]))
        self.teacher = CountingTeacher(
            model.cfg.vocab_size,
            9,  # order
            16.0,  # beta
        )
        self.xs, self.ys = [], []
        self.k = 0
        self.cells_wired = False
        self._base_lm_head = None
        self._xs_buf, self._ys_buf = [], []

    @torch.no_grad()
    def _wire_cells(self):
        V = self.cm.V
        t = count_strings(np.concatenate(self.xs), np.concatenate(self.ys), V, self.N)
        self.cells, outs, base = hire(t, V, self.N, self.hp["kappa"], self.cm.n_cells)
        self.cm.keys(self.cells, outs, base)
        self.cells_wired = True

    @torch.no_grad()
    def _save_base(self):
        """Save grandmother_cells' lm_head for restoration."""
        self._base_lm_head = self.model.lm_head.weight.clone()

    @torch.no_grad()
    def _get_final_hidden(self, x):
        """Forward pass, return final hidden state (after ln_f)."""
        model = self.model
        dev = model.lm_head.weight.device
        x = x.to(dev)
        h = model.tok_emb(x) + model.pos_emb(torch.arange(x.shape[1], device=dev))
        for blk in model.blocks:
            h = blk(h)
        return model.ln_f(h)

    @torch.no_grad()
    def _fit_correction(self, xs, ys):
        """Fit an ORTHOGONAL correction to lm_head.

        Reads ONLY non-logit-slot dims of the hidden state.
        Fits: H_non_logit @ W_corr ≈ teacher_residuals
        Adds W_corr^T to lm_head without touching logit-slot columns.
        """
        model = self.model
        dev = model.lm_head.weight.device
        alpha = self.hp["reg_alpha"]
        max_n = int(self.hp["n_samples"])
        V = self.cm.V
        C = self.cm.C
        l0 = self.cm.l0

        # Restore base lm_head (undo any previous correction)
        model.lm_head.weight.copy_(self._base_lm_head)

        # Collect hidden states and residuals
        all_h = []
        all_residuals = []

        for x, y in zip(xs, ys):
            x_d, y_d = x.to(dev), y.to(dev)

            h = self._get_final_hidden(x)
            logits = model.lm_head(h)

            # Teacher targets
            p_teacher = self.teacher(x_d, y_d, loo=True)
            teacher_logits = torch.log(p_teacher.clamp_min(1e-8))

            # Model's current log-probs
            model_logprobs = logits.log_softmax(-1)

            # Residual: what the base model gets wrong
            residual = teacher_logits - model_logprobs

            all_h.append(h.reshape(-1, C))
            all_residuals.append(residual.reshape(-1, V))

        H = torch.cat(all_h)
        R = torch.cat(all_residuals)

        N = H.shape[0]
        if N > max_n:
            idx = torch.randperm(N, device=dev)[:max_n]
            H, R = H[idx], R[idx]
            N = max_n

        # Extract ONLY non-logit-slot dims
        # Logit slots are at l0:l0+V. Everything else is non-logit.
        non_logit_mask = torch.ones(C, dtype=torch.bool, device=dev)
        non_logit_mask[l0:l0 + V] = False
        non_logit_dims = non_logit_mask.nonzero(as_tuple=True)[0]
        n_dims = non_logit_dims.shape[0]

        H_nl = H[:, non_logit_dims]  # (N, n_dims)

        # Center residuals
        R_centered = R - R.mean(0, keepdim=True)

        # Ridge regression: H_nl @ W ≈ R_centered
        HtH = H_nl.T @ H_nl + alpha * N * torch.eye(n_dims, device=dev)
        HtR = H_nl.T @ R_centered
        try:
            W = torch.linalg.solve(HtH, HtR)  # (n_dims, V)
        except torch.linalg.LinAlgError:
            W = torch.linalg.lstsq(HtH, HtR).solution

        # Add correction to lm_head (non-logit-slot columns only)
        # lm_head.weight is (V, C). We add W^T to the non-logit-slot columns.
        for i, dim_idx in enumerate(non_logit_dims.tolist()):
            model.lm_head.weight[:, dim_idx] += W[i, :]

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
                self._save_base()
                self.xs, self.ys = [], []
                self._xs_buf, self._ys_buf = [], []
            elif self.k in (1, n_count // 10, n_count // 3):
                self._wire_cells()
            return self.model(x, y)[1].item()

        # Phase 2: accumulate data and periodically fit correction
        self._xs_buf.append(x.cpu())
        self._ys_buf.append(y.cpu())

        corr_every = int(self.hp["corr_steps"])
        steps_since = self.k - n_count
        if steps_since == 1 or (steps_since % corr_every == 0 and len(self._xs_buf) >= 16):
            self._fit_correction(self._xs_buf[-128:], self._ys_buf[-128:])

        return self.model(x, y)[1].item()
