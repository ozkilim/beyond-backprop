"""BOOSTED CELLS: grandmother_cells + ridge-regressed residual correction.

grandmother_cells gets 1.899 by compiling the top 2048 n-gram patterns into
MLP AND-gate neurons. But it misses patterns that didn't make the cut, and
its readout (lm_head) only reads the 65 logit-slot dimensions, ignoring the
other 63 dims of the hidden state which contain raw context features.

This rule does grandmother_cells first, then computes residuals:
  residual = teacher_log_probs - model_log_probs

and fits a small correction via ridge regression from the FULL hidden state
to these residuals. The correction is added to lm_head with a small step size.

This is boosting without gradients: the base model (grandmother_cells) handles
the top patterns exactly, and the correction captures signal from unused
features and unrepresented patterns.

The key insight: lm_head starts as a scaled identity on logit slots (the
grandmother_cells readout). Ridge regression finds a correction that uses
the char-code and position-feature dims to improve predictions. Heavy
regularization ensures the base model's good predictions aren't disrupted.

Autograd level: "none" — no .backward() anywhere.
"""
import math

import numpy as np
import torch

from . import Rule, register
from ._teacher import CountingTeacher
from .grandmother_cells import CellModel, count_strings, hire


@register
class BoostedCells(Rule):
    name = "boosted_cells"
    autograd = "none"
    blurb = "Grandmother cells + ridge-regressed residual correction from full hidden state."
    paper = "Novel: forward-only boosting — exact n-gram base + closed-form residual correction."
    defaults = dict(count_steps=1500, order=9, beta=16.0, kappa=4.0, max_len=9,
                    boost_lr=0.3,         # how much of the correction to apply
                    reg_alpha=1.0,        # L2 for regression (high = safe, preserves base)
                    boost_interval=200,   # re-fit every N steps
                    n_samples=8192,       # samples for regression
                    )
    tune = "boost_lr"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.cm = CellModel(model)
        self.N = min(self.cm.N, int(self.hp["max_len"]))
        self.teacher = CountingTeacher(
            model.cfg.vocab_size,
            int(self.hp["order"]),
            self.hp["beta"],
        )
        self.xs, self.ys = [], []
        self.k = 0
        self.cells_wired = False
        self._base_lm_head = None  # save grandmother_cells' original readout

    @torch.no_grad()
    def _wire_cells(self):
        V = self.cm.V
        t = count_strings(np.concatenate(self.xs), np.concatenate(self.ys), V, self.N)
        cells, outs, base = hire(t, V, self.N, self.hp["kappa"], self.cm.n_cells)
        self.cm.keys(cells, outs, base)
        self.cells_wired = True

    @torch.no_grad()
    def _save_base(self):
        """Save the grandmother_cells lm_head as the base."""
        self._base_lm_head = self.model.lm_head.weight.clone()

    @torch.no_grad()
    def _get_hidden_and_logits(self, x):
        """Forward pass, return final hidden state and logits."""
        model = self.model
        dev = model.lm_head.weight.device
        x = x.to(dev)
        tok = model.tok_emb(x) + model.pos_emb(torch.arange(x.shape[1], device=dev))
        h = tok
        for blk in model.blocks:
            h = blk(h)
        h = model.ln_f(h)
        logits = model.lm_head(h)
        return h, logits

    @torch.no_grad()
    def _boost_step(self, xs, ys):
        """Fit a correction to lm_head using ridge regression on residuals."""
        model = self.model
        dev = model.lm_head.weight.device
        alpha = self.hp["reg_alpha"]
        lr = self.hp["boost_lr"]
        max_n = int(self.hp["n_samples"])

        # First, restore the base lm_head (undo any previous correction)
        model.lm_head.weight.copy_(self._base_lm_head)

        all_h, all_residuals = [], []

        for x, y in zip(xs, ys):
            x_d, y_d = x.to(dev), y.to(dev)
            h, logits = self._get_hidden_and_logits(x_d)

            # Teacher targets
            p_teacher = self.teacher(x_d, y_d, loo=True)
            teacher_logits = torch.log(p_teacher.clamp_min(1e-8))

            # Model's current log-probs
            model_logprobs = logits.log_softmax(-1)

            # Residual: what the base model gets wrong
            residual = teacher_logits - model_logprobs  # (B, T, V)

            all_h.append(h.reshape(-1, h.shape[-1]))
            all_residuals.append(residual.reshape(-1, residual.shape[-1]))

        H = torch.cat(all_h)       # (N, C)
        R = torch.cat(all_residuals)  # (N, V)

        N = H.shape[0]
        if N > max_n:
            idx = torch.randperm(N, device=dev)[:max_n]
            H, R = H[idx], R[idx]
            N = max_n

        # Center the residuals
        R = R - R.mean(0, keepdim=True)

        C = H.shape[1]
        HtH = H.T @ H + alpha * N * torch.eye(C, device=dev)
        HtR = H.T @ R
        try:
            W_corr = torch.linalg.solve(HtH, HtR)  # (C, V)
        except torch.linalg.LinAlgError:
            W_corr = torch.linalg.lstsq(HtH, HtR).solution

        # Apply correction with learning rate
        model.lm_head.weight.add_(lr * W_corr.T)

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

        # Phase 2: accumulate data and periodically boost
        self._xs_buf.append(x.cpu())
        self._ys_buf.append(y.cpu())

        boost_every = int(self.hp["boost_interval"])
        steps_since = self.k - n_count
        if steps_since == 1 or (steps_since % boost_every == 0 and len(self._xs_buf) >= 16):
            self._boost_step(self._xs_buf[-128:], self._ys_buf[-128:])

        return self.model(x, y)[1].item()
