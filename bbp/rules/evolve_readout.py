"""EVOLVE_READOUT: hand-wire attention, evolve only the readout layer.

The key insight: grandmother_cells gets 1.900 because it's limited by how many
n-gram patterns fit in the MLP neurons. But the model has 826K parameters total,
and lm_head alone is 65×128 = 8320 params. The attention and MLP layers are
~800K params mostly wasted.

What if we:
1. Hand-wire attention to copy context (like grandmother_cells) — FREE, no learning
2. Use MLP neurons as n-gram detectors (like grandmother_cells) — FREE
3. But then EVOLVE the readout (lm_head) using CMA-ES on just 8320 params?

CMA-ES on 8320 dims is feasible in 5 minutes. The search space is small enough.
Each evaluation: forward pass the model, get loss. The oracle (counting teacher)
gives us a richer signal than raw loss — we can compute KL divergence.

Actually, even simpler: evolve the MLP output projection weights too. Each MLP
neuron fires for a pattern — we just need to learn what logit offset each firing
pattern should contribute. That's the proj.weight matrix, which is (C, n_fc) per
block — so 128×512×4 = 262K params across all blocks. Too many for CMA-ES.

Sticking with the plan: hand-wire everything, ridge-regress the readout against
the counting teacher, and then use CMA-ES to fine-tune just lm_head (8320 params)
against actual test-time loss. The ridge regression gets us close, CMA-ES polishes.

Wait — this is basically compiler.py + CMA-ES polish. Let me try something
different instead.

APPROACH: Zeroth-order optimization of ONLY lm_head (8320 params) while
everything else is hand-wired. Use antithetic sampling (±perturbation pairs)
for variance reduction. With 8320 params and batch eval, we can do ~100
perturbation pairs per step, each step takes ~0.1s, so ~3000 ZO steps in 5 min
after the counting phase.

Autograd level: "none" — no .backward() anywhere.
"""
import math
import itertools

import numpy as np
import torch

from . import Rule, register
from ._teacher import CountingTeacher
from .grandmother_cells import CellModel, count_strings, hire


@register
class EvolveReadout(Rule):
    name = "evolve_readout"
    autograd = "none"
    blurb = "Hand-wire attention + n-gram cells, then evolve only the readout with zeroth-order optimization."
    paper = "Novel: grandmother_cells base + zeroth-order readout tuning against counting oracle."
    defaults = dict(count_steps=1500, order=9, beta=16.0, kappa=4.0, max_len=9,
                    zo_lr=0.01, zo_sigma=0.01, zo_pairs=32)
    tune = "zo_lr"

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
        self.phase = "count"
        self.cells_wired = False

    @torch.no_grad()
    def _wire_cells(self):
        """Build counting table and wire grandmother cells."""
        V = self.cm.V
        t = count_strings(np.concatenate(self.xs), np.concatenate(self.ys), V, self.N)
        cells, outs, base = hire(t, V, self.N, self.hp["kappa"], self.cm.n_cells)
        self.cm.keys(cells, outs, base)
        self.cells_wired = True

    @torch.no_grad()
    def _zo_step_readout(self, x, y):
        """One step of zeroth-order optimization on lm_head only."""
        model = self.model
        dev = model.lm_head.weight.device
        W = model.lm_head.weight  # (V, C)

        sigma = self.hp["zo_sigma"]
        lr = self.hp["zo_lr"]
        n_pairs = int(self.hp["zo_pairs"])

        # Get teacher targets for KL-based objective
        p_teacher = self.teacher(x, y, loo=True)  # (B, T, V)

        # Current loss
        logits_0, loss_0 = model(x, y)

        # Flatten the readout weights
        w_flat = W.reshape(-1)  # (V*C,)
        d = w_flat.numel()

        # Antithetic perturbation pairs
        grad_est = torch.zeros_like(w_flat)
        for _ in range(n_pairs):
            z = torch.randn(d, device=dev)
            # + perturbation
            W.add_(z.reshape_as(W) * sigma)
            _, loss_p = model(x, y)
            # - perturbation (undo + and go negative)
            W.add_(z.reshape_as(W) * (-2 * sigma))
            _, loss_m = model(x, y)
            # Restore
            W.add_(z.reshape_as(W) * sigma)

            # Gradient estimate
            grad_est.add_(z * (loss_p.item() - loss_m.item()) / (2 * sigma))

        grad_est.div_(n_pairs)

        # Update
        W.add_(grad_est.reshape_as(W) * (-lr))

        return loss_0.item()

    @torch.no_grad()
    def step(self, x, y):
        self.forwards += 1
        n_count = int(self.hp["count_steps"])

        if self.phase == "count":
            self.xs.append(x.cpu().numpy())
            self.ys.append(y.cpu().numpy())
            self.teacher.observe(x, y)
            if self.forwards >= n_count:
                self.teacher.build()
                self._wire_cells()
                self.xs, self.ys = [], []
                self.phase = "evolve"
            elif self.forwards in (1, n_count // 10, n_count // 3):
                self._wire_cells()
            return self.model(x, y)[1].item()

        # Phase 2: ZO optimization of readout
        return self._zo_step_readout(x, y)
