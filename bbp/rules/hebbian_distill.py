"""HEBBIAN DISTILL: Hebbian learning from a counting oracle.

The oracle (counting teacher) tells us the correct output distribution for every
input. We DON'T need gradients to learn from it — we need CORRELATIONS.

The idea: at each position, the oracle gives us p(next_char | context). The hidden
state h encodes the context. We want lm_head to map h -> log p. That's just:

  W += lr * h^T @ (log p - h @ W)

This is the delta rule / Widrow-Hoff / LMS — a Hebbian-like update that uses
only LOCAL information (the hidden state h and the target p). No backward pass
through the network, no chain rule, no credit assignment across layers.

For the deeper layers, we use a Hebbian co-activation rule: if neuron i in
layer L fires together with the "correct" output pattern, strengthen i's
contribution to the output. This is Oja's rule or a simple Hebbian outer product.

The genius: the counting teacher gives us the CORRECT output for FREE (it's just
counting). We only need to learn the mapping from hidden states to those outputs.
The attention mechanism copies context characters — that's hand-wired. The MLP
detects patterns — that's either hand-wired (grandmother cells) or learned by
Hebbian association.

Plan:
1. Hand-wire attention (copy heads)
2. Count (build teacher)
3. Hebbian/delta-rule updates on lm_head: W += lr * h^T @ (target - prediction)
4. Optionally: Hebbian updates on MLP proj: strengthen neurons whose output
   correlates with reducing prediction error

This is autograd="none" — no .backward(), but it IS a learning algorithm.

The question: can Hebbian distillation from a counting oracle match backprop?
"""
import math
import itertools

import numpy as np
import torch

from . import Rule, register
from ._teacher import CountingTeacher
from .grandmother_cells import CellModel, count_strings, hire


@register
class HebbianDistill(Rule):
    name = "hebbian_distill"
    autograd = "none"
    blurb = "Hebbian/delta-rule distillation from counting oracle into transformer readout + MLP."
    paper = "Novel: forward-only Widrow-Hoff learning from n-gram oracle, hand-wired attention."
    defaults = dict(count_steps=1500, order=9, beta=16.0, kappa=4.0, max_len=9,
                    lr_head=0.1,      # learning rate for lm_head delta rule
                    lr_mlp=0.01,      # learning rate for MLP Hebbian updates
                    momentum=0.9,     # momentum for the delta rule
                    )
    tune = "lr_head"

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
        # Momentum buffer for lm_head
        self.v_head = torch.zeros_like(model.lm_head.weight)
        # For MLP proj updates
        self.v_mlps = [torch.zeros_like(blk.mlp.proj.weight) for blk in model.blocks]

    @torch.no_grad()
    def _wire_cells(self):
        V = self.cm.V
        t = count_strings(np.concatenate(self.xs), np.concatenate(self.ys), V, self.N)
        cells, outs, base = hire(t, V, self.N, self.hp["kappa"], self.cm.n_cells)
        self.cm.keys(cells, outs, base)

    @torch.no_grad()
    def _hebbian_step(self, x, y):
        """Delta-rule update on readout + Hebbian on MLP."""
        model = self.model
        dev = model.lm_head.weight.device

        # Get teacher soft targets
        p_teacher = self.teacher(x, y, loo=True)  # (B, T, V)
        log_p = torch.log(p_teacher.clamp_min(1e-8))

        # Forward pass to get hidden states
        tok = model.tok_emb(x) + model.pos_emb(torch.arange(x.shape[1], device=dev))
        h = tok
        pre_mlp_acts = []
        post_mlp_acts = []
        for blk in model.blocks:
            # Attention
            h_attn = blk.ln1(h)
            h_attn = blk.attn(h_attn)
            h = h + h_attn
            # MLP — capture pre and post activations
            h_mlp_in = blk.ln2(h)
            fc_out = blk.mlp.fc(h_mlp_in)
            act = torch.nn.functional.gelu(fc_out)
            pre_mlp_acts.append(act)  # (B, T, n_fc)
            mlp_out = blk.mlp.proj(act)
            post_mlp_acts.append(mlp_out)
            h = h + mlp_out

        h = model.ln_f(h)

        # Current predictions
        logits = model.lm_head(h)  # (B, T, V)
        pred = logits.log_softmax(-1)

        # Delta-rule on lm_head: W += lr * h^T @ (target - pred)
        # target = log_p (teacher log-probs), pred = model log-probs
        error = log_p - pred  # (B, T, V)

        # Average outer product: (V, C) = error^T @ h, averaged over B*T
        BT = x.shape[0] * x.shape[1]
        h_flat = h.reshape(-1, h.shape[-1])    # (BT, C)
        e_flat = error.reshape(-1, error.shape[-1])  # (BT, V)

        grad_head = e_flat.T @ h_flat / BT  # (V, C)

        lr_h = self.hp["lr_head"]
        mom = self.hp["momentum"]
        self.v_head.mul_(mom).add_(grad_head)
        model.lm_head.weight.add_(self.v_head * lr_h)

        # Hebbian update on MLP proj weights
        # For each MLP: if the prediction error at a position correlates with
        # a neuron's activation, strengthen that neuron's projection
        lr_m = self.hp["lr_mlp"]
        for i, (blk, act) in enumerate(zip(model.blocks, pre_mlp_acts)):
            # act: (B, T, n_fc), error: (B, T, V)
            # We want: proj.weight (C, n_fc) updated by error correlation
            # But error is in output space. Project error back through lm_head:
            error_hidden = e_flat @ model.lm_head.weight  # (BT, C)
            act_flat = act.reshape(-1, act.shape[-1])  # (BT, n_fc)
            # Outer product: which hidden dims correlate with which neurons
            grad_mlp = error_hidden.T @ act_flat / BT  # (C, n_fc)
            self.v_mlps[i].mul_(mom).add_(grad_mlp)
            blk.mlp.proj.weight.add_(self.v_mlps[i] * lr_m)

        loss = -pred.gather(-1, y.unsqueeze(-1)).mean()
        return loss.item()

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
                self.phase = "learn"
            elif self.forwards in (1, n_count // 10, n_count // 3):
                self._wire_cells()
            return self.model(x, y)[1].item()

        return self._hebbian_step(x, y)
