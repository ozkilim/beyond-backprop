"""ELM ORACLE: Extreme Learning Machine meets counting teacher.

Random features (random fc projections of char-code dims + GELU) create
continuous nonlinear features. Ridge regression decodes them to teacher logits.

Key fix over naive ELM: fc projects ONLY from char-code dims (the first
b*N dims of hidden state), zeroing out K-pair and position dims that would
otherwise dominate the random projection with uninformative signal.

One-shot fitting: collect features after counting, fit once, run the rest.

Autograd level: "none" — no .backward() anywhere.
"""
import math
import itertools

import numpy as np
import torch

from . import Rule, register
from ._teacher import CountingTeacher


def _make_codes(V):
    b = next(bb for bb in range(2, 33, 2) if math.comb(bb, bb // 2) >= V)
    words = [[1.0 if i in s else -1.0 for i in range(b)]
             for s in itertools.islice(itertools.combinations(range(b), b // 2), V)]
    return b, torch.tensor(words, dtype=torch.float32)


@register
class ELMOracle(Rule):
    name = "elm_oracle"
    autograd = "none"
    blurb = "Random features + ridge regression decoding from counting teacher."
    paper = "Novel: ELM (Huang 2006) in transformer with hand-wired attention and counting oracle."
    defaults = dict(count_steps=2000, order=9, beta=16.0,
                    reg_alpha=0.01,
                    fc_scale=25.0,        # compensate for LayerNorm ÷25 from K pair
                    fc_bias_range=2.0,
                    n_fit_samples=16384,
                    seed_rf=42,
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

        self.teacher = CountingTeacher(
            self.V, int(self.hp["order"]), self.hp["beta"]
        )
        self.k = 0
        self._xs_buf, self._ys_buf = [], []
        self.fitted = False

        self._setup(model)

    @torch.no_grad()
    def _setup(self, model):
        V, C, H, T = self.V, self.C, self.H, self.T
        hd = C // H

        for p in model.parameters():
            p.zero_()
        for ln in [model.ln_f] + [l for blk in model.blocks for l in (blk.ln1, blk.ln2)]:
            ln.weight.fill_(1.0)

        self.b, self.code = _make_codes(V)
        b = self.b
        dev = model.lm_head.weight.device

        model.tok_emb.weight[:, :b] = self.code.to(dev)

        F = min(7, hd // 2)
        t_pos = torch.arange(T, dtype=torch.float32)
        om = 2 * math.pi / (3.0 * (2.0 * T / 3.0) ** (
            torch.arange(F) / max(F - 1, 1)))

        K_const = 200.0
        self.np_ = next(n for n in range(H, -1, -1)
                        if b * (1 + n) + 2 * F + 2 + V <= C)
        self.N = self.np_ + 1
        p0 = b * self.N
        k0 = p0 + 2 * F
        self.l0 = k0 + 2
        self.char_dims = b * self.N  # first char_dims dimensions are char codes

        for j in range(F):
            model.pos_emb.weight[:, p0 + 2*j] = torch.cos(om[j] * t_pos).to(dev)
            model.pos_emb.weight[:, p0 + 2*j + 1] = torch.sin(om[j] * t_pos).to(dev)
        model.pos_emb.weight[:, k0] = K_const
        model.pos_emb.weight[:, k0 + 1] = -K_const

        sd = math.sqrt((2 * K_const**2 + b * self.N + F) / C)
        gain = 200.0 * math.sqrt(hd)
        self.sd = sd
        U = 20.0
        self.U = U

        # Wire copy heads
        for back in range(self.np_):
            att = model.blocks[back // H].attn
            h_idx = back % H
            W = att.qkv.weight
            for j in range(F):
                c_r = math.cos(-om[j].item() * (back + 1))
                s_r = math.sin(-om[j].item() * (back + 1))
                r = h_idx * hd + 2 * j
                pc, ps = p0 + 2 * j, p0 + 2 * j + 1
                W[r, pc] = sd * gain * c_r
                W[r, ps] = -sd * gain * s_r
                W[r + 1, pc] = sd * gain * s_r
                W[r + 1, ps] = sd * gain * c_r
                W[C + r, pc] = sd
                W[C + r + 1, ps] = sd
            for i in range(b):
                W[2 * C + h_idx * hd + i, i] = sd
                att.proj.weight[b * (1 + back) + i, h_idx * hd + i] = 1.0

        # lm_head: identity on logit slots
        model.lm_head.weight[:, self.l0:self.l0 + V] = U * sd * torch.eye(V, device=dev)

        # Random fc: project ONLY from char-code dims
        rng = torch.Generator(device='cpu').manual_seed(int(self.hp["seed_rf"]))
        fc_scale = self.hp["fc_scale"]
        bias_range = self.hp["fc_bias_range"]
        cdims = self.char_dims

        for blk in model.blocks:
            n_fc = blk.mlp.fc.out_features
            # Only char-code dims get random weights, everything else zero
            w = torch.zeros(n_fc, C)
            w[:, :cdims] = torch.randn(n_fc, cdims, generator=rng) * fc_scale / math.sqrt(cdims)
            bias = torch.empty(n_fc).uniform_(-bias_range, bias_range, generator=rng)

            blk.mlp.fc.weight.copy_(w.to(dev))
            blk.mlp.fc.bias.copy_(bias.to(dev))
            blk.mlp.proj.weight.zero_()
            blk.mlp.proj.bias.zero_()

    @torch.no_grad()
    def _collect_features_and_targets(self, xs, ys):
        """Collect GELU features from all blocks and teacher targets."""
        model = self.model
        dev = model.lm_head.weight.device
        max_n = int(self.hp["n_fit_samples"])

        all_feats = []  # will be (N, total_features)
        all_targets = []

        for x, y in zip(xs, ys):
            x_d, y_d = x.to(dev), y.to(dev)

            # Forward through attention only (all proj=0, MLP contributes nothing)
            h = model.tok_emb(x_d) + model.pos_emb(torch.arange(x_d.shape[1], device=dev))
            block_feats = []
            for blk in model.blocks:
                # Attention
                h = h + blk.attn(blk.ln1(h))
                # Compute GELU features (but DON'T add MLP output to residual since proj=0)
                h_ln = blk.ln2(h)
                feats = torch.nn.functional.gelu(blk.mlp.fc(h_ln))  # (B, T, n_fc)
                block_feats.append(feats)
                # Skip MLP in residual (proj is zero, so this is a no-op anyway)

            # Concatenate all blocks' features
            combined = torch.cat(block_feats, dim=-1)  # (B, T, 4*n_fc)
            all_feats.append(combined.reshape(-1, combined.shape[-1]))

            # Teacher targets
            p = self.teacher(x_d, y_d, loo=True)
            all_targets.append(
                torch.log(p.clamp_min(1e-8)).reshape(-1, p.shape[-1])
            )

        F = torch.cat(all_feats)
        Y = torch.cat(all_targets)

        N = F.shape[0]
        if N > max_n:
            idx = torch.randperm(N, device=dev)[:max_n]
            F, Y = F[idx], Y[idx]
            N = max_n

        return F, Y, N

    @torch.no_grad()
    def _fit_oneshot(self, xs, ys):
        """Single-shot fitting: ridge regression from concatenated features to teacher logits.

        Instead of fitting each block separately (which requires sequential forward
        passes), fit ALL blocks' proj jointly by concatenating their features.

        Since all proj are zero, the hidden state at each block is the same regardless
        of other blocks' proj. So we can collect all features in one forward pass and
        fit one big regression.

        The regression solution is then split across blocks' proj matrices.
        """
        model = self.model
        dev = model.lm_head.weight.device
        alpha = self.hp["reg_alpha"]
        V, l0 = self.V, self.l0
        U, sd = self.U, self.sd

        F, Y, N = self._collect_features_and_targets(xs, ys)

        # Center targets
        Y_mean = Y.mean(0, keepdim=True)
        Y_c = Y - Y_mean

        # Ridge regression: F @ W ≈ Y_c
        total_feats = F.shape[1]  # 4 * n_fc
        FtF = F.T @ F + alpha * N * torch.eye(total_feats, device=dev)
        FtY = F.T @ Y_c
        try:
            W = torch.linalg.solve(FtF, FtY)  # (total_feats, V)
        except torch.linalg.LinAlgError:
            W = torch.linalg.lstsq(FtF, FtY).solution

        # Split W across blocks and put in proj
        n_fc = model.blocks[0].mlp.fc.out_features
        for bi, blk in enumerate(model.blocks):
            W_block = W[bi * n_fc:(bi + 1) * n_fc, :]  # (n_fc, V)
            # proj.weight is (C, n_fc). Set logit-slot rows.
            proj_w = torch.zeros(self.C, n_fc, device=dev)
            proj_w[l0:l0 + V, :] = W_block.T / (U * sd)
            blk.mlp.proj.weight.copy_(proj_w)

        # Set bias for the mean (in the last block)
        bias_logits = Y_mean.squeeze() / (U * sd)
        model.blocks[-1].mlp.proj.bias[l0:l0 + V] = bias_logits.to(dev)

        self.fitted = True

    @torch.no_grad()
    def step(self, x, y):
        self.k += 1
        self.forwards += 1
        n_count = int(self.hp["count_steps"])

        if not self.teacher.ready:
            self._xs_buf.append(x.cpu())
            self._ys_buf.append(y.cpu())
            self.teacher.observe(x, y)
            if self.k >= n_count:
                self.teacher.build()
            return self.model(x, y)[1].item()

        if not self.fitted:
            # Use last 64 batches for the single fit (avoid OOM)
            self._fit_oneshot(self._xs_buf[-64:], self._ys_buf[-64:])
            self._xs_buf, self._ys_buf = [], []

        return self.model(x, y)[1].item()
