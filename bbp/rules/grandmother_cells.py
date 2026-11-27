"""GRANDMOTHER CELLS: one neuron per memory.

Neuroscience's favourite joke: a single neuron that fires only for your
grandmother. Here every MLP neuron is an AND-gate that fires only when the last
n characters are one specific string ("the ", "ROME", ...). No gradient, no
error signal, no learning rate:

  * COUNT   which character follows which recent string
  * HIRE    the strings worth a neuron: most log-likelihood gained over the
            one-character-shorter string, shorter strings hired first
  * WIRE    each neuron to add  log P(next | string) - log P(next | shorter string)

All the cells along a suffix chain fire together and their contributions
telescope to log P(next | longest hired string). It is a variable-order Markov
model (PPM / context trees) written into a transformer's weights.

What is given, not learned: attention is hand-wired so head h copies the
character h+1 back. The only thing taken from data is counts.

Stream: [char code | copied char codes | position | +K -K | logits]. Char codes
are zero-sum +-1 words so LayerNorm's mean subtraction cannot disturb the
AND-gates, and the big +-K pair makes LayerNorm (almost) a constant division.
"""
import heapq
import itertools
import math

import numpy as np
import torch

from . import Rule, register


class CellModel:
    """The hand-made scaffolding: reflexes (copy heads), codes, AND-gates, logit slots."""
    K, U, A = 200.0, 20.0, 5.0  # constant pair, logit unit, activation of a firing cell

    @torch.no_grad()
    def __init__(self, model, extra=0):
        """extra: copy heads beyond block 0's, placed in later blocks (grandmother_cells uses none)."""
        cfg = model.cfg
        self.model, self.V, self.C, self.H = model, cfg.vocab_size, cfg.n_embd, cfg.n_head
        self.hd, self.dev = self.C // self.H, model.lm_head.weight.device
        V, C, hd = self.V, self.C, self.hd
        self.b = next(b for b in range(2, 33, 2) if math.comb(b, b // 2) >= V)
        words = [[1.0 if i in s else -1.0 for i in range(self.b)]
                 for s in itertools.islice(itertools.combinations(range(self.b), self.b // 2), V)]
        self.code = torch.tensor(words)  # (V, b), every row sums to 0
        self.np_ = next(n for n in range(self.H + extra, -1, -1) if self.b * (1 + n) + 6 + 2 + V <= C)
        self.F = min(7, hd // 2, (C - V - 2 - self.b * (1 + self.np_)) // 2)
        self.N = self.np_ + 1  # longest string a cell can hold
        self.p0 = self.b * self.N
        self.k0 = self.p0 + 2 * self.F
        self.l0 = self.k0 + 2  # logit slots l0 .. l0+V-1
        self.sd = math.sqrt((2 * self.K ** 2 + self.b * self.N + self.F) / C)
        T = cfg.block_size
        t = torch.arange(T, dtype=torch.float32)
        om = 2 * math.pi / (3.0 * (2.0 * T / 3.0) ** (torch.arange(self.F) / max(self.F - 1, 1)))

        for p in model.parameters():
            p.zero_()
        for ln in [model.ln_f] + [l for blk in model.blocks for l in (blk.ln1, blk.ln2)]:
            ln.weight.fill_(1.0)
        model.tok_emb.weight[:, :self.b] = self.code.to(self.dev)
        for j in range(self.F):
            model.pos_emb.weight[:, self.p0 + 2 * j] = torch.cos(om[j] * t).to(self.dev)
            model.pos_emb.weight[:, self.p0 + 2 * j + 1] = torch.sin(om[j] * t).to(self.dev)
        model.pos_emb.weight[:, self.k0], model.pos_emb.weight[:, self.k0 + 1] = self.K, -self.K
        gain = 200.0 * math.sqrt(hd)
        for back in range(self.np_):  # copy the char back+1 back (H heads per block)
            att, h = model.blocks[back // self.H].attn, back % self.H
            W = att.qkv.weight
            for j in range(self.F):
                c, s = math.cos(-om[j] * (back + 1)), math.sin(-om[j] * (back + 1))
                r, pc, ps = h * hd + 2 * j, self.p0 + 2 * j, self.p0 + 2 * j + 1
                W[r, pc], W[r, ps], W[r + 1, pc], W[r + 1, ps] = [self.sd * gain * v for v in (c, -s, s, c)]
                W[C + r, pc] = W[C + r + 1, ps] = self.sd
            for i in range(self.b):
                W[2 * C + h * hd + i, i] = self.sd
                att.proj.weight[self.b * (1 + back) + i, h * hd + i] = 1.0
        model.lm_head.weight[:, self.l0:self.l0 + V] = self.U * self.sd * torch.eye(V, device=self.dev)
        self.n_fc = model.blocks[0].mlp.fc.out_features
        self.n_cells = self.n_fc * len(model.blocks)

    def keys(self, cells, outs, base):
        """cells: list of tuples, each a list of (steps_back, char). outs: (n, V) logit contributions."""
        n, C, g = len(cells), self.C, self.A / 2.0
        fw, fb = torch.zeros(self.n_cells, C), torch.full((self.n_cells,), -2 * self.A)
        pw = torch.zeros(self.n_cells, C)
        for i, cell in enumerate(cells):
            for back, ch in cell:
                fw[i, self.b * back:self.b * (back + 1)] = g * self.sd * self.code[ch]
            fb[i] = self.A - g * self.b * len(cell)  # fires at +A iff every char matches, else <= -A
        outs = outs - outs.mean(1, keepdim=True)
        pw[:n, self.l0:self.l0 + self.V] = outs / (self.A * self.U)
        with torch.no_grad():
            for bi, blk in enumerate(self.model.blocks):
                sl = slice(bi * self.n_fc, (bi + 1) * self.n_fc)
                blk.mlp.fc.weight.copy_(fw[sl].to(self.dev))
                blk.mlp.fc.bias.copy_(fb[sl].to(self.dev))
                blk.mlp.proj.weight.copy_(pw[sl].T.to(self.dev))
            self.model.blocks[-1].mlp.proj.bias[self.l0:self.l0 + self.V] = ((base - base.mean()) / self.U).to(self.dev)


def count_strings(xs, ys, V, N):
    """{n: (pair_keys, counts)} with pair key = string*V + next; string = sum x[t-i]*V^i, length n."""
    out = {0: np.unique(ys.reshape(-1), return_counts=True)}
    for n in range(1, N + 1):
        T = xs.shape[1]
        key = sum(xs[:, n - 1 - i:T - i].astype(np.int64) * V ** i for i in range(n)) * V + ys[:, n - 1:]
        out[n] = np.unique(key.reshape(-1), return_counts=True)
    return out


def hire(tables, V, N, kappa, budget):
    """Best-first choice of `budget` strings; returns cells, their logit residuals and the unigram base."""
    uni = np.zeros(V)
    uni[tables[0][0]] = tables[0][1]
    kappa = kappa * uni.sum() / 1e6  # batches resample the text, so prior strength scales with chars seen
    p0 = (uni + 0.1 * uni.sum() / 1e6) / (uni + 0.1 * uni.sum() / 1e6).sum()
    info, kids, prev_keys, prev_p = {}, {}, None, None
    for n in range(1, N + 1):
        keys, c = tables[n]
        ctx, y = keys // V, keys % V
        parent_p = p0[y] if n == 1 else prev_p[np.searchsorted(prev_keys, (ctx % V ** (n - 1)) * V + y)]
        uc, first = np.unique(ctx, return_index=True)
        tot = np.add.reduceat(c, first)
        p = (c + kappa * parent_p) / (np.repeat(tot, np.diff(np.append(first, len(c)))) + kappa)
        gain = np.add.reduceat(c * (np.log(p) - np.log(parent_p)), first)
        ends = np.append(first[1:], len(c))
        for cx, g, a, e in zip(uc.tolist(), gain.tolist(), first.tolist(), ends.tolist()):
            info[(n, cx)] = (a, e)
            kids.setdefault((n - 1, cx % V ** (n - 1) if n > 1 else 0), []).append((-g, n, cx))
        prev_keys, prev_p = keys, p
    heap = list(kids.get((0, 0), []))
    heapq.heapify(heap)
    dense, cells, outs = {(0, 0): p0}, [], []
    while heap and len(cells) < budget:
        g, n, cx = heapq.heappop(heap)
        if g >= 0:
            continue
        a, e = info[(n, cx)]
        keys, c = tables[n]
        pp = dense[(n - 1, cx % V ** (n - 1) if n > 1 else 0)]
        row = np.zeros(V)
        row[keys[a:e] % V] = c[a:e]
        p = (row + kappa * pp) / (row.sum() + kappa)
        dense[(n, cx)] = p
        cells.append([(i, (cx // V ** i) % V) for i in range(n)])
        outs.append(np.log(p) - np.log(pp))
        for k in kids.get((n, cx), []):
            heapq.heappush(heap, k)
    outs = torch.tensor(np.array(outs).reshape(-1, V), dtype=torch.float32)
    return cells, outs, torch.tensor(np.log(p0), dtype=torch.float32)


@register
class GrandmotherCells(Rule):
    name = "grandmother_cells"
    autograd = "none"
    blurb = "One neuron per memory: each MLP neuron fires for one specific recent string, hired by counting, wired to add what that string says beyond the shorter one. No gradients, no error signal. Copy heads are hand-wired."
    paper = "Lettvin's grandmother cell; Cleary & Witten 1984 (PPM); Willems et al. 1995 (context trees)"
    defaults = dict(kappa=64.0, count_steps=3000, max_len=9)
    tune = "kappa"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.cm = CellModel(model)
        self.N = min(self.cm.N, int(self.hp["max_len"]))
        self.xs, self.ys, self.k, self.cells = [], [], 0, []

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
