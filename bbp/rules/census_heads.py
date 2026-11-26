"""CENSUS HEADS: attention trained by asking a census where to look.

Every gradient-free rule so far stalled at or before the bigram, because none
of them could train attention, and attention is the only way to use anything
earlier than the current character. This rule trains attention with no
error signal crossing a layer. It never calls autograd, but the Q/K update is
the chain rule written out by hand inside one attention layer (a test checks it
equals autograd), so it is labelled "local", not "none".

The trick is an EM story. Pretend the next character y_t was "caused" by one
earlier character x_s (or by nothing beyond the current char x_t). A running
census of letter triples (current char, char d steps back, next char) says how
likely each candidate cause is:

    posterior(s | y_t)  ∝  attention(s) * P(y_t | x_t, x_s, offset d) / P(y_t | x_t)   (E-step)

with the head's own current attention as the prior (proper EM: the mixing
weights are the prior; prior="uniform" is the ablation). That posterior is the attention pattern the head *should* have had for this
query. The head is nudged toward it with a delta rule on its own scores
(dS = attention - posterior), turned into Q/K updates by the head's own
bilinear form (M-step).

The rest of the network is wired, not trained:
  * stream = [one-hot current char | sinusoidal position | what the heads fetched]
  * each head's values are a fixed random code of the character it lands on
  * all other block outputs are zeroed (pass-through)
  * the readout learns with the delta rule (exact for the last layer)

So the readout sees "current char + whatever the heads fetched", and the only
hidden thing that learns is *where the heads look*. Head h is barred from the
nearest h offsets so the heads don't all fetch the same character.

`attn` switches the ablation: "census" (learn), "position-only" (learn Q/K with
neutral census evidence on the same allowed offsets), "frozen" (random heads,
never trained), "wired" (hand-set to look exactly h+1 steps back: a reference,
not a proven bound). The position-only control retains the attention prior and
readout, which can still depend on token content. Census counting and refresh
continue at the same cadence.
"""
import math

import torch
import torch.nn.functional as F

from . import Rule, register


@register
class CensusHeads(Rule):
    name = "census_heads"
    autograd = "local"  # hand-written, but it IS the chain rule inside one attention layer
    blurb = "Attention trained by a layer-local rule: an EM E-step from a running census of letter triples says which earlier char caused the next one; heads are nudged to look there. Readout: delta rule."
    paper = "Dempster et al. 1977 (EM); supervised attention (Liu et al. 2016) with a census as the supervisor"
    defaults = dict(lr=3e-3, heads=2, max_offset=8, smoothing=5.0, beta=1.0, prior="self", sharpen=1.0, attn="census", tok_scale=10.0, refresh=20)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        cfg = model.cfg
        self.V, self.C, self.H = cfg.vocab_size, cfg.n_embd, cfg.n_head
        self.hd = self.C // self.H
        self.nh = min(int(self.hp["heads"]), self.H)
        self.D = int(self.hp["max_offset"])
        dev = model.lm_head.weight.device
        V, C = self.V, self.C
        assert C > V + 4, "needs spare stream dims beyond the one-hot chars"
        self.F = min(7, (C - V) // 4, self.hd // 2)  # sinusoid pairs for position
        self.p0 = V  # position dims start
        self.a0 = V + 2 * self.F  # fetched-char dims start
        self.w = min((C - self.a0) // self.nh, self.hd)  # stream dims per head
        self._wire(dev)

        self.c1 = torch.zeros(V, V, device=dev)
        self.c2 = torch.zeros(self.D, V, V, V, device=dev)  # [offset-1, x_s, x_t, y]
        self.table = torch.zeros(self.D, V, V, V, device=dev)
        self.k = 0
        self.caught = {}
        model.blocks[0].attn.register_forward_hook(lambda m, i, o: self.caught.__setitem__("z", i[0]))
        model.lm_head.register_forward_hook(lambda m, i, o: self.caught.__setitem__("f", i[0]))
        qkv = model.blocks[0].attn.qkv
        params = [model.lm_head.weight] + ([qkv.weight, qkv.bias] if self.hp["attn"] in ("census", "position-only") else [])
        self.opt = torch.optim.Adam(params, lr=self.hp["lr"])

    # ---------------------------------------------------------------- wiring
    @torch.no_grad()
    def _wire(self, dev):
        m, V, C, hd = self.model, self.V, self.C, self.hd
        g = torch.Generator().manual_seed(1234)
        m.tok_emb.weight.zero_()
        m.tok_emb.weight[:, :V] = self.hp["tok_scale"] * torch.eye(V, device=dev)
        T = m.cfg.block_size
        t = torch.arange(T, dtype=torch.float32)
        periods = 3.0 * (2.0 * T / 3.0) ** (torch.arange(self.F) / max(self.F - 1, 1))
        self.omega = 2 * math.pi / periods
        m.pos_emb.weight.zero_()
        for j in range(self.F):
            m.pos_emb.weight[:, self.p0 + 2 * j] = torch.cos(self.omega[j] * t).to(dev)
            m.pos_emb.weight[:, self.p0 + 2 * j + 1] = torch.sin(self.omega[j] * t).to(dev)
        for blk in m.blocks:
            for lin in (blk.attn.proj, blk.mlp.proj):
                lin.weight.zero_()
                lin.bias.zero_()
        m.ln_f.weight.fill_(1.0)
        m.ln_f.bias.zero_()
        m.lm_head.weight.zero_()

        attn = m.blocks[0].attn
        W, b = attn.qkv.weight, attn.qkv.bias
        W[2 * C :].zero_()
        b[2 * C :].zero_()
        # LayerNorm turns tok_scale*onehot into roughly sqrt(C)*onehot; scale codes so values are ~N(0,1)
        zs = math.sqrt(C)
        for h in range(self.nh):
            code = torch.randn(self.w, V, generator=g) / zs
            W[2 * C + h * hd : 2 * C + h * hd + self.w, :V] = code.to(dev)
            for j in range(self.w):
                attn.proj.weight[self.a0 + h * self.w + j, h * hd + j] = 1.0
        if self.hp["attn"] == "wired":
            self._wire_offsets()

    @torch.no_grad()
    def _wire_offsets(self, gain=60.0):
        """Hand-set head h to look exactly h+1 steps back (a reference point)."""
        attn, C, hd, F_ = self.model.blocks[0].attn, self.C, self.hd, self.F
        W, b = attn.qkv.weight, attn.qkv.bias
        W[: 2 * C].zero_()
        b[: 2 * C].zero_()
        for h in range(self.nh):
            delta = h + 1
            for j in range(F_):
                c, s = math.cos(-self.omega[j] * delta), math.sin(-self.omega[j] * delta)
                pc, ps = self.p0 + 2 * j, self.p0 + 2 * j + 1
                r = h * hd + 2 * j
                # q = gain * R(-omega*delta) p_t,  k = p_s   ->  q.k peaks at t - s = delta
                W[r, pc], W[r, ps] = gain * c, -gain * s
                W[r + 1, pc], W[r + 1, ps] = gain * s, gain * c
                W[C + r, pc], W[C + r + 1, ps] = 1.0, 1.0

    # ---------------------------------------------------------------- census
    @torch.no_grad()
    def _count(self, x, y):
        V = self.V
        self.c1 += torch.bincount((x * V + y).reshape(-1), minlength=V * V).view(V, V).float()
        for d in range(1, min(self.D, x.size(1) - 1) + 1):
            idx = (x[:, :-d] * V + x[:, d:]) * V + y[:, d:]
            self.c2[d - 1] += torch.bincount(idx.reshape(-1), minlength=V ** 3).view(V, V, V).float()

    @torch.no_grad()
    def _refresh(self):
        k = self.hp["smoothing"]
        p_bi = (self.c1 + 0.1) / (self.c1 + 0.1).sum(1, keepdim=True)  # [x_t, y]
        n = self.c2.sum(-1, keepdim=True)
        p_tri = (self.c2 + k * p_bi[None, None]) / (n + k)
        self.table = p_tri.log() - p_bi.log()[None, None]

    # ---------------------------------------------------------------- learning
    def _teach_heads(self, x, y):
        z = self.caught["z"]  # (B, T, C): what block 0's attention reads
        B, T, C = z.shape
        attn, hd = self.model.blocks[0].attn, self.hd
        W, bias = attn.qkv.weight, attn.qkv.bias
        dev = z.device
        # census log-likelihood ratio of "s caused y_t" for every query t and key s = t - d
        L = torch.full((B, T, T), float("-inf"), device=dev)
        ar = torch.arange(T, device=dev)
        L[:, ar, ar] = 0.0  # "nothing earlier needed": the bigram explanation
        for d in range(1, min(self.D, T - 1) + 1):
            t = ar[d:]
            if self.hp["attn"] == "position-only":
                L[:, t, t - d] = 0.0  # neutral evidence; preserve -inf outside the allowed window
            else:
                L[:, t, t - d] = self.table[d - 1][x[:, :-d], x[:, d:], y[:, d:]]
        gW = torch.zeros_like(W)
        gb = torch.zeros_like(bias)
        offs = ar[:, None] - ar[None, :]  # t - s
        nq = B * T
        for h in range(self.nh):
            sl_q, sl_k = slice(h * hd, (h + 1) * hd), slice(C + h * hd, C + (h + 1) * hd)
            q = z @ W[sl_q].T + bias[sl_q]
            k = z @ W[sl_k].T + bias[sl_k]
            S = (q @ k.transpose(1, 2)) / math.sqrt(hd)
            S = S.masked_fill(offs < 0, float("-inf"))
            a = S.softmax(-1)
            Lh = L.masked_fill((offs >= 1) & (offs <= h), float("-inf"))  # head h may not look 1..h back
            prior = self.hp["sharpen"] * (a + 1e-9).log() if self.hp["prior"] == "self" else 0.0
            target = (prior + self.hp["beta"] * Lh).softmax(-1)  # E-step: who caused y_t?
            dS = (a - target) / nq  # delta rule on the head's own scores
            dq = dS @ k / math.sqrt(hd)
            dk = dS.transpose(1, 2) @ q / math.sqrt(hd)
            gW[sl_q] = torch.einsum("btd,btc->dc", dq, z)
            gW[sl_k] = torch.einsum("btd,btc->dc", dk, z)
            gb[sl_q], gb[sl_k] = dq.sum((0, 1)), dk.sum((0, 1))
        W.grad, bias.grad = gW, gb

    @torch.no_grad()
    def step(self, x, y):
        self._count(x, y)
        if self.k % self.hp["refresh"] == 0:
            self._refresh()
        self.k += 1
        logits, loss = self.model(x, y)
        f = self.caught["f"]
        N = x.numel()
        dlogits = (logits.softmax(-1) - F.one_hot(y, self.V).float()) / N  # delta rule
        self.model.lm_head.weight.grad = torch.einsum("btv,btc->vc", dlogits, f)
        if self.hp["attn"] in ("census", "position-only"):
            self._teach_heads(x, y)
        self.opt.step()
        self.forwards += 1
        return loss.item()


@register
class CensusHeadsFour(CensusHeads):
    name = "census_heads_4"
    blurb = "census_heads with all 4 heads fetching (fewer stream dims each)."
    defaults = dict(CensusHeads.defaults, heads=4)


def _gelu_grad(a):
    """d/da of exact (erf) GELU."""
    return 0.5 * (1.0 + torch.erf(a / math.sqrt(2.0))) + a * torch.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)


@register
class CensusMLP(CensusHeads):
    """census_heads + an MLP that learns to combine characters, by direct feedback alignment.

    census_heads can only *add up* evidence: the readout is linear in "current char" and
    "fetched char", so it can't express "t then h -> e" any better than "t -> ?" plus "h -> ?".
    A trigram needs conjunctions, and conjunctions need a hidden layer.

    The MLP(s) (blocks 0.. n_mlp-1) read the stream (current char + what the heads fetched) and writes back
    into it. It learns with DFA: each token's output error (p - onehot(y)) is broadcast to
    the MLP through FIXED RANDOM matrices, one for its hidden layer and one for its output,
    and each layer does error (x) its own input. No weight transport. With `ln_aware` the random
    output error is also passed through ln_f's Jacobian, a small hand-written backward step
    through one parameter-free normalisation, so this is "local", not "none". The readout learns to align with the random feedback, not the other way round.

    `feedback` switches the ablation:
      "dfa"    - the rule (fixed random feedback)
      "frozen" - random MLP never trained (readout + heads only)
      "exact"  - DIAGNOSTIC ONLY: the true error for the MLP, hand-computed back through
                 lm_head and ln_f (weight transport). Shows what DFA leaves on the table.
    """
    name = "census_mlp"
    blurb = "census_heads + block-1 MLP trained by direct feedback alignment (output error broadcast through fixed random matrices) so it can learn character conjunctions."
    paper = "Nøkland 2016 (direct feedback alignment) + census_heads (this repo)"
    defaults = dict(CensusHeads.defaults, feedback="dfa", ln_aware=True, n_mlp=1, fc_std=0.1, proj_std=0.02)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        dev = model.lm_head.weight.device
        g = torch.Generator().manual_seed(4321)
        self.mlps = [model.blocks[i].mlp for i in range(min(int(self.hp["n_mlp"]), len(model.blocks)))]
        self.B = []
        for j, mlp in enumerate(self.mlps):
            with torch.no_grad():
                mlp.fc.weight.copy_(torch.randn(mlp.fc.weight.shape, generator=g).to(dev) * self.hp["fc_std"])
                mlp.fc.bias.zero_()
                if self.hp["feedback"] == "frozen":
                    mlp.proj.weight.copy_(torch.randn(mlp.proj.weight.shape, generator=g).to(dev) * self.hp["proj_std"])
            Hd = mlp.fc.out_features
            self.B.append(((torch.randn(self.V, Hd, generator=g) / math.sqrt(self.V)).to(dev),
                           (torch.randn(self.V, self.C, generator=g) / math.sqrt(self.V)).to(dev)))
            mlp.fc.register_forward_hook(lambda m, i, o, j=j: self.caught.update({("u", j): i[0], ("a", j): o}))
            if self.hp["feedback"] != "frozen":
                self.opt.add_param_group({"params": list(mlp.parameters())})
        model.ln_f.register_forward_hook(lambda m, i, o: self.caught.__setitem__("s", i[0]))

    def _ln_back(self, g, x, ln):
        """Hand-written LayerNorm backward. ln=None: just the normalisation (no learned weights)."""
        eps = ln.eps if ln is not None else 1e-5
        if ln is not None:
            g = g * ln.weight
        mu = x.mean(-1, keepdim=True)
        sd = torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + eps)
        xh = (x - mu) / sd
        return (g - g.mean(-1, keepdim=True) - xh * (g * xh).mean(-1, keepdim=True)) / sd

    @torch.no_grad()
    def step(self, x, y):
        self._count(x, y)
        if self.k % self.hp["refresh"] == 0:
            self._refresh()
        self.k += 1
        logits, loss = self.model(x, y)
        f = self.caught["f"]
        err = (logits.softmax(-1) - F.one_hot(y, self.V).float()) / x.numel()
        self.model.lm_head.weight.grad = torch.einsum("btv,btc->vc", err, f)
        if self.hp["attn"] in ("census", "position-only"):
            self._teach_heads(x, y)
        fb = self.hp["feedback"]
        if fb != "frozen":
            e_exact = None
            for j in reversed(range(len(self.mlps))):
                mlp, (B_hid, B_out) = self.mlps[j], self.B[j]
                u, a = self.caught[("u", j)], self.caught[("a", j)]
                h = F.gelu(a)
                if fb == "dfa":
                    e_out = err @ B_out
                    if self.hp["ln_aware"]:  # LayerNorm ignores the size of the stream, so drop the "grow" part
                        e_out = self._ln_back(e_out, self.caught["s"], None)
                    e_hid = (err @ B_hid) * _gelu_grad(a)
                elif fb == "exact":  # residual stream: every MLP sees the same stream error (diagnostic, 1 MLP exact)
                    if e_exact is None:
                        e_exact = self._ln_back(err @ self.model.lm_head.weight, self.caught["s"], self.model.ln_f)
                    e_out = e_exact
                    e_hid = (e_out @ mlp.proj.weight) * _gelu_grad(a)
                else:
                    raise ValueError(fb)
                mlp.proj.weight.grad = torch.einsum("btc,bth->ch", e_out, h)
                mlp.proj.bias.grad = e_out.sum((0, 1))
                mlp.fc.weight.grad = torch.einsum("bth,btc->hc", e_hid, u)
                mlp.fc.bias.grad = e_hid.sum((0, 1))
        self.opt.step()
        self.forwards += 1
        return loss.item()
