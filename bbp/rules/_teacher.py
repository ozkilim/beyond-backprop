"""A counting teacher: a smoothed n-gram table built only from the batches a rule sees,
used as soft targets. No gradients anywhere in it.

Phase 1 (`count_steps` steps): every window position with at least `KEY` characters of
history is recorded once, deduplicated by a hash of those characters, so overlapping
windows don't count the same spot in the text twice. Then one table per context length
0..order-1 is built (dense counts over the next character), all inside step(), so the
arena's clock pays for it.

Phase 2: for each position in a batch the teacher gives the Witten-Bell backoff
distribution over the next character, LEAVE-ONE-OUT: the position's own occurrence is
subtracted, so the teacher says what the table would predict for text it had not seen,
not what it memorised. `beta` scales the backoff weight (bigger = smoother).
"""
import torch

KEY = 24  # chars of history that identify a position in the text


class CountingTeacher:
    def __init__(self, V, order=9, beta=1.0, smoothing="wb", disc=1.0):
        self.V, self.order, self.beta = V, order, beta
        self.smoothing, self.disc = smoothing, disc  # "wb" Witten-Bell (beta) or "kn" Kneser-Ney (disc)
        self.ctx, self.nxt, self.keys = [], [], []
        self.ready = False
        self.mult = None

    @torch.no_grad()
    def _hash(self, x, end, length):
        if self.mult is None:
            g = torch.Generator().manual_seed(0)
            self.mult = (torch.randint(1, 2**62, (KEY,), generator=g) | 1).to(x.device)
        h = torch.zeros(x.shape[0], end.numel(), dtype=torch.long, device=x.device)
        for i in range(length):
            h = h * 1_000_003 + x[:, end - i] * self.mult[i]
        return h

    @torch.no_grad()
    def observe(self, x, y):
        T = x.shape[1]
        k = min(KEY, T)  # tiny test windows can be shorter than KEY
        end = torch.arange(k - 1, T, device=x.device)
        self.keys.append(self._hash(x, end, k).reshape(-1))
        o = self.order - 1
        self.ctx.append(torch.stack([x[:, end - i] for i in range(o)], -1).reshape(-1, o).to(torch.int16))
        self.nxt.append(y[:, end].reshape(-1).to(torch.int16))

    @torch.no_grad()
    def build(self):
        keys = torch.cat(self.keys)
        _, inv = torch.unique(keys, return_inverse=True)
        first = torch.full((int(inv.max()) + 1,), keys.numel(), device=keys.device, dtype=torch.long)
        first.scatter_reduce_(0, inv, torch.arange(keys.numel(), device=keys.device), "amin")
        ctx = torch.cat(self.ctx)[first].long()  # (N, order-1): ctx[:, i] = char i back
        nxt = torch.cat(self.nxt)[first].long()
        self.ctx = self.nxt = self.keys = None
        V = self.V
        self.tabs = []  # per context length n: (sorted ctx keys, dense counts (n_ctx, V))
        rows = []
        for n in range(self.order):
            ck = self._ctxkey(ctx[:, :n])
            uk, row = torch.unique(ck, return_inverse=True)
            counts = torch.zeros(uk.numel(), V, device=ck.device)
            counts.index_put_((row, nxt), torch.ones_like(nxt, dtype=counts.dtype), accumulate=True)
            self.tabs.append((uk, counts))
            rows.append(row)
        if self.smoothing == "kn":
            # continuation counts: cc[n][h, c] = how many different older chars w were seen before (h, c)
            self.cc, self.D, self.Dcc = [], [], []
            for n in range(self.order):
                if n + 1 < self.order:
                    suf = torch.zeros(self.tabs[n + 1][0].numel(), dtype=torch.long, device=ctx.device)
                    suf[rows[n + 1]] = rows[n]
                    cc = torch.zeros_like(self.tabs[n][1])
                    cc.index_add_(0, suf, (self.tabs[n + 1][1] > 0).float())
                else:
                    cc = None
                self.cc.append(cc)
                self.D.append(self._discount(self.tabs[n][1]))
                self.Dcc.append(self._discount(cc) if cc is not None else None)
        self.n_positions = nxt.numel()
        self.ready = True

    def _discount(self, c):
        n1, n2 = (c == 1).sum().float(), (c == 2).sum().float()
        return float((self.disc * n1 / (n1 + 2 * n2).clamp_min(1)).clamp(0.05, 0.99))

    @torch.no_grad()
    def _dist_kn(self, hist, avail, y=None):
        """Interpolated Kneser-Ney: the highest order a position can use counts raw, every lower
        order counts *continuations* (in how many different contexts a char followed h)."""
        V = self.V
        p = torch.full(hist.shape[:-1] + (V,), 1.0 / V, device=hist.device)
        top = avail.clamp_max(self.order - 1)
        idx, hits = [], []
        for n in range(self.order):
            uk = self.tabs[n][0]
            ck = self._ctxkey(hist[..., :n])
            i = torch.searchsorted(uk, ck).clamp_max(uk.numel() - 1)
            idx.append(i)
            hits.append((uk[i] == ck) & (avail >= n))
        for n in range(self.order):
            raw = self.tabs[n][1][idx[n]] * hits[n][..., None]
            is_top = (top == n)[..., None]
            if self.cc[n] is not None:
                cc = self.cc[n][idx[n]] * hits[n][..., None]
                c = torch.where(is_top, raw, cc)
            else:
                c = raw
            if y is not None:
                oh = torch.nn.functional.one_hot(y, V).float()
                own_raw = oh * (raw.gather(-1, y[..., None]) > 0)
                if self.cc[n] is not None:  # own occurrence is a continuation only if it was the only one with its older char
                    up = self.tabs[n + 1][1][idx[n + 1]].gather(-1, y[..., None]) * hits[n + 1][..., None]
                    own_cc = oh * (up == 1) * (cc.gather(-1, y[..., None]) > 0)
                    own = torch.where(is_top, own_raw, own_cc)
                else:
                    own = own_raw
                c = (c - own).clamp_min(0)
            D = torch.where(is_top, torch.tensor(self.D[n], device=c.device),
                            torch.tensor(self.Dcc[n] if self.Dcc[n] is not None else self.D[n], device=c.device))
            tot = c.sum(-1, keepdim=True)
            types = (c > 0).sum(-1, keepdim=True).float()
            q = ((c - D).clamp_min(0) + D * types * p) / tot.clamp_min(1e-9)
            p = torch.where((tot > 0) & (n <= top)[..., None], q, p)
        return p

    def _ctxkey(self, c):  # c (..., n) chars, most recent first -> int64 key (unique for n <= 9)
        k = torch.zeros(c.shape[:-1], dtype=torch.long, device=c.device)
        for i in range(c.shape[-1]):
            k = k * (self.V + 1) + c[..., i] + 1
        return k

    @torch.no_grad()
    def _dist(self, hist, avail, y=None, record=None):
        """Backoff distribution (..., V). hist[..., i] = char i back; avail = chars of history
        actually available; y = the true next char, whose own count is left out (LOO).
        record: a list of context lengths; returns {m: distribution using at most m chars}."""
        if self.smoothing == "kn":
            assert record is None
            return self._dist_kn(hist, avail, y)
        rec = {}
        V = self.V
        p = torch.full(hist.shape[:-1] + (V,), 1.0 / V, device=hist.device)
        onehot = torch.nn.functional.one_hot(y, V).float() if y is not None else None
        for n in range(self.order):
            uk, counts = self.tabs[n]
            ck = self._ctxkey(hist[..., :n])
            i = torch.searchsorted(uk, ck).clamp_max(uk.numel() - 1)
            hit = (uk[i] == ck) & (avail >= n)
            c = counts[i] * hit[..., None]
            if onehot is not None:
                c = (c - onehot * (c.gather(-1, y[..., None]) > 0)).clamp_min(0)
            tot = c.sum(-1, keepdim=True)
            types = (c > 0).sum(-1, keepdim=True).float()
            lam = self.beta * types
            q = (c + lam * p) / (tot + lam).clamp_min(1e-9)
            p = torch.where(tot > 0, q, p)
            if record is not None and n in record:
                rec[n] = p
        if record is not None:
            return {m: rec.get(min(m, self.order - 1), p) for m in record}
        return p

    @torch.no_grad()
    def multi(self, x, y, ctxs, loo=True):
        """One pass, several shorter-memory teachers: {m: soft targets using at most m chars}."""
        hist = torch.stack([torch.roll(x, i, 1) for i in range(self.order - 1)], -1)
        avail = (torch.arange(x.shape[1], device=x.device) + 1).expand(x.shape)
        return self._dist(hist, avail, y if loo else None, record=list(ctxs))

    @torch.no_grad()
    def __call__(self, x, y, loo=True, max_ctx=None):
        """Soft targets (B, T, V) for the next character at every position of x.
        max_ctx: use at most this many characters of context (a shorter-memory teacher)."""
        hist = torch.stack([torch.roll(x, i, 1) for i in range(self.order - 1)], -1)  # char i back
        avail = torch.arange(x.shape[1], device=x.device) + 1
        if max_ctx is not None:
            avail = avail.clamp_max(max_ctx)
        return self._dist(hist, avail.expand(x.shape), y if loo else None)

    @torch.no_grad()
    def dream(self, streams, length, device):
        """Sample `streams` texts of `length` chars from the table, all streams in parallel."""
        g = torch.Generator(device=device).manual_seed(0)
        hist = torch.zeros(streams, self.order - 1, dtype=torch.long, device=device)
        out = torch.empty(streams, length, dtype=torch.long, device=device)
        for t in range(length):
            p = self._dist(hist, torch.full((streams,), t, device=device))
            c = torch.multinomial(p, 1, generator=g)[:, 0]
            out[:, t] = c
            hist = torch.cat([c[:, None], hist[:, :-1]], 1)
        return out


def add_dreams(rule, x, y):
    """Optional (hp dream > 0, needs teach > 0): once the table is built, append
    dream*batch sequences sampled from it to the batch. Returns the batch to train on."""
    T = getattr(rule, "teacher", None)
    nd = int(round(rule.hp.get("dream", 0.0) * x.shape[0]))
    rule.n_real = x.shape[0]
    if not nd or T is None or not T.ready:
        return x, y
    if getattr(rule, "dreams", None) is None:  # one dream corpus, timed like everything else
        rule.dreams = T.dream(int(rule.hp.get("dream_streams", 1024)), int(rule.hp.get("dream_len", 1024)), x.device)
    D, L = rule.dreams, x.shape[1]
    s = torch.randint(D.shape[0], (nd,), device=x.device)
    o = torch.randint(D.shape[1] - L - 1, (nd,), device=x.device)
    idx = o[:, None] + torch.arange(L + 1, device=x.device)
    w = D[s[:, None], idx]
    rule.xd = w[:, :-1]
    return torch.cat([x, w[:, :-1]]), torch.cat([y, w[:, 1:]])


def teach(rule, x, y, logits, loss):
    """Optional (off unless hp teach > 0): mix the data loss with cross-entropy against the
    counting teacher, (1-teach)*CE(data) + teach*CE(teacher). x must be the clean input.
    Dream sequences appended by add_dreams are scored against the teacher alone."""
    a = rule.hp.get("teach", 0.0)
    if not a:
        return loss
    T = getattr(rule, "teacher", None)
    if T is None:
        T = rule.teacher = CountingTeacher(logits.shape[-1], int(rule.hp.get("order", 9)), rule.hp.get("beta", 1.0),
                                           rule.hp.get("smoothing", "wb"), rule.hp.get("disc", 1.0))
        rule.k_teach = 0
    if not T.ready:
        T.observe(x, y)
        rule.k_teach += 1
        if rule.k_teach >= rule.hp.get("count_steps", 1500):
            T.build()
        return loss
    nr = getattr(rule, "n_real", x.shape[0])
    lp = logits.log_softmax(-1)
    real = lp[:nr]
    ce = -real.gather(-1, y[..., None]).mean()
    kd = -(T(x, y) * real).sum(-1).mean()
    out = (1 - a) * ce + a * kd
    if lp.shape[0] > nr:
        xd = rule.xd
        yd = torch.zeros_like(xd)  # unused when loo=False
        dream = -(T(xd, yd, loo=False) * lp[nr:]).sum(-1).mean()
        out = (nr * out + (lp.shape[0] - nr) * dream) / lp.shape[0]
    return out
