import math

import pytest
import torch

from bbp.arena import run
from bbp.guard import BackpropDetected, no_backprop
from bbp.model import GPT, GPTConfig
from bbp.rules import RULES, Rule, register

TINY = dict(vocab_size=11, block_size=16, n_layer=2, n_head=2, n_embd=32)


def batch(cfg, B=4):
    g = torch.Generator().manual_seed(0)
    x = torch.randint(cfg.vocab_size, (B, cfg.block_size), generator=g)
    return x, torch.roll(x, -1, 1)


@pytest.mark.parametrize("name", sorted(RULES))
def test_rule_steps_and_learns_something(name):
    torch.manual_seed(0)
    cfg = GPTConfig(**TINY)
    model = GPT(cfg)
    rule = RULES[name](model)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    x, y = batch(cfg)
    for _ in range(4):
        if rule.autograd == "none":
            with no_backprop(name):
                loss = rule.step(x, y)
        else:
            loss = rule.step(x, y)
        assert isinstance(loss, float) and math.isfinite(loss)
    assert rule.forwards >= 4
    changed = any(not torch.equal(before[k], v) for k, v in model.named_parameters())
    if name != "hill_climb":  # hill climbing may legitimately reject every jiggle
        assert changed, f"{name} never touched the weights"
    # the model's own forward still works after the rule rewired it
    _, l = model(x, y)
    assert math.isfinite(l.item())


def test_guard_catches_a_cheater():
    class Cheater(Rule):
        name, autograd = "cheater", "none"

        def step(self, x, y):
            self.model(x, y)[1].backward()
            return 0.0

    cfg = GPTConfig(**TINY)
    r = Cheater(GPT(cfg))
    with pytest.raises(BackpropDetected):
        with no_backprop("cheater"):
            r.step(*batch(cfg))
    # and the guard un-patches itself
    t = torch.ones(1, requires_grad=True)
    (t * 2).sum().backward()
    assert t.grad.item() == 2.0


def test_guard_catches_autograd_grad():
    with pytest.raises(BackpropDetected):
        with no_backprop("x"):
            t = torch.ones(1, requires_grad=True)
            torch.autograd.grad((t * 2).sum(), t)


def test_arena_end_to_end_cpu():
    res = run("backprop", minutes=1, device="cpu", hp={}, cfg=GPTConfig(block_size=16, n_layer=1, n_embd=32, n_head=2),
              batch_size=4, max_steps=20, save=False, quiet=True)
    assert res["steps"] == 20 and math.isfinite(res["test_loss"]) and len(res["sample"]) > 300
    assert res["test_loss"] < res["init_val"]
    assert math.isfinite(res["unigram_val_loss"])
    assert math.isfinite(res["unigram_test_loss"])


def test_every_rule_is_honestly_labelled():
    for name, cls in RULES.items():
        assert cls.blurb and cls.autograd


@pytest.mark.parametrize("attn", ["census", "position-only"])
@pytest.mark.parametrize("prior,beta,sharpen", [("self", 1.0, 1.0), ("self", 0.7, 1.4), ("uniform", 0.7, 1.4)])
def test_census_heads_update_is_the_local_attention_gradient(attn, prior, beta, sharpen):
    """census_heads' hand-written Q/K update == autograd of CE(EM posterior, attention), prior held fixed."""
    from bbp.rules.census_heads import CensusHeads
    torch.manual_seed(0)
    cfg = GPTConfig(**TINY)
    m = GPT(cfg)
    r = CensusHeads(m, max_offset=3, attn=attn, prior=prior, beta=beta, sharpen=sharpen)
    x, y = batch(cfg)
    for _ in range(3):
        r.step(x, y)
    r._refresh()
    with torch.no_grad():
        m(x)
        r._teach_heads(x, y)
    W = m.blocks[0].attn.qkv.weight
    manual = W.grad.clone()
    manual_bias = m.blocks[0].attn.qkv.bias.grad.clone()
    z = r.caught["z"].detach()
    b = m.blocks[0].attn.qkv.bias.detach().clone().requires_grad_(True)
    W2 = W.detach().clone().requires_grad_(True)
    B, T, C, hd = x.size(0), cfg.block_size, cfg.n_embd, r.hd
    ar = torch.arange(T)
    offs = ar[:, None] - ar[None, :]
    allowed = (offs >= 0) & (offs <= 3)
    L = torch.zeros((B, T, T)).masked_fill(~allowed, float("-inf"))
    if attn == "census":
        for d in range(1, 4):
            t = ar[d:]
            L[:, t, t - d] = r.table[d - 1][x[:, :-d], x[:, d:], y[:, d:]]
    loss = 0.0
    for h in range(r.nh):
        q = z @ W2[h * hd:(h + 1) * hd].T + b[h * hd:(h + 1) * hd]
        k = z @ W2[C + h * hd:C + (h + 1) * hd].T + b[C + h * hd:C + (h + 1) * hd]
        S = ((q @ k.transpose(1, 2)) / math.sqrt(hd)).masked_fill(offs < 0, float("-inf"))
        a = S.softmax(-1)
        log_prior = sharpen * (a.detach() + 1e-9).log() if prior == "self" else 0.0
        tgt = (log_prior + beta * L.masked_fill((offs >= 1) & (offs <= h), float("-inf"))).softmax(-1)
        loss = loss - (tgt * S.log_softmax(-1).masked_fill(offs < 0, 0)).sum() / (B * T)
    loss.backward()
    assert torch.allclose(W2.grad, manual, atol=1e-6, rtol=1e-4)
    assert torch.allclose(b.grad, manual_bias, atol=1e-6, rtol=1e-4)


def test_backdrop_leaves_forward_alone_and_thins_lower_gradients():
    torch.manual_seed(0)
    cfg = GPTConfig(**TINY)
    x, y = batch(cfg)
    ref, bd = GPT(cfg), GPT(cfg)
    bd.load_state_dict(ref.state_dict())
    RULES["backdrop"](bd, p=0.5)
    l0, l1 = ref(x, y)[1], bd(x, y)[1]
    assert torch.allclose(l0, l1)
    l0.backward(); l1.backward()
    top = lambda m: m.lm_head.weight.grad
    assert torch.allclose(top(ref), top(bd))  # above every mask: exact gradient
    low = lambda m: m.blocks[0].mlp.fc.weight.grad
    assert not torch.allclose(low(ref), low(bd))  # below a mask: thinned


def test_mirror_shortcut_equals_a_separate_adamw_on_the_feedback():
    """The in-place mirror update must equal giving B its own AdamW fed W's gradient."""
    torch.manual_seed(0)
    cfg = GPTConfig(**TINY)
    x, y = batch(cfg)
    rule = RULES["mirror_neurons"](GPT(cfg), weight_decay=0.1, lr=3e-3)
    B0 = [b.detach().clone().requires_grad_(True) for b in rule.B]
    ref = torch.optim.AdamW(B0, lr=3e-3, weight_decay=0.1)
    for _ in range(5):
        rule.step(x, y)  # W's gradient from this step is still in .grad afterwards
        for b, w in zip(B0, rule.W):
            b.grad = w.grad.clone()
        ref.step()
    for b, r in zip(rule.B, B0):
        assert torch.allclose(b, r, atol=1e-6)


def test_counting_teacher_leave_one_out_stops_memorising():
    from bbp.rules._teacher import CountingTeacher
    torch.manual_seed(0)
    text = torch.randint(5, (1, 400))  # random text: no real structure to generalise from
    x, y = text[:, :-1], text[:, 1:]
    T = CountingTeacher(5, order=9, beta=1.0)
    T.observe(x, y)
    T.build()
    p_mem = T(x, y, loo=False).gather(-1, y[..., None]).mean().item()
    p_loo = T(x, y, loo=True).gather(-1, y[..., None]).mean().item()
    assert p_mem > 0.5  # with itself counted, long contexts are unique and give the answer away
    assert p_loo < 0.35  # without itself, close to chance (0.2) on structureless text
    assert torch.allclose(T(x, y).sum(-1), torch.ones(x.shape))


def test_teach_hook_counts_then_teaches():
    torch.manual_seed(0)
    cfg = GPTConfig(**TINY)
    rule = RULES["mirror_neurons"](GPT(cfg), teach=0.5, count_steps=2, order=3)
    x, y = batch(cfg)
    for _ in range(4):
        assert math.isfinite(rule.step(x, y))
    assert rule.teacher.ready


def test_dreams_are_appended_and_trained_on():
    torch.manual_seed(0)
    cfg = GPTConfig(**TINY)
    rule = RULES["backprop"](GPT(cfg), teach=0.5, count_steps=2, order=3, dream=0.5, dream_streams=8, dream_len=64)
    x, y = batch(cfg)
    for _ in range(4):
        assert math.isfinite(rule.step(x, y))
    assert rule.dreams.shape == (8, 64) and rule.xd.shape == (2, cfg.block_size)


def test_mirror_born_as_copy_is_exact_backprop():
    torch.manual_seed(0)
    cfg = GPTConfig(**TINY)
    x, y = batch(cfg)
    ref = GPT(cfg)
    mir = GPT(cfg)
    mir.load_state_dict(ref.state_dict())
    rule = RULES["mirror_neurons"](mir, born="copy")
    bp = RULES["backprop"](ref, lr=rule.hp["lr"], weight_decay=rule.hp["weight_decay"])
    for _ in range(3):
        rule.step(x, y)
        bp.step(x, y)
    for (k, a), b in zip(ref.named_parameters(), mir.parameters()):
        assert torch.allclose(a, b, atol=1e-5), k


def test_kneser_ney_teacher_is_a_distribution():
    from bbp.rules._teacher import CountingTeacher
    torch.manual_seed(0)
    text = torch.randint(5, (2, 300))
    x, y = text[:, :-1], text[:, 1:]
    T = CountingTeacher(5, order=4, smoothing="kn")
    T.observe(x, y)
    T.build()
    for loo in (True, False):
        p = T(x, y, loo=loo)
        assert (p >= 0).all() and torch.allclose(p.sum(-1), torch.ones(x.shape), atol=1e-5)


def test_proxy_split_holds_out_a_play_and_never_touches_val_or_test():
    import bbp.proxy as proxy
    from bbp.data import DATA_DIR, CharData
    if not (DATA_DIR / "tinyshakespeare.txt").exists():
        pytest.skip("needs the dataset")
    real = CharData(16, "cpu")
    p = proxy.ProxyData(16, "cpu")
    n = sum(len(v) for v in real.splits.values())
    assert len(p.splits["train"]) + len(p.splits["val"]) == len(real.splits["train"])
    lo = int(n * proxy.LO)
    assert torch.equal(p.splits["val"][:100], real.splits["train"][lo:lo + 100])
