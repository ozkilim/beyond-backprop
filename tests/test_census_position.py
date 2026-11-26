import math

import pytest
import torch

from bbp.model import GPT, GPTConfig
from bbp.rules.census_heads import CensusHeads, CensusHeadsFour, CensusMLP


@pytest.mark.parametrize("rule_cls", [CensusHeads, CensusHeadsFour, CensusMLP])
def test_position_only_is_a_matched_trainable_control(rule_cls):
    cfg = GPTConfig(vocab_size=11, block_size=8, n_layer=1, n_head=4, n_embd=48)
    torch.manual_seed(0)
    reference = rule_cls(GPT(cfg), max_offset=3, refresh=2)
    torch.manual_seed(0)
    control = rule_cls(GPT(cfg), attn="position-only", max_offset=3, refresh=2)
    for name, value in control.model.state_dict().items():
        assert torch.equal(value, reference.model.state_dict()[name]), name

    g = torch.Generator().manual_seed(1)
    x = torch.randint(cfg.vocab_size, (2, cfg.block_size), generator=g)
    y = torch.roll(x, -1, 1)
    qkv = control.model.blocks[0].attn.qkv.weight
    before = qkv.detach().clone()
    assert math.isfinite(control.step(x, y))
    reference.step(x, y)
    # The first readout update sees the same activations and uses the same optimizer.
    assert torch.equal(control.model.lm_head.weight, reference.model.lm_head.weight)
    for h in range(control.nh):
        q = slice(h * control.hd, (h + 1) * control.hd)
        k = slice(cfg.n_embd + h * control.hd, cfg.n_embd + (h + 1) * control.hd)
        assert not torch.equal(qkv[q], before[q]), f"head {h} query stayed frozen"
        assert not torch.equal(qkv[k], before[k]), f"head {h} key stayed frozen"
    assert torch.equal(qkv[2 * cfg.n_embd:], before[2 * cfg.n_embd:])

    first_table = control.table.clone()
    x, y = x.flip(1), (y + 1) % cfg.vocab_size
    control.step(x, y)
    reference.step(x, y)
    assert torch.equal(control.table, first_table)
    control.step(x, y)
    reference.step(x, y)
    assert not torch.equal(control.table, first_table)
    assert control.k == reference.k == 3
    assert control.forwards == reference.forwards == 3
    assert control.c1.sum().item() == 3 * x.numel()
    assert torch.equal(control.c1, reference.c1)
    assert torch.equal(control.c2, reference.c2)
    assert torch.equal(control.table, reference.table)


def test_position_only_head_update_ignores_census_table_and_labels():
    torch.manual_seed(2)
    cfg = GPTConfig(vocab_size=11, block_size=8, n_layer=1, n_head=2, n_embd=32)
    model = GPT(cfg)
    rule = CensusHeads(model, attn="position-only", max_offset=3)
    x = torch.randint(cfg.vocab_size, (2, cfg.block_size))
    y = torch.roll(x, -1, 1)
    qkv = model.blocks[0].attn.qkv
    with torch.no_grad():
        model(x)
        rule.table.normal_()
        rule._teach_heads(x, y)
        expected_weight, expected_bias = qkv.weight.grad.clone(), qkv.bias.grad.clone()

        rule.table.normal_(mean=4.0, std=3.0)
        rule._teach_heads(x, y)
        assert torch.equal(qkv.weight.grad, expected_weight)
        assert torch.equal(qkv.bias.grad, expected_bias)

        rule._teach_heads(x, (y + 1) % cfg.vocab_size)
        assert torch.equal(qkv.weight.grad, expected_weight)
        assert torch.equal(qkv.bias.grad, expected_bias)
