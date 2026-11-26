"""Greedy local learning: every transformer block gets its own little
next-token readout and trains on its own loss. The input to each block is
detached, so no gradient ever crosses a block boundary.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Rule, register


@register
class LocalLosses(Rule):
    name = "local_losses"
    autograd = "local"
    blurb = "Each block predicts the next char through its own readout; gradients never cross block boundaries."
    paper = "Belilovsky et al. 2019, 'Greedy layerwise learning can scale to ImageNet'; Nokland & Eidnes 2019"
    defaults = dict(lr=2e-3, weight_decay=0.01)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        C, V = model.cfg.n_embd, model.cfg.vocab_size
        dev = next(model.parameters()).device
        # aux readouts for every block except the last, which uses the real head
        self.aux = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(C), nn.Linear(C, V, bias=False)) for _ in model.blocks[:-1]]
        ).to(dev)
        params = list(model.parameters()) + list(self.aux.parameters())
        self.opt = torch.optim.AdamW(params, lr=self.hp["lr"], weight_decay=self.hp["weight_decay"])

    def step(self, x, y):
        h = self.model.embed(x)
        losses = []
        n = len(self.model.blocks)
        for i, blk in enumerate(self.model.blocks):
            h = blk(h if i == 0 or not getattr(self, "local", True) else h.detach())  # block 0 also trains the embeddings
            logits = self.aux[i](h) if i < n - 1 else self.model.head(h)
            losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)))
        self.opt.zero_grad(set_to_none=True)
        sum(losses).backward()  # sum of independent local graphs
        self.opt.step()
        self.forwards += 1
        return losses[-1].item()
