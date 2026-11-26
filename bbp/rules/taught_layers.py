"""TAUGHT LAYERS: greedy local learning (no error ever crosses a block boundary), where every
block is also taught by a counting table cut to a memory that suits its depth. With four
blocks, block 0 learns from a 2-character table, block 1 from 4, block 2 from 6, and the top
block from 8. The tables come from one leave-one-out Witten-Bell census of the batches the
rule sees (bbp/rules/_teacher.py). Each block's loss: (1-teach) CE + teach CE(its table).
Gradients stay inside each block, so 'local'."""
import torch
import torch.nn.functional as F

from . import register
from ._teacher import CountingTeacher
from .local_losses import LocalLosses


@register
class TaughtLayers(LocalLosses):
    name = "taught_layers"
    autograd = "local"
    blurb = "Local per-block learning where each block is also taught by a counting table cut to a memory suited to its depth (2/4/6/8 chars)."
    paper = "Belilovsky et al. 2019 (local losses); Witten & Bell 1991; this repo"
    defaults = dict(LocalLosses.defaults, teach=0.3, beta=16.0, count_steps=1500, ctx=(2, 4, 6, 8))

    local = True  # cut the error at every block boundary

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.teacher = CountingTeacher(model.cfg.vocab_size, 9, self.hp["beta"])
        self.k = 0

    def step(self, x, y):
        T = self.teacher
        if not T.ready:
            T.observe(x, y)
            self.k += 1
            if self.k >= self.hp["count_steps"]:
                T.build()
            return super().step(x, y)
        a, n = self.hp["teach"], len(self.model.blocks)
        ctx = list(self.hp["ctx"])[-n:] if len(self.hp["ctx"]) >= n else [None] * n
        P = T.multi(x, y, [c if c is not None else T.order - 1 for c in ctx])  # one pass for every block's table
        h = self.model.embed(x)
        losses = []
        for i, blk in enumerate(self.model.blocks):
            h = blk(h if i == 0 or not self.local else h.detach())
            logits = self.aux[i](h) if i < n - 1 else self.model.head(h)
            lp = logits.log_softmax(-1)
            ce = -lp.gather(-1, y[..., None]).mean()
            kd = -(P[ctx[i] if ctx[i] is not None else T.order - 1] * lp).sum(-1).mean()
            losses.append((1 - a) * ce + a * kd)
        self.opt.zero_grad(set_to_none=True)
        sum(losses).backward()
        self.opt.step()
        self.forwards += 1
        return losses[-1].item()


@register
class TaughtLayersBackprop(TaughtLayers):
    """The control: the same per-block readouts and depth-suited tables, but the error flows end
    to end (plain backprop through every block, deep supervision)."""
    name = "taught_layers_bp"
    autograd = "full"
    blurb = "Control for taught_layers: same per-block tables and readouts, but trained by end-to-end backprop."
    local = False
