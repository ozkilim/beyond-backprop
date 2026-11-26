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


@register
class TaughtMirrorLayers(TaughtLayers):
    """mirror_neurons credit end to end (backward weights born random, given the same update as
    their forward twins; no weight is ever copied, including in the per-block readouts), plus the
    same depth-suited per-block counting tables as taught_layers. End-to-end backward pass with
    learned stand-in weights, so 'modified'. Control: taught_layers_bp (exact backprop, same tables)."""
    name = "taught_mirror_layers"
    autograd = "modified"
    blurb = "Mirror-neuron credit (no weight transport) plus a depth-suited counting table teaching every block's readout."
    paper = "Kolen & Pollack 1994; Witten & Bell 1991; this repo"
    defaults = dict(TaughtLayers.defaults, lr=1e-3)
    local = False

    def __init__(self, model, **hp):
        from .feedback_alignment import FALinear, swap_linears
        super().__init__(model, **hp)
        swap_linears(model, "random", 1.0)
        swap_linears(self.aux, "random", 1.0)
        twins = [m for m in list(model.modules()) + list(self.aux.modules()) if isinstance(m, FALinear)]
        self.W, self.B = [m.weight for m in twins], [m.feedback for m in twins]
        params = list(model.parameters()) + list(self.aux.parameters())
        self.opt = torch.optim.AdamW(params, lr=self.hp["lr"], weight_decay=self.hp["weight_decay"])
        step = self.opt.step

        def mirrored_step(*a, **k):  # each B gets exactly its forward twin's AdamW update (see mirror_neurons)
            with torch.no_grad():
                before = torch._foreach_mul(self.W, 1.0)
            out = step(*a, **k)
            with torch.no_grad():
                lr_wd = self.opt.param_groups[0]["lr"] * self.hp["weight_decay"]
                gap = torch._foreach_sub(self.B, before)
                torch._foreach_add_(self.B, torch._foreach_sub(self.W, before))
                torch._foreach_add_(self.B, gap, alpha=-lr_wd)
            return out
        self.opt.step = mirrored_step
