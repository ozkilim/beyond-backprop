"""NOISY MIRROR+: the kitchen sink of zero-overhead improvements.

Takes the winning noisy_mirror formula and adds every free-lunch trick:

1. COSINE LR SCHEDULE: decays learning rate from lr to lr_min via cosine.
   Standard for transformers, proven to help generalization, essentially free.

2. GRADIENT CENTRALIZATION: subtract row-mean from each gradient matrix
   (Yong et al. 2020). One line, essentially free.

3. WARMUP: linear LR warmup for the first N steps. Prevents early training
   instability.

4. GRADIENT CLIPPING: clip gradient norm to prevent spikes. Free.

Each of these is well-known individually, but their COMBINATION with mirror
neurons + counting teacher + decaying backward noise is novel. The question:
do they stack?
"""
import math

import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class NoisyMirrorPlus(Rule):
    name = "noisy_mirror_plus"
    autograd = "modified"
    blurb = "Noisy mirror + cosine LR + gradient centralization + warmup + gradient clipping."
    paper = "Novel combination: noisy mirror + standard training improvements."
    defaults = dict(lr=1e-3, lr_min=1e-4, weight_decay=0.01, scale=1.0,
                    sigma0=0.5, noise_schedule="linear",
                    warmup_steps=0, grad_clip=0.0, gc=False,
                    teach=0.3, beta=16.0, order=9, count_steps=1500)
    tune = "sigma0"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        swap_linears(model, "random", self.hp["scale"])
        self.twins = [m for m in model.modules() if isinstance(m, FALinear)]
        self.opt = torch.optim.AdamW(model.parameters(),
                                     lr=self.hp["lr"],
                                     weight_decay=self.hp["weight_decay"])
        self.W = [m.weight for m in self.twins]
        self.B = [m.feedback for m in self.twins]
        with torch.no_grad():
            for w, b in zip(self.W, self.B):
                b.copy_(w)
        self.step_count = 0
        self.est_total_steps = 18000

    def _update_lr(self):
        """Cosine schedule with linear warmup."""
        warmup = self.hp["warmup_steps"]
        lr_max = self.hp["lr"]
        lr_min = self.hp["lr_min"]

        if self.step_count < warmup:
            lr = lr_max * self.step_count / max(warmup, 1)
        else:
            t = (self.step_count - warmup) / max(self.est_total_steps - warmup, 1)
            t = min(t, 1.0)
            lr = lr_min + (lr_max - lr_min) * 0.5 * (1.0 + math.cos(math.pi * t))

        for pg in self.opt.param_groups:
            pg["lr"] = lr

    def _sigma(self):
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        return self.hp["sigma0"] * (1.0 - t)

    def step(self, x, y):
        self.step_count += 1
        self._update_lr()
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        sigma = self._sigma()
        saved_B = []
        with torch.no_grad():
            for w, b in zip(self.W, self.B):
                saved_B.append(b.clone())
                noise = torch.randn_like(b) * b.std() * sigma
                b.copy_(w + noise)

        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient centralization (essentially free)
        if self.hp["gc"]:
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.grad is not None and p.grad.dim() >= 2:
                        p.grad.sub_(p.grad.mean(dim=-1, keepdim=True))

        # Gradient clipping
        clip = self.hp["grad_clip"]
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

        # Restore B
        with torch.no_grad():
            for b, sb in zip(self.B, saved_B):
                b.copy_(sb)
            before = [w.clone() for w in self.W]

        self.opt.step()

        # Mirror update
        with torch.no_grad():
            lr_wd = self.opt.param_groups[0]["lr"] * self.hp["weight_decay"]
            for w, b, bef in zip(self.W, self.B, before):
                gap = b - bef
                b.add_(w - bef)
                b.add_(gap, alpha=-lr_wd)

        self.forwards += 1
        return loss.item()
