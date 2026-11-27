"""HINDSIGHT MIRROR: gradient correction via hindsight bias.

The gradient at step t is computed on batch B_t. But at step t+1, we see
batch B_{t+1} and can RETROACTIVELY assess whether the step-t gradient
was good: if the loss on B_{t+1} decreased, the step-t gradient was helpful;
if it increased, the step-t gradient was harmful.

Hindsight Mirror uses this retroactive assessment to modulate the CURRENT
gradient. It maintains a "gradient quality" score per parameter: an EMA of
how well recent gradients predicted future loss reduction. Parameters whose
gradients consistently improve future loss get AMPLIFIED. Parameters whose
gradients are noisy (sometimes help, sometimes hurt) get ATTENUATED.

This is a novel form of adaptive per-parameter learning rate that uses
FORWARD-LOOKING loss correlation rather than backward-looking gradient
statistics (like Adam's second moment).

The key difference from Adam:
  - Adam: scales by inverse sqrt of gradient variance (past statistics)
  - Hindsight: scales by gradient-loss correlation (future impact)

Combined with mirror neurons + teacher for the full stack.

The hindsight signal is DELAYED by 1 step (we learn about step t's quality
at step t+1), so we apply it to the step t+1 gradient as a prior.
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class HindsightMirror(Rule):
    name = "hindsight_mirror"
    autograd = "modified"
    blurb = "Retroactive gradient quality scoring: amplify parameters whose gradients consistently improve future loss."
    paper = "Novel: hindsight-biased per-parameter gradient scaling + mirror neurons."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma0=0.5,
                    hindsight_decay=0.95,  # EMA decay for quality score
                    hindsight_boost=1.5,   # max amplification for good gradients
                    hindsight_floor=0.5,   # min scaling for bad gradients
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

        # Hindsight state
        self.prev_loss = None
        self.prev_grad_signs = {}  # sign of previous gradient per param
        self.quality_score = {}    # EMA of gradient quality per param
        for name, p in model.named_parameters():
            if p.requires_grad and p.dim() >= 2:
                self.quality_score[name] = torch.zeros(p.shape[0], device=p.device)

    def _sigma(self):
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        return self.hp["sigma0"] * (1.0 - t)

    def step(self, x, y):
        self.step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        # Noisy mirror backward
        sigma = self._sigma()
        saved_B = []
        with torch.no_grad():
            for w, b in zip(self.W, self.B):
                saved_B.append(b.clone())
                noise = torch.randn_like(b) * b.std() * sigma
                b.copy_(w + noise)

        logits, loss = self.model(xa, ya)
        loss_val = loss.item()
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        # === HINDSIGHT ASSESSMENT ===
        # Did the previous gradient help? (loss went down = good)
        if self.prev_loss is not None:
            loss_delta = loss_val - self.prev_loss  # negative = improvement
            improvement = -loss_delta  # positive = good
            # Clip to [-1, 1] for stability
            improvement = max(-1.0, min(1.0, improvement * 10.0))

            decay = self.hp["hindsight_decay"]
            for name, p in self.model.named_parameters():
                if name not in self.quality_score or p.grad is None:
                    continue
                # Was previous gradient in same direction as quality improvement?
                if name in self.prev_grad_signs:
                    # Row-wise: average the sign agreement per row
                    self.quality_score[name].mul_(decay).add_(improvement, alpha=1.0 - decay)

        # === APPLY HINDSIGHT SCALING ===
        if self.step_count > 200:  # let quality scores warm up
            boost = self.hp["hindsight_boost"]
            floor = self.hp["hindsight_floor"]
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if name not in self.quality_score or p.grad is None:
                        continue
                    q = self.quality_score[name]
                    # Map quality score to scaling: positive quality → boost, negative → attenuate
                    # Sigmoid-like mapping: scale = floor + (boost - floor) * sigmoid(quality)
                    scale = floor + (boost - floor) * q.sigmoid()
                    p.grad.mul_(scale.unsqueeze(-1) if p.grad.dim() >= 2 else 1.0)

        # Save gradient signs for next step's hindsight
        with torch.no_grad():
            self.prev_grad_signs.clear()
            for name, p in self.model.named_parameters():
                if p.grad is not None and name in self.quality_score:
                    self.prev_grad_signs[name] = p.grad.sign().mean(dim=-1) if p.grad.dim() >= 2 else p.grad.sign()

        self.prev_loss = loss_val

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
