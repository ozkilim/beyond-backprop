"""GRADIENT-CENTRALIZED MIRROR: mirror neurons + gradient centralization.

Gradient Centralization (GC) is a simple technique from Yong et al. 2020:
subtract the mean of each row of the gradient before the optimizer step.
This constrains the weight updates to have zero mean, which acts as a
projected gradient descent regularizer.

GC has been shown to:
  - Improve generalization by constraining the solution space
  - Act as a form of implicit weight decay on output biases
  - Reduce overfitting without additional hyperparameters

Combined with mirror neurons + decaying noise + teacher, this is a
three-pronged attack:
  - Teacher: better targets (data-level improvement)
  - Mirror + noise: better credit assignment (gradient direction)
  - GC: better optimization surface (gradient magnitude/distribution)

Each addresses a different axis. All three are cheap (essentially free
per-step overhead).
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class GradientCentralizedMirror(Rule):
    name = "gradient_centralized_mirror"
    autograd = "modified"
    blurb = "Mirror neurons + gradient centralization: zero-mean gradient rows as implicit regularization."
    paper = "Novel combination: Kolen-Pollack mirror + Yong 2020 GC + counting teacher."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma0=0.5, gc_strength=1.0,
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

    def _sigma(self):
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        return self.hp["sigma0"] * (1.0 - t)

    def step(self, x, y):
        self.step_count += 1
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

        # === GRADIENT CENTRALIZATION ===
        gc = self.hp["gc_strength"]
        if gc > 0:
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.grad is not None and p.grad.dim() >= 2:
                        # Subtract row mean (centralize)
                        p.grad.sub_(p.grad.mean(dim=-1, keepdim=True) * gc)

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
