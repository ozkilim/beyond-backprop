"""OUROBOROS: Stochastic Weight Averaging during mirror-neuron training.

Maintains an exponential moving average (EMA) of the weights. Every K steps,
the actual weights are SNAPPED toward the EMA:
    W = (1-α) * W + α * EMA_W

This is a form of online SWA (Izmailov et al. 2018), but applied during
training rather than at the end. It acts as an implicit ensemble:
the model's weights at any point are a weighted average of all past
weight configurations, biased toward recent ones.

Combined with mirror neurons, this means:
  - The backward weights B track W (mirror)
  - W itself is periodically pulled toward an averaged version of W
  - The noise on B provides exploration around this averaged point

The EMA provides a "gravity well" that prevents the weights from wandering
too far into overfitting territory. The mirror noise provides local
exploration around this stable point. The teacher provides better targets.

Three stabilising forces at three levels:
  1. Teacher: stable TARGETS
  2. EMA snap: stable WEIGHTS
  3. Mirror: stable CREDIT ASSIGNMENT
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class Ouroboros(Rule):
    name = "ouroboros"
    autograd = "modified"
    blurb = "Online SWA + mirror neurons: EMA weight snapping as implicit ensemble regularization."
    paper = "Novel combination: online SWA (Izmailov 2018) + Kolen-Pollack mirror + counting teacher."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma0=0.5,
                    ema_decay=0.999,     # EMA decay for weight averaging
                    snap_every=50,       # steps between weight snaps
                    snap_alpha=0.3,      # how much to pull toward EMA
                    snap_warmup=1500,    # steps before snapping starts
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

        # EMA shadow weights
        self.ema_params = {name: p.detach().clone()
                          for name, p in model.named_parameters()
                          if p.requires_grad}

    def _sigma(self):
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        return self.hp["sigma0"] * (1.0 - t)

    @torch.no_grad()
    def _update_ema(self):
        decay = self.hp["ema_decay"]
        for name, p in self.model.named_parameters():
            if name in self.ema_params:
                self.ema_params[name].mul_(decay).add_(p.data, alpha=1.0 - decay)

    @torch.no_grad()
    def _snap_to_ema(self):
        """Pull current weights toward EMA."""
        if self.step_count < self.hp["snap_warmup"]:
            return
        if self.step_count % self.hp["snap_every"] != 0:
            return
        alpha = self.hp["snap_alpha"]
        # Decay snap strength over time (less snapping as model stabilizes)
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        alpha = alpha * (1.0 - t * 0.5)
        for name, p in self.model.named_parameters():
            if name in self.ema_params:
                p.data.lerp_(self.ema_params[name], alpha)

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
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        # Restore B
        with torch.no_grad():
            for b, sb in zip(self.B, saved_B):
                b.copy_(sb)
            before = [w.clone() for w in self.W]

        self.opt.step()

        # Update EMA and periodically snap
        self._update_ema()
        self._snap_to_ema()

        # Mirror update
        with torch.no_grad():
            lr_wd = self.opt.param_groups[0]["lr"] * self.hp["weight_decay"]
            for w, b, bef in zip(self.W, self.B, before):
                gap = b - bef
                b.add_(w - bef)
                b.add_(gap, alpha=-lr_wd)

        self.forwards += 1
        return loss.item()
