"""SLINGSHOT: periodically catapult weights past the current minimum to escape local basins.

Inspired by gravitational slingshots: the optimizer follows the gradient most of
the time, but every `sling_every` steps, the model's weights are "flung" forward
along the recent trajectory by a factor `sling_strength`. This overshoots the
current minimum, potentially landing in a wider basin on the other side.

The slingshot direction is the difference between current weights and an
exponential-moving-average "anchor". After the sling, the anchor is reset.

This creates a rhythmic explore-exploit cycle:
  - Exploit: normal gradient descent toward the nearest minimum
  - Explore: slingshot past it, landing somewhere new
  - Exploit: descend from the new landing spot

If the new basin is better (wider, lower), the model stays. If worse, the
model quickly returns. Combined with mirror neurons and the counting teacher.

The key insight: backprop gets stuck in the FIRST good minimum it finds.
Slingshot gives the model many chances to find a BETTER one.
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class Slingshot(Rule):
    name = "slingshot"
    autograd = "modified"
    blurb = "Mirror neurons with periodic weight slingshots: fling weights past the current minimum along the recent trajectory to escape local basins and find wider ones."
    paper = "Novel: gravitational slingshot applied to weight-space exploration."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0, mirror=True,
                    sling_every=500, sling_strength=2.0, anchor_ema=0.995,
                    teach=0.0, beta=16.0, order=9, count_steps=1500)
    tune = "sling_every"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        swap_linears(model, "random", self.hp["scale"])
        self.twins = [m for m in model.modules() if isinstance(m, FALinear)]
        self.opt = torch.optim.AdamW(model.parameters(),
                                     lr=self.hp["lr"],
                                     weight_decay=self.hp["weight_decay"])
        self.W = [m.weight for m in self.twins]
        self.B = [m.feedback for m in self.twins]
        # All parameters, not just FALinear weights
        self.all_params = [p for p in model.parameters() if p.requires_grad]
        # Anchor: EMA of weights (the "planet" we slingshot around)
        self.anchors = [p.detach().clone() for p in self.all_params]
        self._step_count = 0

    def step(self, x, y):
        self._step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            before = [w.clone() for w in self.W]

        self.opt.step()

        # Mirror update
        if self.hp["mirror"]:
            with torch.no_grad():
                lr_wd = self.opt.param_groups[0]["lr"] * self.hp["weight_decay"]
                for w, b, bef in zip(self.W, self.B, before):
                    gap = b - bef
                    b.add_(w - bef)
                    b.add_(gap, alpha=-lr_wd)

        with torch.no_grad():
            # Update anchors (EMA)
            alpha = self.hp["anchor_ema"]
            for a, p in zip(self.anchors, self.all_params):
                a.mul_(alpha).add_(p, alpha=1.0 - alpha)

            # Slingshot: fling weights past the anchor
            if self._step_count % int(self.hp["sling_every"]) == 0 and self._step_count > 100:
                s = self.hp["sling_strength"]
                for a, p in zip(self.anchors, self.all_params):
                    # Trajectory: current - anchor
                    traj = p - a
                    # Fling forward
                    p.add_(traj, alpha=s)
                # Reset anchors to current position
                for a, p in zip(self.anchors, self.all_params):
                    a.copy_(p)

        self.forwards += 1
        return loss.item()

    @torch.no_grad()
    def alignment(self):
        return sum(torch.cosine_similarity(m.weight.flatten(), m.feedback.flatten(), 0).item()
                   for m in self.twins) / len(self.twins)
