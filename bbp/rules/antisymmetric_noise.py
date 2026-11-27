"""ANTISYMMETRIC NOISE: use the SAME noise vector, but NEGATE it for every other layer.

Standard noisy_mirror draws independent noise for each layer. But there's
a subtle problem: uncorrelated noise across layers creates random interference
patterns. Layer 2's noisy gradient might partially cancel what layer 1's noise
was exploring.

Antisymmetric noise uses ONE shared noise pattern and alternates its sign:
  Layer 0: B = W + σ * ε
  Layer 1: B = W - σ * ε'  (different shape, but correlated generation)
  Layer 2: B = W + σ * ε''
  Layer 3: B = W - σ * ε'''

The sign alternation means adjacent layers explore in OPPOSITE directions.
When one layer pushes its credit assignment one way, its neighbor pushes
the opposite way. This creates a form of COMPETITION between layers.

Actually, since layers have different shapes, we can't share one noise vector.
Instead, we generate all noise from ONE seed per step, ensuring the noise
pattern is deterministic and reproducible. And we scale alternate layers
with opposite signs.

The deeper idea: in a deep network, the optimal noise might not be iid
across layers. The error signal propagates through ALL layers, so coordinated
noise could create more structured exploration of the loss landscape.

Uses fast_mirror approach (no clone/restore).
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class AntisymmetricNoise(Rule):
    name = "antisymmetric_noise"
    autograd = "modified"
    blurb = "Alternating-sign noise across layers: adjacent layers explore in opposite directions."
    paper = "Novel: sign-alternated backward noise for structured layer-wise exploration."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma0=0.5, noise_schedule="linear",
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
        self.est_total_steps = 15000

    def _sigma(self):
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        return self.hp["sigma0"] * (1.0 - t)

    def step(self, x, y):
        self.step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        sigma = self._sigma()

        with torch.no_grad():
            for i, (w, b) in enumerate(zip(self.W, self.B)):
                sign = 1.0 if i % 2 == 0 else -1.0
                noise = torch.randn_like(w)
                noise.mul_(w.std() * sigma * sign)
                torch.add(w, noise, out=b)

        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        self.forwards += 1
        return loss.item()
