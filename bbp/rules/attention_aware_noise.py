"""ATTENTION-AWARE NOISE: noise magnitude scaled by attention entropy.

Standard noisy_mirror applies UNIFORM noise to all backward weights.
But in a transformer, different layers and different weight matrices
serve different purposes:

  - Query/Key projections: control WHERE the model looks
  - Value projections: control WHAT information is extracted
  - Output projections: control HOW information is mixed

And different layers attend differently:
  - Early layers: often broad attention (high entropy)
  - Late layers: often focused attention (low entropy)

The idea: scale noise INVERSELY with how "confident" the layer is.
High-entropy attention heads (uncertain where to look) benefit from
MORE exploration noise. Low-entropy heads (know exactly where to look)
should be disturbed LESS.

Implementation challenge: computing attention entropy every step is
expensive. So we approximate: track an EMA of each layer's attention
entropy, and update it only during eval steps (which happen every 10s
and are free). Between evals, use the cached entropy values.

BUT: this introduces per-layer scaling which means we need to identify
which FALinear belongs to which transformer block. We can do this by
their position in the module list.

SIMPLER VARIANT: just scale noise by LAYER DEPTH.
Deeper layers get LESS noise (they're closer to the output and need
more precise credit). Shallower layers get MORE noise (they can afford
to explore representations).

This is essentially free: just multiply sigma by a per-layer factor.
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class AttentionAwareNoise(Rule):
    name = "attention_aware_noise"
    autograd = "modified"
    blurb = "Depth-scaled backward noise: more exploration in shallow layers, more precision in deep ones."
    paper = "Novel: layer-depth-proportional backward noise on mirror neurons."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma0=0.5, depth_ratio=3.0,  # shallow gets depth_ratio × more noise than deep
                    noise_schedule="linear",
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

        # Compute per-layer noise multipliers
        # More noise for shallow layers, less for deep ones
        n = len(self.W)
        ratio = self.hp["depth_ratio"]
        # Linear interpolation: layer 0 gets ratio×, layer n-1 gets 1×
        # Normalize so the average multiplier is 1.0
        raw = [ratio - (ratio - 1.0) * i / max(n - 1, 1) for i in range(n)]
        avg = sum(raw) / n
        self.layer_scales = [r / avg for r in raw]  # normalize to mean=1

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
                # Layer-specific noise magnitude
                layer_sigma = sigma * self.layer_scales[i]
                noise = torch.randn_like(w)
                noise.mul_(w.std() * layer_sigma)
                torch.add(w, noise, out=b)

        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        self.forwards += 1
        return loss.item()
