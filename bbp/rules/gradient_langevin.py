"""GRADIENT LANGEVIN: add noise DIRECTLY to gradients instead of backward weights.

This is the control experiment for noisy_mirror. If the win comes from the
noise acting as regularization on the GRADIENT (like Langevin dynamics / noisy
SGD), then adding noise directly to gradients should work just as well — and
it's CHEAPER because there's no FALinear overhead at all.

If noisy_mirror wins over this, it proves the backward-weight noise has
a STRUCTURAL effect beyond simple gradient noise — the noise is filtered
through the activation patterns, creating sample-dependent perturbations.

Standard backprop + teacher + decaying gradient noise:
  g_t = ∇L(W_t) + σ(t) * ε,  ε ~ N(0, I)
  W_{t+1} = W_t - lr * g_t

This is essentially stochastic gradient Langevin dynamics (SGLD, Welling &
Teh 2011) but with the temperature annealed to zero.

No FALinear, no mirror tracking, no backward weight manipulation.
Just noise on gradients. Same sigma schedule as noisy_mirror for fair comparison.
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach


@register
class GradientLangevin(Rule):
    name = "gradient_langevin"
    autograd = "full"
    blurb = "Standard backprop + decaying gradient noise (Langevin dynamics). Control for noisy_mirror."
    paper = "Welling & Teh 2011: Bayesian Learning via Stochastic Gradient Langevin Dynamics."
    defaults = dict(lr=1e-3, weight_decay=0.01,
                    sigma0=0.5, noise_schedule="linear",
                    teach=0.3, beta=16.0, order=9, count_steps=1500)
    tune = "sigma0"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        # No FALinear! Standard model, standard backprop.
        self.opt = torch.optim.AdamW(model.parameters(),
                                     lr=self.hp["lr"],
                                     weight_decay=self.hp["weight_decay"])
        self.step_count = 0
        self.est_total_steps = 20000  # faster than mirror, more steps

    def _sigma(self):
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        return self.hp["sigma0"] * (1.0 - t)

    def step(self, x, y):
        self.step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        # Add decaying noise to gradients (Langevin dynamics)
        sigma = self._sigma()
        if sigma > 0:
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.grad is not None:
                        noise = torch.randn_like(p.grad)
                        noise.mul_(p.grad.std() * sigma)
                        p.grad.add_(noise)

        self.opt.step()
        self.forwards += 1
        return loss.item()
