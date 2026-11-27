"""ECHO GRADIENT: use the PREVIOUS step's gradient as this step's backward noise.

Instead of random noise on backward weights, use INFORMATION from the
previous training step. Specifically, the gradient from step t-1 tells us
something about the local loss landscape. We can use it to bias the
exploration direction at step t.

The idea: at step t, we set:
  B = W + σ(t) * normalize(grad_{t-1})

This means the backward pass is biased in the direction of the PREVIOUS
gradient. The effect:
  - If the loss landscape is consistent (same curvature direction), the
    echo amplifies the gradient — faster convergence
  - If the landscape has changed, the echo adds exploration in a
    direction that WAS useful — a reasonable prior
  - As sigma decays, the echo fades and we converge to pure backprop

This is related to:
  - Momentum (which uses previous gradients to bias the UPDATE direction)
  - Natural gradient (which uses curvature to scale the gradient)
  - Nesterov momentum (which does a lookahead)

But it's different because the previous gradient biases the CREDIT
ASSIGNMENT, not the update. It's like saying "last step, this was the
important direction — let's look harder in that direction this step."

Essentially zero overhead: we store the previous gradient (already computed),
normalize it once, and add it to B. The normalization ensures the echo
magnitude is controlled by sigma, not by the gradient scale.

Uses fast_mirror approach (no clone/restore).
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class EchoGradient(Rule):
    name = "echo_gradient"
    autograd = "modified"
    blurb = "Previous gradient as backward noise: credit assignment biased by recent loss landscape information."
    paper = "Novel: gradient echo — previous step's gradient direction as structured backward noise."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma0=0.5, noise_frac=0.3,  # fraction of noise that's echo vs random
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
        # Store previous gradients (initialized to zero)
        self.prev_grads = [torch.zeros_like(w) for w in self.W]
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
        echo_frac = self.hp["noise_frac"]

        # B = W + sigma * (echo_frac * normalized_prev_grad + (1-echo_frac) * random_noise)
        with torch.no_grad():
            for i, (w, b) in enumerate(zip(self.W, self.B)):
                # Random component
                random_noise = torch.randn_like(w)
                random_noise.mul_(w.std())

                # Echo component (normalized previous gradient)
                pg = self.prev_grads[i]
                pg_norm = pg.norm()
                if pg_norm > 1e-8:
                    echo = pg * (w.std() / pg_norm)
                else:
                    echo = random_noise  # fallback for first step

                # Mix echo and random
                noise = echo * echo_frac + random_noise * (1.0 - echo_frac)
                noise.mul_(sigma)
                torch.add(w, noise, out=b)

        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        # Save current gradients for next step's echo
        with torch.no_grad():
            for i, w in enumerate(self.W):
                if w.grad is not None:
                    self.prev_grads[i].copy_(w.grad)

        self.opt.step()

        self.forwards += 1
        return loss.item()
