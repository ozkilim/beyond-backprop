"""DOUBLE NOISE: noise on BOTH forward and backward weights.

Noisy mirror adds noise to backward weights (B = W + ε_b).
What if we also add noise to forward weights during the forward pass?

  Forward: use W + ε_f (noisy forward pass — like weight noise / multiplicative noise)
  Backward: use W + ε_b (noisy backward pass — like noisy mirror)
  Update: applied to the clean W

The forward noise acts as REGULARIZATION on the representations:
  - The network must learn representations robust to weight perturbations
  - This is similar to weight dropout (Wan 2013) but continuous

The backward noise acts as EXPLORATION of the credit assignment:
  - Different gradient directions are explored
  - Combined with forward noise, the model must find solutions that are
    robust to perturbations in BOTH the computation and the credit assignment

The two noises are INDEPENDENT — they don't need to be the same vector.
Both decay on the same schedule as noisy_mirror.

The key: forward noise uses a SMALLER magnitude than backward noise,
because forward noise directly affects the loss (and can destabilize
training), while backward noise only affects the gradient direction.

Uses fast_mirror approach: no clone/restore, just set W/B fresh each step.
But we need to restore W after forward pass since optimizer updates clean W.
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class DoubleNoise(Rule):
    name = "double_noise"
    autograd = "modified"
    blurb = "Noise on both forward and backward weights: regularize representations AND credit assignment."
    paper = "Novel: dual independent noise injection on forward and backward weights."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma0=0.5,       # backward noise magnitude
                    fwd_sigma0=0.1,   # forward noise magnitude (smaller!)
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
        self.step_count = 0
        self.est_total_steps = 15000  # slightly slower due to extra noise

    def _sigma(self, base):
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        return base * (1.0 - t)

    def step(self, x, y):
        self.step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        b_sigma = self._sigma(self.hp["sigma0"])
        f_sigma = self._sigma(self.hp["fwd_sigma0"])

        # Inject noise into BOTH forward and backward weights
        saved_W = []
        with torch.no_grad():
            for w, b in zip(self.W, self.B):
                # Save clean W for restoration
                saved_W.append(w.clone())
                # B = W + backward noise
                noise_b = torch.randn_like(w)
                noise_b.mul_(w.std() * b_sigma)
                torch.add(w, noise_b, out=b)
                # W = W + forward noise (smaller)
                if f_sigma > 0:
                    noise_f = torch.randn_like(w)
                    noise_f.mul_(w.std() * f_sigma)
                    w.add_(noise_f)

        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        # Restore clean W before optimizer step
        with torch.no_grad():
            for w, sw in zip(self.W, saved_W):
                w.copy_(sw)

        self.opt.step()

        self.forwards += 1
        return loss.item()
