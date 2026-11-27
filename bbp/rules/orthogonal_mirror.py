"""ORTHOGONAL MIRROR: backward noise orthogonal to forward weights.

Standard noisy_mirror adds RANDOM Gaussian noise to B. But random noise
partially overlaps with the forward weights W — that overlap is redundant
(we already have W in the mix). Pure random noise wastes some of its
perturbation on the direction we already know.

Orthogonal Mirror adds noise that is ORTHOGONAL to each row of W.
For each row w_i, the noise n_i is projected to be perpendicular to w_i:

    n_i = randn() - (randn() · w_i / ||w_i||²) * w_i

This means ALL of the perturbation goes into exploring credit assignment
directions that are maximally different from the current weights.

Information-theoretic argument: orthogonal noise has maximum entropy
given the constraint that it's uncorrelated with W. Each noisy backward
pass explores a maximally different credit hypothesis.

The overhead is minimal: just an extra dot product and subtraction per row,
same O(n) as the original noise generation.
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class OrthogonalMirror(Rule):
    name = "orthogonal_mirror"
    autograd = "modified"
    blurb = "Mirror neurons with noise orthogonal to forward weights: maximally informative credit exploration."
    paper = "Novel: orthogonal backward perturbations for maximum exploration efficiency."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma0=0.5,
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

    @staticmethod
    @torch.no_grad()
    def _orthogonal_noise(w, sigma):
        """Generate noise orthogonal to each row of w, scaled by sigma * row_std."""
        noise = torch.randn_like(w)
        # Project out component along each row of w
        # For each row: n = n - (n·w / ||w||²) * w
        w_norm_sq = (w * w).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        proj = (noise * w).sum(dim=-1, keepdim=True) / w_norm_sq
        noise = noise - proj * w
        # Scale by row std of w (like original noisy_mirror)
        noise = noise * w.std() * sigma
        return noise

    def step(self, x, y):
        self.step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        sigma = self._sigma()
        saved_B = []
        with torch.no_grad():
            for w, b in zip(self.W, self.B):
                saved_B.append(b.clone())
                noise = self._orthogonal_noise(w, sigma)
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

        # Mirror update
        with torch.no_grad():
            lr_wd = self.opt.param_groups[0]["lr"] * self.hp["weight_decay"]
            for w, b, bef in zip(self.W, self.B, before):
                gap = b - bef
                b.add_(w - bef)
                b.add_(gap, alpha=-lr_wd)

        self.forwards += 1
        return loss.item()
