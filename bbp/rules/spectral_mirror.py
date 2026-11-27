"""SPECTRAL MIRROR: mirror neurons with periodic low-rank projection.

Same as mirror_neurons: separate backward weights that learn the same
Hebbian product as their forward twins. But periodically (every `project_every`
steps), each backward weight matrix is projected to its top `rank` singular
vectors via truncated SVD.

The projection forces credit assignment to flow through a low-dimensional
subspace. This:
  1. Acts as a regulariser (the backward pass can't encode noise)
  2. Removes the accumulated drift between B and W's small singular values
  3. Effectively creates a "principled feedback alignment" where the random
     backward directions are replaced by the PRINCIPAL directions of the
     true gradient flow

This is different from mirror_neurons' `frank` option (which projects W's
SVD onto B) because here we project B's OWN SVD — the backward weights'
own learned structure, not the forward weights' structure imposed externally.
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class SpectralMirror(Rule):
    name = "spectral_mirror"
    autograd = "modified"
    blurb = "Mirror neurons with periodic SVD: backward weights are projected to their own top-k singular vectors, forcing credit through a low-dimensional manifold."
    paper = "Novel: spectral regularisation of the backward pathway."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    rank=16, project_every=100,
                    teach=0.0, beta=16.0, order=9, count_steps=1500)
    tune = "rank"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        swap_linears(model, "random", self.hp["scale"])
        self.twins = [m for m in model.modules() if isinstance(m, FALinear)]
        self.opt = torch.optim.AdamW(model.parameters(),
                                     lr=self.hp["lr"],
                                     weight_decay=self.hp["weight_decay"])
        self.W = [m.weight for m in self.twins]
        self.B = [m.feedback for m in self.twins]
        self._step_count = 0

    @torch.no_grad()
    def _project(self):
        """Project each backward weight to its own top-k SVD."""
        k = int(self.hp["rank"])
        for b in self.B:
            # Use float32 for SVD stability
            U, S, Vh = torch.linalg.svd(b.float(), full_matrices=False)
            b.copy_((U[:, :k] * S[:k]) @ Vh[:k])

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

        # Mirror update: backward weights get the same step as forward
        with torch.no_grad():
            lr_wd = self.opt.param_groups[0]["lr"] * self.hp["weight_decay"]
            for w, b, bef in zip(self.W, self.B, before):
                gap = b - bef
                b.add_(w - bef)
                b.add_(gap, alpha=-lr_wd)

        # Periodic spectral projection
        if self._step_count % int(self.hp["project_every"]) == 0:
            self._project()

        self.forwards += 1
        return loss.item()

    @torch.no_grad()
    def alignment(self):
        return sum(torch.cosine_similarity(m.weight.flatten(), m.feedback.flatten(), 0).item()
                   for m in self.twins) / len(self.twins)
