"""IMMUNE MIRROR: Clonal selection on weight subsets.

Inspired by the adaptive immune system: when the body encounters a pathogen,
it doesn't redesign all antibodies — it CLONES the ones that already work
and mutates the clones slightly. The best mutants survive, the rest die.

Applied to neural network training:
  1. At each step, identify the "best" rows of each weight matrix (the ones
     whose gradients are largest / most useful)
  2. "Clone" them: copy their values to the WORST rows (smallest/least useful)
  3. Add slight mutations to the clones

This creates a form of weight-space evolution WITHIN the network, running
alongside the gradient-based updates. The best neurons get duplicated and
diversified, while the worst neurons are recycled.

The cloning is RARE (every K steps) and PARTIAL (only top/bottom 25% of
rows), so it doesn't overwhelm gradient learning. It's a nudge, not a
replacement.

Combined with mirror neurons + teacher, the immune system acts as a
THIRD optimization force: gradients improve individual weights, the teacher
improves targets, and clonal selection improves the POPULATION of neurons.
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class ImmuneMirror(Rule):
    name = "immune_mirror"
    autograd = "modified"
    blurb = "Clonal selection on neurons: periodically clone the best rows and replace the worst, with mutations."
    paper = "Novel: immune-system-inspired neuron selection + mirror neurons + counting teacher."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma0=0.5,
                    clone_every=200,    # steps between cloning events
                    clone_frac=0.1,     # fraction of rows to clone/replace
                    clone_noise=0.05,   # mutation noise on clones
                    clone_warmup=2000,  # steps before cloning starts
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

        # Track gradient magnitude per row (EMA)
        self.row_importance = {id(m.weight): torch.zeros(m.weight.shape[0], device=m.weight.device)
                              for m in self.twins}

    def _sigma(self):
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        return self.hp["sigma0"] * (1.0 - t)

    @torch.no_grad()
    def _clonal_selection(self):
        """Clone the best rows and replace the worst."""
        if self.step_count < self.hp["clone_warmup"]:
            return
        if self.step_count % self.hp["clone_every"] != 0:
            return

        frac = self.hp["clone_frac"]
        noise_scale = self.hp["clone_noise"]

        for m in self.twins:
            w = m.weight
            b = m.feedback
            n_rows = w.shape[0]
            n_clone = max(1, int(n_rows * frac))

            # Score rows by their recent gradient magnitude
            importance = self.row_importance.get(id(w))
            if importance is None or importance.abs().max() < 1e-8:
                continue

            # Find best and worst rows
            _, best_idx = importance.topk(n_clone, largest=True)
            _, worst_idx = importance.topk(n_clone, largest=False)

            # Clone best rows into worst positions (with mutation)
            for src, dst in zip(best_idx, worst_idx):
                mutation = torch.randn_like(w[dst]) * w[src].std() * noise_scale
                w[dst].copy_(w[src] + mutation)
                # Also update backward weight for the cloned row
                b[dst].copy_(b[src] + mutation)

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

        # Update row importance from gradients
        with torch.no_grad():
            for m in self.twins:
                w = m.weight
                if w.grad is not None:
                    row_grad_mag = w.grad.abs().mean(dim=-1)
                    imp = self.row_importance.get(id(w))
                    if imp is not None:
                        imp.mul_(0.99).add_(row_grad_mag, alpha=0.01)

        # Restore B
        with torch.no_grad():
            for b, sb in zip(self.B, saved_B):
                b.copy_(sb)
            before = [w.clone() for w in self.W]

        self.opt.step()

        # Clonal selection (periodic)
        self._clonal_selection()

        # Mirror update
        with torch.no_grad():
            lr_wd = self.opt.param_groups[0]["lr"] * self.hp["weight_decay"]
            for w, b, bef in zip(self.W, self.B, before):
                gap = b - bef
                b.add_(w - bef)
                b.add_(gap, alpha=-lr_wd)

        self.forwards += 1
        return loss.item()
