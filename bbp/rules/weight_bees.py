"""WEIGHT BEES: bee colony optimisation in weight space.

Inspired by the waggle dance of honeybees: a swarm of "scouts" explore
weight-space perturbations, report back how good each direction is, then
the whole colony shifts toward the best-reported direction.

Each step:
  1. Forward pass to get the base loss
  2. K "scout" perturbations: each flips the SIGN of the gradient for a
     random subset of parameters (not random noise — structured perturbations
     that negate specific gradient components)
  3. Each scout gets a loss from a forward pass
  4. The final update is a weighted average of all scouts' perturbation
     directions, weighted by how much they improved over the base loss

The key insight: this is NOT evolution strategies (which uses random noise).
The scouts perturb the GRADIENT DIRECTION, keeping its magnitude. This
efficiently searches the space of "which gradient components to trust."

A scout that IMPROVES by negating some gradient components has found
components where the gradient is misleading (pointing to a worse direction).
The colony learns which parts of the gradient to follow and which to invert.

Uses mirror neurons for the base gradient computation.
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class WeightBees(Rule):
    name = "weight_bees"
    autograd = "modified"
    blurb = "Scout perturbations negate random gradient components; the colony follows directions that actually improve loss, filtering out misleading gradient signals."
    paper = "Novel: bee-colony-inspired gradient direction selection."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0, mirror=True,
                    n_scouts=4, flip_frac=0.3,
                    teach=0.0, beta=16.0, order=9, count_steps=1500)
    tune = "flip_frac"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        swap_linears(model, "random", self.hp["scale"])
        self.twins = [m for m in model.modules() if isinstance(m, FALinear)]
        self.opt = torch.optim.AdamW(model.parameters(),
                                     lr=self.hp["lr"],
                                     weight_decay=self.hp["weight_decay"])
        self.W = [m.weight for m in self.twins]
        self.B = [m.feedback for m in self.twins]
        self.params = [p for p in model.parameters() if p.requires_grad]

    def step(self, x, y):
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        # Base forward + backward to get gradient
        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.forwards += 1

        base_loss = loss.item()
        n_scouts = int(self.hp["n_scouts"])
        flip_frac = self.hp["flip_frac"]

        # Save gradients
        grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                 for p in self.params]

        # Scout phase: perturb gradient directions and evaluate
        scout_masks = []
        scout_improvements = []

        for _ in range(n_scouts):
            # Each scout flips the sign of a random fraction of gradient components
            masks = []
            with torch.no_grad():
                for g in grads:
                    mask = torch.where(
                        torch.rand_like(g) < flip_frac,
                        torch.tensor(-1.0, device=g.device),
                        torch.tensor(1.0, device=g.device)
                    )
                    masks.append(mask)

                # Apply perturbed gradient as a trial step
                lr = self.opt.param_groups[0]["lr"]
                for p, g, m in zip(self.params, grads, masks):
                    p.sub_(g * m, alpha=lr)

                # Evaluate
                _, scout_loss = self.model(xa, ya)
                self.forwards += 1
                improvement = base_loss - scout_loss.item()
                scout_improvements.append(max(improvement, 0))
                scout_masks.append(masks)

                # Undo trial step
                for p, g, m in zip(self.params, grads, masks):
                    p.add_(g * m, alpha=lr)

        # Combine: weight scouts by their improvement
        total_imp = sum(scout_improvements) + 1e-8
        with torch.no_grad():
            for i, p in enumerate(self.params):
                combined_grad = grads[i].clone()
                if total_imp > 1e-7:
                    for s, (masks, imp) in enumerate(zip(scout_masks, scout_improvements)):
                        w = imp / total_imp
                        combined_grad.add_(grads[i] * (masks[i] - 1.0), alpha=w)
                p.grad = combined_grad

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

        return base_loss
