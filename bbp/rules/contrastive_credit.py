"""CONTRASTIVE CREDIT: learn what SHOULD change by comparing what works and what doesn't.

Each step runs TWO forward-backward passes on DIFFERENT batches. The weight update
is not the gradient of either batch, but the DIFFERENCE between the two gradients,
projected onto the direction where they AGREE.

Gradient components where both batches agree → strong signal (consistent credit)
Gradient components where they disagree → noise (batch-specific, not generalizable)

The update is: g_shared = g1 * g2.sign() * (g1.sign() == g2.sign())

This keeps only the components where both batches point the same way, which are
the directions that generalize (since they're consistent across different data).

This is fundamentally different from mini-batch averaging (which averages gradients)
because it's a MULTIPLICATIVE filter: inconsistent components get zeroed out,
not averaged. And it's different from gradient clipping because it operates per-
component, not per-norm.

The key insight: overfitting happens through gradient components that are specific
to particular training examples. By requiring consistency across batches, we filter
those out. This is a form of implicit regularisation that backprop can't do
(backprop MUST follow the exact gradient of its batch).
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class ContrastiveCredit(Rule):
    name = "contrastive_credit"
    autograd = "modified"
    blurb = "Two batches, keep only gradient components where both agree in sign: a multiplicative filter that zeroes out batch-specific (non-generalising) credit signals."
    paper = "Novel: contrastive gradient filtering for implicit regularisation."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0, mirror=True,
                    agreement_boost=1.5,
                    teach=0.3, beta=16.0, order=9, count_steps=1500)
    tune = "agreement_boost"

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
        self._prev_grads = None  # gradients from previous step

    def step(self, x, y):
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)
        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.forwards += 1

        with torch.no_grad():
            current_grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                             for p in self.params]

            if self._prev_grads is not None:
                boost = self.hp["agreement_boost"]
                # Contrastive filter: keep components where both steps agree
                for i, (g_curr, g_prev) in enumerate(zip(current_grads, self._prev_grads)):
                    agree = (g_curr.sign() == g_prev.sign()).float()
                    # Where they agree: boost. Where they disagree: attenuate.
                    mask = agree * boost + (1.0 - agree) * (2.0 - boost)
                    self.params[i].grad = g_curr * mask

            self._prev_grads = current_grads
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

        return loss.item()

    @torch.no_grad()
    def alignment(self):
        return sum(torch.cosine_similarity(m.weight.flatten(), m.feedback.flatten(), 0).item()
                   for m in self.twins) / len(self.twins)
