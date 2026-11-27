"""GRADIENT LOTTERY: at each step, randomly select a sparse subset of gradient
components and amplify them. The rest are zeroed.

Like the lottery ticket hypothesis but for GRADIENT SIGNALS: at each step,
only a random sparse subset of weight updates are applied, amplified to
compensate for the sparsity. Over many steps, every weight gets updated many
times, but never in a coordinated way with all other weights.

This breaks the tight correlation between gradient components that enables
memorisation. When the model can only update a random 10% of weights per step,
it can't make the precise, coordinated changes needed to memorise individual
training examples. It CAN still learn general patterns because those show up
consistently across many steps (and thus many random subsets).

Different from dropout (which affects the forward pass, not the gradient)
and from gradient dropout (Tseng et al., which zeros entire layer gradients).
This operates at the individual weight level.

Combined with mirror neurons and the counting teacher.
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class GradientLottery(Rule):
    name = "gradient_lottery"
    autograd = "modified"
    blurb = "At each step, only a random sparse subset of gradient components are applied (amplified to compensate), breaking the coordinated updates that enable memorisation."
    paper = "Novel: lottery-ticket-inspired gradient sparsification."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0, mirror=True,
                    keep_frac=0.3,
                    teach=0.3, beta=16.0, order=9, count_steps=1500)
    tune = "keep_frac"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        swap_linears(model, "random", self.hp["scale"])
        self.twins = [m for m in model.modules() if isinstance(m, FALinear)]
        self.opt = torch.optim.AdamW(model.parameters(),
                                     lr=self.hp["lr"],
                                     weight_decay=self.hp["weight_decay"])
        self.W = [m.weight for m in self.twins]
        self.B = [m.feedback for m in self.twins]

    def step(self, x, y):
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)
        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.forwards += 1

        # Sparse gradient lottery: keep only a random fraction, amplify to compensate
        keep = self.hp["keep_frac"]
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    mask = (torch.rand_like(p.grad) < keep).float()
                    p.grad.mul_(mask / keep)
            before = [w.clone() for w in self.W]

        self.opt.step()

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
