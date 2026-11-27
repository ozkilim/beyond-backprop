"""METABOLIC: backward signal attenuated by weight magnitude — biological pruning pressure.

In biology, large synapses are metabolically expensive. This rule applies that
pressure to the BACKWARD pass: each backward weight is modulated by a soft gate
that attenuates signal through large-magnitude forward weights and amplifies it
through small ones.

    effective_B[i,j] = B[i,j] * gate(|W[i,j]|)

where gate(m) = 2 * sigmoid(-steepness * (m - threshold))

This creates two pressures:
  1. Large weights get weaker credit signals → they grow more slowly
  2. Small weights get stronger credit signals → they're explored more aggressively

The result is a natural sparsification / diversification of the weight structure
through the credit-assignment mechanism, NOT through explicit regularisation.
Backprop can't do this because its backward pass must be exact.

Combined with mirror neurons for the base backward mechanism.
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class Metabolic(Rule):
    name = "metabolic"
    autograd = "modified"
    blurb = "Mirror neurons with a metabolic gate: large forward weights get attenuated backward signals, small weights get amplified, creating biological-style pruning pressure."
    paper = "Novel: metabolic cost as a backward-pass mechanism."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0, mirror=True,
                    steepness=3.0, threshold=1.0,
                    teach=0.0, beta=16.0, order=9, count_steps=1500)
    tune = "steepness"

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

        # Apply metabolic gate to backward weights before the backward pass
        steepness = self.hp["steepness"]
        threshold = self.hp["threshold"]
        saved_B = []
        with torch.no_grad():
            for w, b in zip(self.W, self.B):
                saved_B.append(b.clone())
                # Gate: attenuate where |W| is large, amplify where small
                # Normalise threshold relative to each layer's weight scale
                w_scale = w.abs().mean()
                gate = 2.0 * torch.sigmoid(-steepness * (w.abs() / w_scale.clamp_min(1e-6) - threshold))
                b.mul_(gate)

        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        # Restore backward weights before the optimizer step (gate was just for credit)
        with torch.no_grad():
            for b, sb in zip(self.B, saved_B):
                b.copy_(sb)
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

        self.forwards += 1
        return loss.item()

    @torch.no_grad()
    def alignment(self):
        return sum(torch.cosine_similarity(m.weight.flatten(), m.feedback.flatten(), 0).item()
                   for m in self.twins) / len(self.twins)
