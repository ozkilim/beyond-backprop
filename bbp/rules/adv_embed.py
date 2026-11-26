"""ADVERSARIAL EMBEDDINGS: every step, first find which way each input character's embedding
should move to hurt the prediction most (one backward pass), push it `eps` x its own size that
way, and learn from the gradient taken at those worst-case inputs (a second pass). A continuous,
worst-case cousin of replacing random input characters. The update is not the gradient at
the real inputs, so 'modified'. Miyato et al. 2017 (FGM); Zhu et al. 2020 (FreeLB)."""
import torch

from . import register
from ._regularise import regularise
from ._teacher import teach
from .backprop import Backprop


@register
class AdvEmbed(Backprop):
    name = "adv_embed"
    autograd = "modified"
    blurb = "Learns from the gradient at input embeddings pushed in their worst-case direction (one extra pass)."
    paper = "Miyato et al. 2017 (adversarial training for text); Zhu et al. 2020 (FreeLB)"
    defaults = dict(Backprop.defaults, eps=0.5, mix=1.0)

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        self.delta = None
        self._grab = False
        orig = model.embed

        def embed(idx):
            e = orig(idx)
            if self._grab:
                e.retain_grad()
                self._e = e
            if self.delta is not None:
                e = e + self.delta
            return e
        model.embed = embed

    def step(self, x, y):
        T = getattr(self, "teacher", None)
        if self.hp.get("teach") and (T is None or not T.ready):
            return super().step(x, y)
        x0, x = x, regularise(self, [self.opt], x)
        self.opt.zero_grad(set_to_none=True)
        self._grab = True
        logits, loss = self.model(x, y)
        loss = teach(self, x0, y, logits, loss)
        loss.backward()
        self._grab = False
        with torch.no_grad():
            g, e = self._e.grad, self._e.detach()
            scale = e.norm(dim=-1, keepdim=True) * self.hp["eps"]
            self.delta = g / (g.norm(dim=-1, keepdim=True) + 1e-12) * scale
        mix = self.hp["mix"]
        if mix < 1:  # keep part of the clean gradient
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.mul_(1 - mix)
        else:
            self.opt.zero_grad(set_to_none=True)
        logits, l2 = self.model(x, y)
        l2 = teach(self, x0, y, logits, l2)
        (l2 * mix).backward() if mix < 1 else l2.backward()
        self.delta = None
        self.opt.step()
        self.forwards += 2
        return loss.item()
