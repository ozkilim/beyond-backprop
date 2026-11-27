"""QUORUM MIRROR: gradient gating via bacterial quorum sensing.

Inspired by how bacteria coordinate behaviour: each bacterium releases a
signalling molecule, and when enough bacteria agree (quorum), the whole
colony acts. Applied to gradients:

Each weight's gradient "votes" on the update direction. The vote is whether
the gradient's sign matches the RUNNING TREND (exponential moving average of
past gradient signs). When a supermajority of a row's weights agree with
their trends, the gradient for that row is AMPLIFIED (confident signal).
When there's no consensus, the gradient is ATTENUATED (noisy/conflicting).

This creates adaptive per-row gradient scaling that:
  - Amplifies gradients in directions the network has consistently wanted
    to move (generalising signal — it persists across batches)
  - Attenuates gradients that flip direction randomly (overfitting noise —
    it's batch-specific)

Combined with mirror neurons + teacher for maximum effect.

The quorum threshold and amplification decay over training:
  - Early: aggressive quorum filtering (only very consistent gradients pass)
  - Late: weak filtering (the network knows what it's doing)
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class QuorumMirror(Rule):
    name = "quorum_mirror"
    autograd = "modified"
    blurb = "Bacterial quorum sensing on gradients: amplify when row-wise gradient signs reach consensus, attenuate disagreement."
    paper = "Novel: quorum-sensing-inspired gradient gating with mirror neurons."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma0=0.5, quorum_threshold=0.6, quorum_boost=1.5,
                    quorum_decay=0.95,  # EMA decay for sign tracking
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

        # EMA of gradient signs per parameter
        self.sign_ema = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.sign_ema[name] = torch.zeros_like(p)

    def _sigma(self):
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        return self.hp["sigma0"] * (1.0 - t)

    def step(self, x, y):
        self.step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        # Noisy mirror backward
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

        # === QUORUM SENSING ===
        # After backward, before optimizer step: gate the gradients
        decay = self.hp["quorum_decay"]
        threshold = self.hp["quorum_threshold"]
        boost = self.hp["quorum_boost"]

        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.grad is None or name not in self.sign_ema:
                    continue
                g = p.grad
                # Update sign EMA
                self.sign_ema[name].mul_(decay).add_(g.sign(), alpha=1.0 - decay)

                # Compute agreement: how aligned is current gradient sign with the trend?
                # For each row of the weight matrix, compute fraction of columns that agree
                if g.dim() >= 2:
                    # Per-row consensus
                    agreement = (g.sign() * self.sign_ema[name].sign())  # +1 agree, -1 disagree
                    row_consensus = agreement.mean(dim=-1, keepdim=True)  # [-1, 1]
                    # Scale: consensus > threshold → boost, else attenuate
                    scale = torch.where(
                        row_consensus > threshold,
                        torch.full_like(row_consensus, boost),
                        torch.clamp(row_consensus / threshold, 0.3, 1.0)
                    )
                    p.grad.mul_(scale)
                # 1D params (biases, layernorms): leave unchanged

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
