"""DRIFT-DIFFUSION: gradient noise scaled by backward-signal uncertainty.

Inspired by the drift-diffusion model in neuroscience (Ratcliff 1978):
each weight accumulates a deterministic "drift" (from the backward pass)
plus stochastic "diffusion" that is proportional to how UNCERTAIN the
backward signal has been recently.

Uncertainty = running variance of the gradient for each weight. Where the
gradient has been fluctuating wildly, we ADD more exploration noise. Where
it's been consistent, we trust the signal and add little noise.

This is fundamentally different from Adam (which uses variance to adjust
the LEARNING RATE). Here variance adjusts the NOISE MAGNITUDE in the
update direction. High variance → more exploration, low variance → more
exploitation, per weight.

Combined with mirror neurons as the backward mechanism.
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class DriftDiffusion(Rule):
    name = "drift_diffusion"
    autograd = "modified"
    blurb = "Mirror-neuron credit plus noise that scales with per-weight gradient uncertainty: noisy where the signal is unreliable, precise where it's clear."
    paper = "Novel: Ratcliff's drift-diffusion (1978) applied as a credit-assignment mechanism."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0, mirror=True,
                    sigma=0.3, ema_var=0.999,
                    teach=0.0, beta=16.0, order=9, count_steps=1500)
    tune = "sigma"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        swap_linears(model, "random", self.hp["scale"])
        self.twins = [m for m in model.modules() if isinstance(m, FALinear)]
        self.opt = torch.optim.AdamW(model.parameters(),
                                     lr=self.hp["lr"],
                                     weight_decay=self.hp["weight_decay"])
        self.W = [m.weight for m in self.twins]
        self.B = [m.feedback for m in self.twins]
        # Running mean and variance of gradients for each parameter
        self.grad_mean = {}
        self.grad_var = {}

    def step(self, x, y):
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)
        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        sigma = self.hp["sigma"]
        ema = self.hp["ema_var"]

        with torch.no_grad():
            before = [w.clone() for w in self.W] if self.hp["mirror"] else None

            # Add uncertainty-scaled noise to gradients before optimizer step
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                g = p.grad
                # Update running stats
                if name not in self.grad_mean:
                    self.grad_mean[name] = g.clone()
                    self.grad_var[name] = torch.zeros_like(g)
                else:
                    delta = g - self.grad_mean[name]
                    self.grad_mean[name].mul_(ema).add_(g, alpha=1.0 - ema)
                    self.grad_var[name].mul_(ema).add_(delta.pow(2), alpha=1.0 - ema)

                # Noise proportional to sqrt(variance)
                std = self.grad_var[name].sqrt().clamp_min(1e-8)
                noise = torch.randn_like(g) * std * sigma
                p.grad.add_(noise)

        self.opt.step()

        # Mirror update for backward weights
        if self.hp["mirror"]:
            with torch.no_grad():
                lr_wd = self.opt.param_groups[0]["lr"] * self.hp["weight_decay"]
                for w, b, bef in zip(self.W, self.B, before):
                    gap = b - bef
                    b.add_(w - bef)  # same step as forward
                    b.add_(gap, alpha=-lr_wd)  # own decay

        self.forwards += 1
        return loss.item()

    @torch.no_grad()
    def alignment(self):
        return sum(torch.cosine_similarity(m.weight.flatten(), m.feedback.flatten(), 0).item()
                   for m in self.twins) / len(self.twins)
