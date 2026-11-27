"""PREDATOR-PREY: two competing noise processes on backward weights.

Inspired by Lotka-Volterra dynamics in ecology. We maintain TWO noise
processes on the backward weights that interact:

  1. PREDATOR noise: large, rare perturbations (exploration spikes)
  2. PREY noise: small, constant perturbations (steady exploration)

The predator is active only when the prey is "abundant" (the model is
in a flat region where small noise doesn't help). The prey recovers when
the predator is dormant (the model is making progress).

Concretely:
  - Track a smoothed loss. When loss is FLAT (not decreasing), the
    predator activates: large noise spike on B, then it goes dormant.
  - Between predator activations, steady prey noise (like standard
    noisy_mirror but smaller).

The dynamics: the model trains with steady small noise → plateaus →
predator kicks it out → new descent → prey resumes → plateaus again → ...

This creates PUNCTUATED EQUILIBRIUM in the training dynamics, like
evolution: long periods of gradual improvement punctuated by sudden
large jumps.

Implementation: the predator is a flag that triggers based on loss
stagnation. When triggered, sigma spikes for one step, then returns
to base. Almost zero overhead — just a comparison per step.
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class PredatorPrey(Rule):
    name = "predator_prey"
    autograd = "modified"
    blurb = "Punctuated equilibrium: steady exploration with burst kicks when training plateaus."
    paper = "Novel: loss-triggered noise bursts on mirror backward weights (ecological dynamics)."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma_base=0.15,       # steady "prey" noise
                    sigma_spike=0.8,       # "predator" spike magnitude
                    stagnation_window=200, # steps to detect plateau
                    stagnation_tol=0.005,  # loss must improve by this much
                    cooldown=300,          # steps between spikes
                    teach=0.3, beta=16.0, order=9, count_steps=1500)
    tune = "sigma_base"

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
        self.est_total_steps = 15000

        # Predator-prey state
        self.loss_history = []
        self.last_spike = -9999  # step of last predator activation

    def _sigma(self):
        """Determine noise magnitude: steady base + occasional spikes."""
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        base = self.hp["sigma_base"] * (1.0 - t)  # decaying base noise

        # Check for stagnation (plateau detection)
        window = self.hp["stagnation_window"]
        cooldown = self.hp["cooldown"]
        if (len(self.loss_history) >= window and
                self.step_count - self.last_spike > cooldown):
            recent = self.loss_history[-window:]
            first_half = sum(recent[:window//2]) / (window//2)
            second_half = sum(recent[window//2:]) / (window//2)
            improvement = first_half - second_half
            if improvement < self.hp["stagnation_tol"]:
                # Plateau detected! Predator activates.
                self.last_spike = self.step_count
                spike = self.hp["sigma_spike"] * (1.0 - t)
                return max(spike, base)

        return base

    def step(self, x, y):
        self.step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        sigma = self._sigma()

        with torch.no_grad():
            for w, b in zip(self.W, self.B):
                noise = torch.randn_like(w)
                noise.mul_(w.std() * sigma)
                torch.add(w, noise, out=b)

        logits, loss = self.model(xa, ya)
        self.loss_history.append(loss.item())
        # Keep history bounded
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-500:]

        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        self.forwards += 1
        return loss.item()
