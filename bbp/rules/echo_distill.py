"""ECHO DISTILL: the model teaches its own future self through temporal echoes.

Each step, the model:
  1. Makes a prediction (normal forward pass)
  2. Stores the prediction as a "temporal echo"
  3. On the NEXT occurrence of similar context, trains not just against the
     true target but also against its own PREVIOUS prediction for that context

The self-distillation effect: the model learns to be CONSISTENT with itself
across time. This acts as a powerful regulariser because:
  - Random fluctuations in predictions get penalised (they differ from the echo)
  - Stable, robust predictions get reinforced
  - The echo is always slightly stale, creating a temporal ensemble effect

This is fundamentally different from standard self-distillation (which uses a
copy of the model): here the "teacher" is the model's OWN PAST, with no copy.

The echo table is keyed by the input context (last few characters), similar
to the counting teacher, but stores the MODEL's predictions, not corpus statistics.

Combined with mirror neurons for the backward pass.
"""
import torch
import torch.nn.functional as F

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class EchoDistill(Rule):
    name = "echo_distill"
    autograd = "modified"
    blurb = "Mirror neurons plus self-distillation from temporal echoes: the model's past predictions on the same context regularise its current ones."
    paper = "Novel: temporal self-distillation without a model copy."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0, mirror=True,
                    echo_alpha=0.2, echo_ctx=4, echo_ema=0.9,
                    teach=0.0, beta=16.0, order=9, count_steps=1500)
    tune = "echo_alpha"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        swap_linears(model, "random", self.hp["scale"])
        self.twins = [m for m in model.modules() if isinstance(m, FALinear)]
        self.opt = torch.optim.AdamW(model.parameters(),
                                     lr=self.hp["lr"],
                                     weight_decay=self.hp["weight_decay"])
        self.W = [m.weight for m in self.twins]
        self.B = [m.feedback for m in self.twins]
        self.V = model.cfg.vocab_size
        # Echo table: context_key -> EMA of model's softmax prediction
        self.echo_table = {}  # dict of int -> tensor(V)
        self._step_count = 0

    def _context_keys(self, x):
        """Hash the last `echo_ctx` characters at each position into an int key."""
        B, T = x.shape
        ctx = int(self.hp["echo_ctx"])
        keys = torch.zeros(B, T, dtype=torch.long, device=x.device)
        for i in range(min(ctx, T)):
            keys = keys * 67 + x[:, torch.arange(T, device=x.device) - i].clamp_min(0)
        return keys

    def step(self, x, y):
        self._step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)

        # Echo distillation: match past predictions on the same contexts
        echo_alpha = self.hp["echo_alpha"]
        if echo_alpha > 0 and self._step_count > 50:
            with torch.no_grad():
                probs = logits[:x0.shape[0]].softmax(-1)
                keys = self._context_keys(x0)
            # Build echo targets from table
            B, T = x0.shape
            echo_target = torch.zeros(B, T, self.V, device=x0.device)
            has_echo = torch.zeros(B, T, device=x0.device, dtype=torch.bool)
            for b in range(B):
                for t in range(T):
                    k = keys[b, t].item()
                    if k in self.echo_table:
                        echo_target[b, t] = self.echo_table[k]
                        has_echo[b, t] = True
            if has_echo.any():
                # KL divergence against echo targets (only where echoes exist)
                log_p = logits[:B].log_softmax(-1)
                echo_loss = -(echo_target * log_p).sum(-1)  # cross-entropy against echo
                echo_loss = (echo_loss * has_echo.float()).sum() / has_echo.float().sum().clamp_min(1)
                loss = (1.0 - echo_alpha) * loss + echo_alpha * echo_loss

            # Update echo table (EMA of current predictions)
            ema = self.hp["echo_ema"]
            with torch.no_grad():
                for b in range(min(B, 4)):  # only a few batch elements to keep it fast
                    for t in range(0, T, 4):  # subsample positions
                        k = keys[b, t].item()
                        p = probs[b, t].cpu()
                        if k in self.echo_table:
                            self.echo_table[k] = ema * self.echo_table[k] + (1 - ema) * p
                        else:
                            self.echo_table[k] = p
                # Cap table size
                if len(self.echo_table) > 100000:
                    # Remove oldest half (approximate by keeping random half)
                    keys_list = list(self.echo_table.keys())
                    for k in keys_list[::2]:
                        del self.echo_table[k]

        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            before = [w.clone() for w in self.W]

        self.opt.step()

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
