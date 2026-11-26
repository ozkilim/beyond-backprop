"""Run the whole leaderboard: for every rule, a quick 3-point sweep of its
`tune` hyperparameter (default x1/3, x1, x3), then a full timed run with the
winner. Trials use validation only; test is scored once for the selected final run.

    python -m bbp.tournament                       # everything, 1-min sweeps, 5-min finals
    python -m bbp.tournament --rules mezo es       # just some
    python -m bbp.tournament --minutes 2 --sweep-minutes 0.5
"""
from __future__ import annotations

import argparse
import math

from .arena import pick_device, run
from .model import GPTConfig
from .rules import RULES


def tournament(rule: str, minutes: float, sweep_minutes: float, device: str, factors=(1 / 3, 1, 3)):
    cls = RULES[rule]
    key, base = cls.tune, cls.defaults[cls.tune]
    sweep = []

    def trial(f):
        val = base * f
        r = run(rule, sweep_minutes, device, {key: val}, GPTConfig(), save=False, quiet=True,
                evaluate_test=False)
        v = r["best_val"] if math.isfinite(r["best_val"]) else float("inf")
        sweep.append({key: val, "factor": f, "val_loss": v})
        print(f"  sweep {rule:20s} {key}={val:.3g}: val {v:.3f}", flush=True)

    for f in factors:
        trial(f)
    for _ in range(2):  # winner on the edge of the grid? keep walking that way
        fs = sorted(s["factor"] for s in sweep)
        best_f = min(sweep, key=lambda s: s["val_loss"])["factor"]
        if best_f == fs[-1]:
            trial(best_f * 3)
        elif best_f == fs[0]:
            trial(best_f / 3)
        else:
            break
    best = min(sweep, key=lambda s: s["val_loss"])
    print(f"  -> {rule}: {key}={best[key]:.3g}, final {minutes}-min run")
    res = run(rule, minutes, device, {key: best[key]}, GPTConfig(), save=False, evaluate_test=True)
    res["sweep"] = dict(key=key, minutes=sweep_minutes, grid=sweep)
    from .arena import RESULTS
    import json
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{rule}.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", nargs="*", default=sorted(RULES))
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--sweep-minutes", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = pick_device(a.device)
    for r in a.rules:
        print(f"=== {r} ===", flush=True)
        tournament(r, a.minutes, a.sweep_minutes, dev)
    from .leaderboard import main as lb
    lb([])


if __name__ == "__main__":
    main()
