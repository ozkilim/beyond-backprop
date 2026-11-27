"""Paired comparison: run two rules with the same seed back-to-back,
then alternate seeds. This keeps MPS state comparable between runs."""
import sys
sys.path.insert(0, '.')
from bbp.arena import run, pick_device
from bbp.model import GPTConfig

DEVICE = pick_device("auto")

import sys
SEEDS = [0, 1, 2, 3, 4]
RULES = [
    ("backprop",         dict(teach=0.3, lr=0.000667)),
    ("noisy_mirror",     dict(teach=0.4, sigma0=0.5)),
    ("phase_transition", dict(teach=0.4, warmup_frac=0.05)),
]

results = {r: [] for r, _ in RULES}

for seed in SEEDS:
    for rule_name, hp in RULES:
        r = run(rule_name, 5.0, DEVICE, hp, GPTConfig(), seed=seed, save=False, quiet=True)
        test = r["test_loss"]
        results[rule_name].append(test)
        msg = f"seed={seed} {rule_name:25s} test={test:.4f}"
        print(msg, flush=True)
        sys.stdout.flush()

print("\n=== SUMMARY ===")
for rule_name, _ in RULES:
    vals = results[rule_name]
    mean = sum(vals) / len(vals)
    print(f"{rule_name:25s} mean={mean:.4f}  seeds={[f'{v:.4f}' for v in vals]}")

# Paired differences
print("\n=== PAIRED vs backprop ===")
bp = results["backprop"]
for rule_name, _ in RULES:
    if rule_name == "backprop":
        continue
    vals = results[rule_name]
    diffs = [v - b for v, b in zip(vals, bp)]
    mean_diff = sum(diffs) / len(diffs)
    print(f"{rule_name:25s} mean_diff={mean_diff:+.4f}  diffs={[f'{d:+.4f}' for d in diffs]}")
    wins = sum(1 for d in diffs if d < 0)
    print(f"  {'WINS' if wins > len(diffs)/2 else 'LOSES'} {wins}/{len(diffs)} seeds")
