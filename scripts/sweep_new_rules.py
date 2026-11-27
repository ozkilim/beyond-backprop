"""Run the most promising new rules for 5 minutes each, with clean GPU access.
Run this AFTER the paired comparison finishes."""
import sys
sys.path.insert(0, '.')
from bbp.arena import run, pick_device
from bbp.model import GPTConfig

D = pick_device("auto")

configs = [
    # Annealed mirror: scheduled teach decay
    ("annealed_mirror", dict(teach_max=0.5, teach_min=0.15, sigma0=0.5)),
    ("annealed_mirror", dict(teach_max=0.5, teach_min=0.1, sigma0=0.5)),
    ("annealed_mirror", dict(teach_max=0.6, teach_min=0.2, sigma0=0.5)),

    # Backward dropout
    ("backward_dropout", dict(teach=0.3, drop_max=0.3)),
    ("backward_dropout", dict(teach=0.4, drop_max=0.2)),

    # Cosine LR (noisy_mirror_plus with gc=False, clip=0, warmup=0)
    ("noisy_mirror_plus", dict(teach=0.3, sigma0=0.5)),
    ("noisy_mirror_plus", dict(teach=0.4, sigma0=0.5)),

    # Teacher HP variations on base noisy_mirror
    ("noisy_mirror", dict(teach=0.4, sigma0=0.5, beta=8.0)),
    ("noisy_mirror", dict(teach=0.4, sigma0=0.5, beta=24.0)),
    ("noisy_mirror", dict(teach=0.5, sigma0=0.3)),
    ("noisy_mirror", dict(teach=0.4, sigma0=0.5, smoothing='kn')),
]

for rule, hp in configs:
    r = run(rule, 5.0, D, hp, GPTConfig(), save=False, quiet=True)
    print(f"{rule:30s} {str(hp):60s} test={r['test_loss']:.4f} val={r['best_val']:.4f} steps={r['steps']}", flush=True)
