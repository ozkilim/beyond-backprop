"""Sweep of all new rule variants, run AFTER paired comparison finishes.
One config per idea (best guess HP), 16 configs × 5 min ≈ 80 min.
"""
import sys
sys.path.insert(0, '.')
from bbp.arena import run, pick_device
from bbp.model import GPTConfig

D = pick_device("auto")

configs = [
    # === BASELINE ===
    ("backprop", dict(teach=0.3)),

    # === FAST MIRROR (speed-optimized noisy mirror) ===
    ("fast_mirror", dict(teach=0.4, sigma0=0.5)),

    # === WARM RESTART MIRROR (periodic noise spikes) ===
    ("warm_restart_mirror", dict(teach=0.3, sigma0=0.5, t0_frac=0.15)),

    # === GRADIENT LANGEVIN (control: is mirror structure needed?) ===
    ("gradient_langevin", dict(teach=0.3, sigma0=0.5)),

    # === DOUBLE NOISE (forward + backward) ===
    ("double_noise", dict(teach=0.3, sigma0=0.5, fwd_sigma0=0.1)),

    # === MOMENTUM NOISE (temporally correlated) ===
    ("momentum_noise", dict(teach=0.3, sigma0=0.5, noise_momentum=0.9)),

    # === ANTISYMMETRIC NOISE (alternating sign) ===
    ("antisymmetric_noise", dict(teach=0.3, sigma0=0.5)),

    # === ECHO GRADIENT (prev grad as structured noise) ===
    ("echo_gradient", dict(teach=0.3, sigma0=0.5, noise_frac=0.3)),

    # === ANNEALED MIRROR (teach schedule) ===
    ("annealed_mirror", dict(teach_max=0.5, teach_min=0.15, sigma0=0.5)),

    # === SYNTHETIC GRADIENT MIX (front-loaded teacher) ===
    ("synthetic_gradient_mix", dict(teach_max=0.6, teach_min=0.08, sigma0=0.5)),

    # === ATTENTION-AWARE (depth-scaled noise) ===
    ("attention_aware_noise", dict(teach=0.3, sigma0=0.5, depth_ratio=3.0)),

    # === PREDATOR-PREY (punctuated equilibrium) ===
    ("predator_prey", dict(teach=0.3, sigma_base=0.15, sigma_spike=0.8)),

    # === BACKWARD DROPOUT (multiplicative sparsity) ===
    ("backward_dropout", dict(teach=0.3, drop_max=0.3)),

    # === NOISY MIRROR HP VARIANTS ===
    ("noisy_mirror", dict(teach=0.4, sigma0=0.5)),   # best single-run config
    ("noisy_mirror", dict(teach=0.5, sigma0=0.3)),    # more teacher, less noise
    ("noisy_mirror", dict(teach=0.3, sigma0=0.7)),    # more noise
]

print(f"{'rule':30s} {'hp':60s} {'test':>8s} {'val':>8s} {'steps':>7s}", flush=True)
print("-" * 120, flush=True)

for rule, hp in configs:
    try:
        r = run(rule, 5.0, D, hp, GPTConfig(), save=False, quiet=True)
        print(f"{rule:30s} {str(hp):60s} test={r['test_loss']:.4f} val={r['best_val']:.4f} steps={r['steps']:>6d}", flush=True)
    except Exception as e:
        print(f"{rule:30s} {str(hp):60s} ERROR: {e}", flush=True)
