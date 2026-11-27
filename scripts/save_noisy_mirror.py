"""Save official noisy_mirror result for leaderboard (seed 0, save=True)."""
import sys
sys.path.insert(0, '.')
from bbp.arena import run, pick_device
from bbp.model import GPTConfig

D = pick_device("auto")

# Use teach=0.4 (best config from paired comparison)
r = run("noisy_mirror", 5.0, D, dict(teach=0.4, sigma0=0.5), GPTConfig(),
        seed=0, save=True, quiet=False)
print(f"\nnoisy_mirror test={r['test_loss']:.4f} val={r['best_val']:.4f} steps={r['steps']}")
