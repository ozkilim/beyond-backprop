"""Recompute held-out unigram baselines in existing results without retraining.

    python -m scripts.update_unigram_baselines
    python -m bbp.leaderboard
"""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from bbp.arena import RESULTS, pick_device, unigram_losses
from bbp.data import CharData


def update_results(results_dir=RESULTS, device="cpu"):
    baselines = {}
    updated = 0
    for path in sorted(results_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if "unigram_loss" not in result and {"unigram_val_loss", "unigram_test_loss"} <= result.keys():
            continue
        key = (result["model"]["block_size"], result["batch_size"], result["test_loss"] is not None)
        if key not in baselines:
            data = CharData(key[0], pick_device(device))
            baselines[key] = unigram_losses(data, key[1], evaluate_test=key[2])
        result.pop("unigram_loss", None)
        result.update(baselines[key])
        tmp = NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False)
        tmp_path = Path(tmp.name)
        try:
            with tmp:
                json.dump(result, tmp, indent=1)
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)
        updated += 1
    print(f"Updated unigram baselines in {updated} result files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Device used to score the held-out batches")
    update_results(device=parser.parse_args().device)
