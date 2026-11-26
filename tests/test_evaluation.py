import json
import math

import pytest

from bbp import arena, data as data_module, leaderboard
from bbp.model import GPTConfig
from bbp.tournament import tournament


def tiny_config():
    return GPTConfig(block_size=16, n_layer=1, n_embd=32, n_head=2)


def test_validation_only_run_and_backfill_never_sample_test(tmp_path, monkeypatch):
    from scripts.update_unigram_baselines import update_results

    original_batch = data_module.CharData.batch

    def forbid_test(self, split, batch_size, generator=None):
        assert split != "test", "validation-only evaluation touched test targets"
        return original_batch(self, split, batch_size, generator)

    monkeypatch.setattr(data_module.CharData, "batch", forbid_test)
    monkeypatch.setattr(arena, "RESULTS", tmp_path)
    result = arena.run("backprop", minutes=1, device="cpu", hp={}, cfg=tiny_config(),
                       batch_size=2, max_steps=1, tag="control", evaluate_test=False, quiet=True)

    assert math.isfinite(result["best_val"])
    assert math.isfinite(result["unigram_val_loss"])
    assert result["test_loss"] is None
    assert result["unigram_test_loss"] is None
    update_results(tmp_path)
    saved = json.loads((tmp_path / "backprop-control-val-only.json").read_text())
    assert saved["test_loss"] is None
    assert saved["unigram_test_loss"] is None
    monkeypatch.setattr(leaderboard, "RESULTS", tmp_path)
    assert leaderboard.load() == []


def test_tournament_scores_test_only_for_the_selected_final_run(tmp_path, monkeypatch):
    import bbp.tournament as tournament_module

    real_run = arena.run
    test_flags = []

    def tiny_run(rule, minutes, device, hp, cfg, **kwargs):
        test_flags.append(kwargs.get("evaluate_test", True))
        return real_run(rule, minutes, device, hp, tiny_config(),
                        batch_size=2, max_steps=1, **kwargs)

    monkeypatch.setattr(tournament_module, "run", tiny_run)
    monkeypatch.setattr(arena, "RESULTS", tmp_path)
    result = tournament("backprop", minutes=1, sweep_minutes=1, device="cpu")

    assert len(test_flags) >= 4
    assert test_flags[:-1] == [False] * (len(test_flags) - 1)
    assert test_flags[-1] is True
    assert math.isfinite(result["test_loss"])
    assert math.isfinite(result["unigram_test_loss"])
    winner = min(result["sweep"]["grid"], key=lambda trial: trial["val_loss"])
    assert result["hp"]["lr"] == winner["lr"]


@pytest.mark.parametrize(("tag", "name"), [("", "backprop"), ("reg", "backprop-reg")])
def test_validation_run_preserves_existing_test_result(tmp_path, monkeypatch, tag, name):
    original = json.dumps({"rule": "backprop", "tag": tag, "test_loss": 1.75}).encode()
    scored_path = tmp_path / f"{name}.json"
    scored_path.write_bytes(original)
    monkeypatch.setattr(arena, "RESULTS", tmp_path)

    arena.run("backprop", minutes=1, device="cpu", hp={}, cfg=tiny_config(),
              batch_size=2, max_steps=0, tag=tag, evaluate_test=False, quiet=True)

    assert scored_path.read_bytes() == original
    validation = json.loads((tmp_path / f"{name}-val-only.json").read_text())
    assert validation["test_loss"] is None
    assert validation["unigram_test_loss"] is None
