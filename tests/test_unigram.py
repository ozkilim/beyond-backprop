import json
import math

import pytest
import torch

from bbp import arena, data as data_module, leaderboard


def test_unigram_scores_heldout_targets_with_training_frequencies(tmp_path, monkeypatch):
    # Train: p(A)=3/4, p(B)=1/4. Val is all A; test is all B.
    (tmp_path / "tinyshakespeare.txt").write_text("AAAB" * 4 + "A" * 8 + "B" * 8)
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    data = data_module.CharData(2, "cpu", val_frac=0.25, test_frac=0.25)

    losses = arena.unigram_losses(data, batch_size=2)

    assert losses["unigram_val_loss"] == pytest.approx(0.2876820725)
    assert losses["unigram_test_loss"] == pytest.approx(1.3862943611)


def test_unigram_keeps_unseen_token_ids_and_zero_probabilities():
    baseline = data_module.UnigramBaseline(torch.tensor([0, 0, 2, 2]), vocab_size=3)
    x = torch.tensor([[0]])

    assert math.isinf(baseline(x, torch.tensor([[1]]))[1].item())
    assert baseline(x, torch.tensor([[2]]))[1].item() == pytest.approx(0.6931471806)


@pytest.fixture
def result_row():
    row = json.loads((arena.RESULTS / "dreaming.json").read_text())
    row.update(unigram_loss=3.309, unigram_val_loss=2.345, unigram_test_loss=3.456)
    return row


def test_leaderboard_reports_test_unigram(result_row):
    assert "*unigram (letter frequencies only)* | | 3.456 |" in leaderboard.table([result_row])


def test_curve_reports_validation_unigram(result_row, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(leaderboard, "RESULTS", tmp_path)
    try:
        leaderboard.plot([result_row])
        baseline = plt.gcf().axes[0].lines[-1]
        assert baseline.get_ydata() == pytest.approx([2.345, 2.345])
    finally:
        plt.close("all")


def test_historical_update_preserves_model_results(result_row, tmp_path, monkeypatch):
    from scripts.update_unigram_baselines import update_results

    # The default 90/5/5 split gives 120 training chars, then 7 val and 7 test.
    (tmp_path / "tinyshakespeare.txt").write_text("AAAB" * 30 + "A" * 7 + "B" * 7)
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    result_row["model"]["block_size"] = 2
    result_row["batch_size"] = 2
    results = tmp_path / "results"
    results.mkdir()
    path = results / "dreaming.json"
    path.write_text(json.dumps(result_row, indent=1))

    update_results(results)

    updated = json.loads(path.read_text())
    assert "unigram_loss" not in updated
    assert updated["unigram_val_loss"] == pytest.approx(0.2876820725)
    assert updated["unigram_test_loss"] == pytest.approx(1.3862943611)
    baseline_fields = {"unigram_loss", "unigram_val_loss", "unigram_test_loss"}
    assert {k: v for k, v in updated.items() if k not in baseline_fields} == {
        k: v for k, v in result_row.items() if k not in baseline_fields
    }
    first_update = path.read_bytes()
    update_results(results)
    assert path.read_bytes() == first_update


def test_migration_keeps_existing_split_specific_scores(result_row, tmp_path):
    from scripts.update_unigram_baselines import update_results

    result_row.pop("unigram_loss")
    path = tmp_path / "custom-split.json"
    original = json.dumps(result_row).encode()
    path.write_bytes(original)
    update_results(tmp_path)
    assert path.read_bytes() == original


@pytest.fixture
def legacy_row(result_row):
    result_row.pop("unigram_val_loss")
    result_row.pop("unigram_test_loss")
    return result_row


def test_table_keeps_legacy_model_score_without_unigram(legacy_row):
    table = leaderboard.table([legacy_row])
    assert "`dreaming`" in table
    assert f"| {legacy_row['test_loss']:.3f} |" in table
    assert "*unigram (letter frequencies only)*" not in table


def test_plot_keeps_legacy_model_curve_without_unigram(legacy_row, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(leaderboard, "RESULTS", tmp_path)
    try:
        leaderboard.plot([legacy_row])
        curve, = plt.gcf().axes[0].lines
        assert curve.get_label() == legacy_row["rule"]
        assert curve.get_ydata() == pytest.approx([point["val"] for point in legacy_row["curve"]])
    finally:
        plt.close("all")
