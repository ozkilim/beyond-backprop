"""Build the leaderboard table + loss-curve plot from results/*.json, splice them into
README.md between the LEADERBOARD markers, and write the model samples to results/SAMPLES.md.

    python -m bbp.leaderboard
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
README = ROOT / "README.md"
START, END = "<!-- LEADERBOARD:START -->", "<!-- LEADERBOARD:END -->"
BADGE = {"none": "🚫 none", "local": "🧱 local", "modified": "🪞 modified", "full": "✅ full"}


def load():
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS.glob("*.json"))]
    rows = [r for r in rows if r["test_loss"] is not None]  # validation-only runs have no test rank
    return sorted(rows, key=lambda r: r["test_loss"] if math.isfinite(r["test_loss"]) else 1e9)


def table(rows) -> str:
    bp = next((r["test_loss"] for r in rows if r["rule"] == "backprop" and not r["tag"]), None)
    res = next((r["test_loss"] for r in rows if r["rule"] == "reservoir" and not r["tag"]), None)
    out = [
        "| # | rule | autograd | test loss ↓ | vs backprop | beats random reservoir? | best ckpt at | steps | fwd passes | tuned |",
        "|---|------|----------|-----------:|------------:|:--:|------:|------:|-----------:|-------|",
    ]
    for i, r in enumerate(rows, 1):
        name = r["rule"] + (f" ({r['tag']})" if r["tag"] else "")
        v = r["test_loss"]
        d = f"{v - bp:+.3f}" if bp is not None and math.isfinite(v) else "—"
        beat = "—" if r["rule"] in ("reservoir",) or res is None else ("yes" if v < res else "no")
        sw = r.get("sweep")
        tuned = f"{sw['key']}={r['hp'][sw['key']]:.2g}" if sw else ""
        out.append(
            f"| {i} | `{name}` | {BADGE[r['autograd']]} | {v:.3f} | {d} | {beat} | {r['best_at_s']/60:.1f} min | {r['steps']:,} | {r['forwards']:,} | {tuned} |"
        )
    if rows:
        r0 = rows[0]
        if r0.get("unigram_test_loss") is not None:
            out.append(
                f"| | *unigram (letter frequencies only)* | | {r0['unigram_test_loss']:.3f} | | | | | | |"
            )
        out.append(f"| | *untrained / uniform guess* | | {r0['init_val']:.3f} | | | | | | |")
        out.append("")
        out.append(
            f"<sub>{r0['params']/1e6:.2f}M-param char GPT ({r0['model']['n_layer']} layers, d={r0['model']['n_embd']}, "
            f"ctx {r0['model']['block_size']}), batch {r0['batch_size']}, TinyShakespeare, {r0['budget_minutes']:g} min of "
            f"training time per rule on `{r0['device']}`, loss in nats/char, best checkpoint by val scored on test. "
            "Tagged rows are controls: (taught) is backprop with the counting teacher `taught_mirror` uses.</sub>"
        )
    return "\n".join(out)


def plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=130)
    colors = plt.cm.tab20(range(20))
    for i, r in enumerate(rows):
        c = r["curve"]
        ls = {"none": ":", "local": "--", "modified": "-.", "full": "-"}[r["autograd"]]
        ax.plot([p["t"] / 60 for p in c], [p["val"] for p in c], ls, lw=2, label=r["rule"], color=colors[i % 20])
    unigram = rows[0].get("unigram_val_loss")
    if unigram is not None:
        ax.axhline(unigram, color="grey", lw=1)
        ax.text(0.02, unigram + 0.03, "unigram baseline", color="grey", fontsize=8)
    ax.set_xlabel("training time (min)")
    ax.set_ylabel("val loss (nats/char)")
    ax.set_title("beyond-backprop: who can teach a tiny GPT Shakespeare?")
    ax.set_ylim(1.3, 4.3)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, title="line = autograd use:  — full  -. modified  -- local  ·· none", title_fontsize=7)
    fig.tight_layout()
    out = RESULTS / "curves.png"
    fig.savefig(out)
    return out


def samples(rows) -> str:
    parts = ["# Samples\n", "What each rule's model writes after its leaderboard run (prompt `ROMEO:`).\n"]
    for r in rows:
        name = r["rule"] + (f" ({r['tag']})" if r["tag"] else "")
        parts.append(f"**{name}** (test {r['test_loss']:.3f})\n\n```\n{r['sample'][:300]}\n```\n")
    return "\n".join(parts)


def main(argv=None):
    rows = load()
    if not rows:
        print("no results yet - run python -m bbp.tournament")
        return
    if any(rows[0].get(key) is None for key in ("unigram_val_loss", "unigram_test_loss")):
        print("Missing held-out unigram scores. For standard arena results, run python -m scripts.update_unigram_baselines.")
    md = table(rows)
    png = plot(rows)
    (RESULTS / "SAMPLES.md").write_text(samples(rows), encoding="utf-8", newline="\n")
    block = md + ("\n\n![loss curves](results/curves.png)\n" if png else "") + "\nWhat each model writes: [results/SAMPLES.md](results/SAMPLES.md)."
    print(md)
    if README.exists():
        text = README.read_text(encoding="utf-8")
        if START in text:
            text = re.sub(re.escape(START) + ".*?" + re.escape(END), lambda _: f"{START}\n{block}\n{END}", text, flags=re.S)
            README.write_text(text, encoding="utf-8", newline="\n")
            print(f"\nupdated {README}")


if __name__ == "__main__":
    main(sys.argv[1:])
