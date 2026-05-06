"""
Final report figures only (compact, categorized):
  1) All oracle scalar runs together (episode score used for logging + health)
  2) Outcomes by experiment family: unp-native Crafter reward sum + health (2 x 4 grid)
  3) Proxy-true gap by family: smoothed Δ = logged episode/score minus env reward sum

Run from repo root:  python plot_final_figures.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTDIR = ROOT
W_EP = 220  # episode / sparse metric smoothing
W_GAP = 320  # gap smoothing slightly wider (noisier)

# --- Data definitions ---------------------------------------------------------

EXTRA_SCORE_FILES: dict[str, list[str]] = {
    # Oracle epsilon=5 scores extended from Colab stdout tail merges
    "Oracle α=0.3, ε=5": ["scores_oracle0.3_epsilon5_tail_from_logs.jsonl"],
}


def _read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_trace(label: str, score_files: list[str], metric_file: str) -> pd.DataFrame:
    score_rows = []
    for rel in score_files:
        score_rows.extend(_read_jsonl(RESULTS / rel.strip().replace("results/", "")))

    met_rows = []
    for row in _read_jsonl(RESULTS / metric_file.strip().replace("results/", "")):
        out = {}
        if row.get("step") is None:
            continue
        out["step"] = row["step"]
        if "epstats/log/reward/sum" in row:
            out["epstats/log/reward/sum"] = row["epstats/log/reward/sum"]
        if "epstats/log/health_level/avg" in row:
            out["epstats/log/health_level/avg"] = row["epstats/log/health_level/avg"]
        if len(out) > 1:
            met_rows.append(out)

    dfp = pd.DataFrame(score_rows)
    dfm = pd.DataFrame(met_rows)
    if dfp.empty and dfm.empty:
        return pd.DataFrame()

    if not dfp.empty:
        dfp = dfp.dropna(subset=["step"])
        cols = ["step"] + [c for c in ("episode/score", "episode/length") if c in dfp.columns]
        dfp = dfp[cols].groupby("step", as_index=False).mean()

    if not dfm.empty:
        dfm = dfm.dropna(subset=["step"]).groupby("step", as_index=False).mean()

    if dfp.empty:
        merged = dfm.sort_values("step")
    elif dfm.empty:
        merged = dfp.sort_values("step")
    else:
        merged = pd.merge(dfp, dfm, on="step", how="outer").sort_values("step")
        merged = merged.ffill()

    merged["curve"] = label
    merged["gap_proxy_minus_true"] = np.nan
    if "episode/score" in merged.columns and "epstats/log/reward/sum" in merged.columns:
        merged["gap_proxy_minus_true"] = merged["episode/score"] - merged["epstats/log/reward/sum"]
    return merged.sort_values("step")


def trace_for_key(key: str) -> tuple[str, list[str], str]:
    """Return label, score file list relative to results/, metric file."""
    tails = EXTRA_SCORE_FILES.get(key, [])
    # core mapping
    CORE = {
        "P1 ε=0": (["scores_baseline.jsonl"], "metrics_baseline.jsonl"),
        "P1 ε=1": (["scores_epsilon1.jsonl"], "metrics_epsilon1.jsonl"),
        "P1 ε=3": (["scores_epsilon3.jsonl"], "metrics_epsilon3.jsonl"),
        "P1 ε=5": (["scores_epsilon5.jsonl"], "metrics_epsilon5.jsonl"),
        "Static ε=0": (["scores_phase3_baseline.jsonl"], "metrics_phase3_baseline.jsonl"),
        "Static ε=1": (["scores_phase3_epsilon1.jsonl"], "metrics_phase3_epsilon1.jsonl"),
        "Static ε=3": (["scores_phase3_epsilon3.jsonl"], "metrics_phase3_epsilon3.jsonl"),
        "Static ε=5": (["scores_phase3_epsilon5.jsonl"], "metrics_phase3_epsilon5.jsonl"),
        "Dyn ε=0": (["scores_dynamic_baseline.jsonl"], "metrics_dynamic_baseline.jsonl"),
        "Dyn ε=3": (["scores_dynamic_epsilon3.jsonl"], "metrics_dynamic_epsilon3.jsonl"),
        "Dyn ε=5": (["scores_dynamic_epsilon5.jsonl"], "metrics_dynamic_epsilon5.jsonl"),
        "Oracle α=0.3, ε=1": (["scores_oracle0.3_epsilon1.jsonl"], "metrics_oracle0.3_epsilon1.jsonl"),
        "Oracle α=0.3, ε=5": (["scores_oracle0.3_epsilon5.jsonl"], "metrics_oracle0.3_epsilon5.jsonl"),
    }
    score_list, mf = CORE[key]
    score_paths = score_list + tails
    return key, score_paths, mf


def smooth_series(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=min(25, window // 4)).mean()


CATEGORIES: list[tuple[str, list[str]]] = [
    ("Phase 1 — scalar DreamerV3", ["P1 ε=0", "P1 ε=1", "P1 ε=3", "P1 ε=5"]),
    ("Phase 3A — static MO", ["Static ε=0", "Static ε=1", "Static ε=3", "Static ε=5"]),
    ("Phase 3B — dynamic MO", ["Dyn ε=0", "Dyn ε=3", "Dyn ε=5"]),
    ("Phase 3C — oracle scalar (α = 0.3)", ["Oracle α=0.3, ε=1", "Oracle α=0.3, ε=5"]),
]

COLORS_EPS = {
    "P1 ε=0": "#444444",
    "P1 ε=1": "#4DA6FF",
    "P1 ε=3": "#FF8C42",
    "P1 ε=5": "#D62839",
    "Static ε=0": "#444444",
    "Static ε=1": "#4DA6FF",
    "Static ε=3": "#FF8C42",
    "Static ε=5": "#D62839",
    "Dyn ε=0": "#444444",
    "Dyn ε=3": "#3A86FF",
    "Dyn ε=5": "#D62839",
    "Oracle α=0.3, ε=1": "#6A994E",
    "Oracle α=0.3, ε=5": "#8338EC",
}


def plot_oracles_combined(outname: str = "fig_oracles_combined.png"):
    keys = ["Oracle α=0.3, ε=1", "Oracle α=0.3, ε=5"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for k in keys:
        label, sfiles, mf = trace_for_key(k)
        df = load_trace(label, sfiles, mf)
        if df.empty:
            print("  missing oracle:", k)
            continue
        c = COLORS_EPS[k]
        if "episode/score" in df.columns:
            s = df.set_index("step")["episode/score"].astype(float).sort_index()
            axes[0].plot(smooth_series(s, W_EP).index.to_numpy(), smooth_series(s, W_EP).to_numpy(),
                        color=c, label=k, lw=2.0)
        if "epstats/log/health_level/avg" in df.columns:
            h = df.set_index("step")["epstats/log/health_level/avg"].astype(float).sort_index()
            axes[1].plot(smooth_series(h, W_EP).index.to_numpy(), smooth_series(h, W_EP).to_numpy(),
                        color=c, label=k, lw=2.0)
    axes[0].set_title("Oracle runs — logged episode score")
    axes[0].set_xlabel("environment steps")
    axes[0].set_ylabel("episode / score")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Oracle runs — avg health")
    axes[1].set_xlabel("environment steps")
    axes[1].set_ylabel("avg health")
    axes[1].set_ylim(6.4, 9.45)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.savefig(OUTDIR / outname, dpi=200)
    plt.close(fig)
    print("Saved:", outname)


def plot_outcomes_by_category(outname: str = "fig_outcomes_by_category.png"):
    n_rows = len(CATEGORIES)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4.8 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = np.array([axes])
    for r, (title, keys) in enumerate(CATEGORIES):
        axes[r, 0].set_title(f"{title} — Crafter Σ native reward")
        axes[r, 1].set_title(f"{title} — avg health")
        for k in keys:
            label, sfiles, mf = trace_for_key(k)
            df = load_trace(label, sfiles, mf)
            if df.empty:
                print("  missing:", k)
                continue
            c = COLORS_EPS[k]
            st = df.set_index("step").sort_index()
            if "epstats/log/reward/sum" in st.columns:
                x = smooth_series(st["epstats/log/reward/sum"].astype(float), W_EP)
                axes[r, 0].plot(x.index.to_numpy(), x.to_numpy(), color=c, lw=1.85, label=k)
            if "epstats/log/health_level/avg" in st.columns:
                h = smooth_series(st["epstats/log/health_level/avg"].astype(float), W_EP)
                axes[r, 1].plot(h.index.to_numpy(), h.to_numpy(), color=c, lw=1.85, label=k)
        axes[r, 0].legend(fontsize=7.8, ncol=2, frameon=False, loc="upper left")
        axes[r, 0].grid(True, alpha=0.3)
        axes[r, 1].legend(fontsize=7.8, ncol=2, frameon=False, loc="upper right")
        axes[r, 1].grid(True, alpha=0.3)
        axes[r, 1].set_ylim(6.4, 9.45)
    for r in range(n_rows):
        axes[r, 0].set_xlabel("environment steps")
        axes[r, 1].set_xlabel("environment steps")
    fig.savefig(OUTDIR / outname, dpi=200)
    plt.close(fig)
    print("Saved:", outname)


def plot_gap_by_category(outname: str = "fig_gap_by_category.png"):
    n_rows = len(CATEGORIES)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.35 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = [axes]
    for r, (title, keys) in enumerate(CATEGORIES):
        ax = axes[r]
        ax.axhline(0.0, color="#999999", ls=":", lw=1.1)
        for k in keys:
            label, sfiles, mf = trace_for_key(k)
            df = load_trace(label, sfiles, mf)
            if df.empty:
                continue
            if df["gap_proxy_minus_true"].isna().all():
                continue
            st = df.set_index("step").sort_index()
            d = smooth_series(st["gap_proxy_minus_true"].astype(float), W_GAP)
            ax.plot(d.index.to_numpy(), d.to_numpy(), color=COLORS_EPS[k], lw=1.9, label=k)
        ax.set_title(
            title + " — Delta = episode/score minus native Crafter reward sum (logged)"
        )
        ax.legend(fontsize=7.8, ncol=4, frameon=False, loc="upper left")
        ax.set_xlabel("environment steps")
        ax.set_ylabel("Δ (smoothed)")
        ax.grid(True, alpha=0.35)
    fig.savefig(OUTDIR / outname, dpi=200)
    plt.close(fig)
    print("Saved:", outname)


def write_gap_tail_tsv(
    path: str = "results/gap_tail200k.tsv",
    last_k: int = 200_000,
):
    """Mean / std of smoothed gap over the final last_k steps (single seed)."""
    rows_out = []
    for cat, keys in CATEGORIES:
        for k in keys:
            _, sfiles, mf = trace_for_key(k)
            df = load_trace(k, sfiles, mf)
            if df.empty:
                continue
            st = df.set_index("step").sort_index()
            if "gap_proxy_minus_true" not in st.columns or st[
                "gap_proxy_minus_true"
            ].isna().all():
                continue
            d = smooth_series(st["gap_proxy_minus_true"].astype(float), W_GAP).dropna()
            step_max = int(d.index.max())
            tail = d[d.index >= step_max - last_k]
            rows_out.append(
                (
                    cat,
                    k,
                    round(float(tail.mean()), 3),
                    round(float(tail.std()), 3),
                    step_max,
                )
            )
    out = RESULTS / path.replace("results/", "")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["category\tcurve\tmean_gap\tstd_gap\tmax_step"] + [
        f"{a}\t{b}\t{c}\t{d}\t{e}" for (a, b, c, d, e) in rows_out
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote:", out)


def main():
    os.chdir(ROOT)
    print("Writing final-report figures ->", OUTDIR)
    plot_oracles_combined()
    plot_outcomes_by_category()
    plot_gap_by_category()
    write_gap_tail_tsv()
    print("Done.")


if __name__ == "__main__":
    main()
