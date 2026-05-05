import pandas as pd
import json
import matplotlib.pyplot as plt
import os

# All keys use ASCII only (Windows cp1252 safe)
RUNS = {
    "Baseline (e=0)":         ("results/scores_baseline.jsonl",          "results/metrics_baseline.jsonl"),
    "Phase 1 - e=1":          ("results/scores_epsilon1.jsonl",           "results/metrics_epsilon1.jsonl"),
    "Phase 1 - e=3":          ("results/scores_epsilon3.jsonl",           "results/metrics_epsilon3.jsonl"),
    "Phase 1 - e=5":          ("results/scores_epsilon5.jsonl",           "results/metrics_epsilon5.jsonl"),
    "Phase 3 Static - e=0":   ("results/scores_phase3_baseline.jsonl",    "results/metrics_phase3_baseline.jsonl"),
    "Phase 3 Static - e=1":   ("results/scores_phase3_epsilon1.jsonl",    "results/metrics_phase3_epsilon1.jsonl"),
    "Phase 3 Static - e=3":   ("results/scores_phase3_epsilon3.jsonl",    "results/metrics_phase3_epsilon3.jsonl"),
    "Phase 3 Static - e=5":   ("results/scores_phase3_epsilon5.jsonl",    "results/metrics_phase3_epsilon5.jsonl"),
    "Phase 3 Dynamic - e=0":  ("results/scores_dynamic_baseline.jsonl",   "results/metrics_dynamic_baseline.jsonl"),
    "Phase 3 Dynamic - e=3":  ("results/scores_dynamic_epsilon3.jsonl",   "results/metrics_dynamic_epsilon3.jsonl"),
    "Phase 3 Dynamic - e=5":  ("results/scores_dynamic_epsilon5.jsonl",   "results/metrics_dynamic_epsilon5.jsonl"),
    "Oracle Scalar - e=5 a=0.3": ("results/scores_oracle0.3_epsilon5.jsonl", "results/metrics_oracle0.3_epsilon5.jsonl"),
}

EXTRA_SCORE_FILES = {
    "Oracle Scalar - e=5 a=0.3": [
        "results/scores_oracle0.3_epsilon5_tail_from_logs.jsonl",
    ],
}

# Load all data
raw = {k: [] for k in RUNS}
print("Loading data...")
for name, (score_path, metric_path) in RUNS.items():
    score_files = [score_path] + EXTRA_SCORE_FILES.get(name, [])
    for sf in score_files:
        if os.path.exists(sf):
            with open(sf, encoding="utf-8") as f:
                for line in f:
                    try:
                        raw[name].append(json.loads(line))
                    except:
                        pass
    if os.path.exists(metric_path):
        with open(metric_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    metric_row = {}
                    if "step" in row:
                        metric_row["step"] = row["step"]
                    if "epstats/log/health_level/avg" in row:
                        metric_row["epstats/log/health_level/avg"] = row["epstats/log/health_level/avg"]
                    if "epstats/log/reward/sum" in row:
                        metric_row["epstats/log/reward/sum"] = row["epstats/log/reward/sum"]
                    if metric_row:
                        raw[name].append(metric_row)
                except:
                    pass

dfs = {}
for name, d in raw.items():
    if d:
        df = pd.DataFrame(d).groupby("step").mean()
        dfs[name] = df
        print(f"  {name}: score={('episode/score' in df.columns)}, health={('epstats/log/health_level/avg' in df.columns)}, rows={len(df)}")

W = 150  # smoothing window
OUTDIR = r"C:\Users\Akash Pramod Yalla\UMASS\Spring 2026\690S AI Alignment\project\dreamerv3"

# Style: name -> (color, linestyle, linewidth)
PHASE1_STYLE = {
    "Baseline (e=0)": ("#333333", "--", 2.5),
    "Phase 1 - e=1":  ("#4DA6FF", "-",  1.8),
    "Phase 1 - e=3":  ("#FF8C42", "-",  1.8),
    "Phase 1 - e=5":  ("#D62839", "-",  1.8),
}
P3S_STYLE = {
    "Baseline (e=0)":        ("#333333", "--", 2.5),
    "Phase 3 Static - e=0":  ("#2EC4B6", "-",  1.8),
    "Phase 3 Static - e=1":  ("#4DA6FF", "-",  1.8),
    "Phase 3 Static - e=3":  ("#FF8C42", "-",  1.8),
    "Phase 3 Static - e=5":  ("#D62839", "-",  1.8),
}
P3D_STYLE = {
    "Baseline (e=0)":         ("#333333", "--", 2.5),
    "Phase 3 Dynamic - e=0":  ("#8338EC", "-",  1.8),
    "Phase 3 Dynamic - e=3":  ("#3A86FF", "-",  1.8),
    "Phase 3 Dynamic - e=5":  ("#D62839", "-",  1.8),
}
ORACLE_STYLE = {
    "Phase 1 - e=5":            ("#D62839", "-",  2.0),
    "Phase 3 Static - e=5":     ("#FF8C42", "-",  2.0),
    "Phase 3 Dynamic - e=5":    ("#3A86FF", "-",  2.0),
    "Oracle Scalar - e=5 a=0.3": ("#8338EC", "-", 2.2),
}


def make_figure(style_dict, title, outname):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for name, (color, ls, lw) in style_dict.items():
        if name not in dfs:
            print(f"  MISSING data for: {name}")
            continue
        df = dfs[name]

        if "episode/score" in df.columns:
            s = df["episode/score"].dropna().sort_index()
            axes[0].plot(s.rolling(W, min_periods=1).mean(), label=name,
                         color=color, linestyle=ls, linewidth=lw)

        if "epstats/log/health_level/avg" in df.columns:
            h = df["epstats/log/health_level/avg"].dropna().sort_index()
            axes[1].plot(h.rolling(W, min_periods=1).mean(), label=name,
                         color=color, linestyle=ls, linewidth=lw)

    axes[0].set_title("Episode Score (Hacked Reward)", fontsize=11)
    axes[0].set_xlabel("Environment Steps")
    axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Average Agent Health per Episode", fontsize=11)
    axes[1].set_xlabel("Environment Steps")
    axes[1].set_ylabel("Health Level (0-9)")
    axes[1].set_ylim([6.5, 9.3])
    axes[1].legend(fontsize=8.5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTDIR, outname)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", outname)


def make_true_reward_figure(style_dict, title, outname):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for name, (color, ls, lw) in style_dict.items():
        if name not in dfs:
            print(f"  MISSING data for: {name}")
            continue
        df = dfs[name]

        # Un-inflated reward directly from environment logs (not hacked score).
        if "epstats/log/reward/sum" in df.columns:
            r = df["epstats/log/reward/sum"].dropna().sort_index()
            axes[0].plot(r.rolling(W, min_periods=1).mean(), label=name,
                         color=color, linestyle=ls, linewidth=lw)

        if "epstats/log/health_level/avg" in df.columns:
            h = df["epstats/log/health_level/avg"].dropna().sort_index()
            axes[1].plot(h.rolling(W, min_periods=1).mean(), label=name,
                         color=color, linestyle=ls, linewidth=lw)

    axes[0].set_title("Episode Reward (Un-inflated Env Reward)", fontsize=11)
    axes[0].set_xlabel("Environment Steps")
    axes[0].set_ylabel("log/reward/sum")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Average Agent Health per Episode", fontsize=11)
    axes[1].set_xlabel("Environment Steps")
    axes[1].set_ylabel("Health Level (0-9)")
    axes[1].set_ylim([6.5, 9.3])
    axes[1].legend(fontsize=8.5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTDIR, outname)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", outname)


print("\nGenerating figures...")
make_figure(PHASE1_STYLE,
    "Phase 1: Scalar DreamerV3  -  Baseline vs Reward Hacking (e in {1,3,5})",
    "fig_phase1.png")

make_figure(P3S_STYLE,
    "Phase 3A: Static MO-Dreamer  -  All Epsilon vs Baseline",
    "fig_phase3_static.png")

make_figure(P3D_STYLE,
    "Phase 3B: Dynamic Homeostatic MO-Dreamer  -  vs Baseline",
    "fig_phase3_dynamic.png")

make_figure(ORACLE_STYLE,
    "Phase 3C: Oracle Scalar Baseline (epsilon=5, alpha=0.3)",
    "fig_phase3_oracle.png")

print("\nGenerating un-inflated reward figures...")
make_true_reward_figure(PHASE1_STYLE,
    "Phase 1 (Comparable): Un-inflated Env Reward vs Baseline",
    "fig_phase1_true_reward.png")

make_true_reward_figure(P3S_STYLE,
    "Phase 3A (Comparable): Un-inflated Env Reward vs Baseline",
    "fig_phase3_static_true_reward.png")

make_true_reward_figure(P3D_STYLE,
    "Phase 3B (Comparable): Un-inflated Env Reward vs Baseline",
    "fig_phase3_dynamic_true_reward.png")

make_true_reward_figure(ORACLE_STYLE,
    "Phase 3C (Comparable): Oracle vs e=5 Runs (Un-inflated Reward)",
    "fig_phase3_oracle_true_reward.png")

print("All done.")
