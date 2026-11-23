"""
Generates the dual-panel Pareto frontier and silent-compute CoT compression plot
from results/varloop_35_summary.csv.
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(summary_csv: str, out_png: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    summary = pd.read_csv(summary_csv)

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), dpi=220)
    fig.patch.set_facecolor("#ffffff")

    # Panel 1: Pareto Frontier (Latency vs Pass@1 Accuracy)
    ax1.plot(summary["avg_latency"], summary["pass_at_1"], color="#2563eb", linewidth=2.6, zorder=2)
    ax1.scatter(
        summary["avg_latency"],
        summary["pass_at_1"],
        s=130,
        c=summary["loops"],
        cmap="viridis",
        edgecolors="#0f172a",
        linewidth=1.5,
        zorder=3,
    )

    offsets = {1: (0, 14), 2: (0, -26), 3: (0, 14), 4: (-14, 14), 5: (10, -26)}
    for _, r in summary.iterrows():
        loop_n = int(r["loops"])
        tag = " (Default)" if loop_n == 3 else (" (Peak)" if loop_n == 4 else "")
        ax1.annotate(
            f"{loop_n}x loop{tag}\n{r['pass_at_1']}% | {r['decode_tps']} t/s",
            (r["avg_latency"], r["pass_at_1"]),
            textcoords="offset points",
            xytext=offsets.get(loop_n, (0, 12)),
            ha="center",
            fontsize=8.8,
            fontweight="bold",
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#cbd5e1", lw=0.8),
        )

    ax1.set_title("Test-Time Loop Scaling: Latency vs. Pass@1 (HumanEval+ Hard-35)", fontsize=11.5, fontweight="bold", pad=12)
    ax1.set_xlabel("Mean End-to-End Task Latency (seconds) — Lower is Faster", fontsize=10)
    ax1.set_ylabel("Pass@1 Accuracy (%) — Higher is Better", fontsize=10)
    ax1.margins(x=0.22, y=0.25)
    ax1.grid(True, linestyle="--", alpha=0.45)

    # Panel 2: Silent Loops vs Visible CoT Length & Decode Throughput
    x = np.arange(len(summary))
    width = 0.38

    ax2.bar(x - width / 2, summary["avg_think_chars"], width, label="Avg Visible <think> Chars", color="#0d9488")
    ax2_twin = ax2.twinx()
    ax2_twin.bar(x + width / 2, summary["decode_tps"], width, label="Decode Speed (tok/s)", color="#6366f1")

    ax2.set_title("Silent Recurrent Compute vs. Visible CoT & Speed", fontsize=11.5, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(l)}x Loop" for l in summary["loops"]], fontweight="bold")
    ax2.set_ylabel("Avg Visible <think> Characters", color="#0f766e", fontsize=10, fontweight="bold")
    ax2_twin.set_ylabel("C++ Decode Throughput (tok/s)", color="#4338ca", fontsize=10, fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, fontsize=8.5)

    fig.suptitle(
        "Thinking Machines TML-d24-varloop-research (IQ4_NL) — Variable Loop Pareto Benchmark",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print(f"Saved chart to {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/varloop_35_summary.csv")
    parser.add_argument("--out", default="assets/inkling2_varloop_pareto_35.png")
    args = parser.parse_args()
    plot_results(args.summary, args.out)
