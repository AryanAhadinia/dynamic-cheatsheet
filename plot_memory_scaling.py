"""
Aggregate memory-scaling runs and plot held-out test accuracy vs. fraction of training data seen.

Reads every results/memory_scaling/<dataset>/<run>/summary.json, writes a CSV of all checkpoints,
prints a table, and saves one small-multiples figure (one panel per dataset, one line per run).
"""

import glob
import json
import math
import os
import re

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tap import Tap

# Categorical slots in fixed order (validated for CVD separation and contrast on a light surface)
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#1f1f1e"
TEXT_SECONDARY = "#5f5e58"
GRID = "#e4e3dc"
DATASET_TITLES = {"gsm8k": "GSM8K", "math500": "MATH500", "aime": "AIME 2020-2024"}


class Arguments(Tap):
    results_directory: str = "results/memory_scaling"
    output_path: str = "results/memory_scaling/memory_scaling.png"
    exclude: list = []  # Leave runs out of the figure (not the CSV) if their name contains any of these strings


def wilson_interval(correct: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - half, center + half


def load_runs(results_directory: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_directory, "*", "*", "summary.json"))):
        dataset, run = path.split(os.sep)[-3:-1]
        for s in json.load(open(path)):
            low, high = wilson_interval(s["test_correct"], s["n_test"])
            rows.append({"dataset": dataset, "run": run, **s, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def describe_run(run: str):
    """Split a run directory name into (model label, memory label)."""
    model = re.split(r"_append-k-?\d+|_feedback", run)[0]
    match = re.search(r"_append-k(-?\d+)", run)
    memory = ("append, all items" if match.group(1) == "-1" else f"append, top-{match.group(1)}") if match else "rewrite (DC-Cumulative)"
    feedback = re.search(r"_feedback-(\w+?)_seed", run)
    if feedback and feedback.group(1) != "gold":
        memory += f", feedback {feedback.group(1)}"
    tag = run.split("_seed")[-1].split("_", 1)  # Seed number plus the optional run tag
    if len(tag) > 1:
        memory += f", {tag[1]}"
    return model, memory


def plot(df: pd.DataFrame, output_path: str) -> None:
    datasets = [d for d in DATASET_TITLES if d in set(df["dataset"])] + sorted(set(df["dataset"]) - set(DATASET_TITLES))
    runs = sorted(df["run"].unique())
    # Color follows the model and line style the memory mode, identically in every panel
    models = sorted({describe_run(run)[0] for run in runs})
    model_colors = {m: SERIES_COLORS[i % len(SERIES_COLORS)] for i, m in enumerate(models)}
    colors = {run: model_colors[describe_run(run)[0]] for run in runs}
    styles = {run: "-" if describe_run(run)[1].startswith("append") else "--" for run in runs}

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 4.0), squeeze=False, facecolor=SURFACE)
    for ax, dataset in zip(axes[0], datasets):
        ax.set_facecolor(SURFACE)
        sub = df[df["dataset"] == dataset]
        end_labels = []
        for run in runs:
            r = sub[sub["run"] == run].sort_values("train_pct")
            if r.empty:
                continue
            ax.plot(r["train_pct"], r["test_accuracy"], color=colors[run], linestyle=styles[run], linewidth=2, marker="o",
                    markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=" · ".join(describe_run(run)))
            last = r.iloc[-1]
            end_labels.append([last["train_pct"], last["test_accuracy"], f"{last['test_accuracy']:.0%}"])
        # Spread end-of-line labels at the same x so they do not overlap
        low, high = ax.get_ylim()
        min_gap = 0.045 * (high - low)
        for x in {label[0] for label in end_labels}:
            group = sorted((label for label in end_labels if label[0] == x), key=lambda label: label[1])
            positions = [group[0][1]] if group else []
            for label in group[1:]:
                positions.append(max(label[1], positions[-1] + min_gap))
            shift = (positions[-1] - group[-1][1]) / 2 if group else 0  # Re-center the stack on the data
            for label, y in zip(group, positions):
                ax.text(x + 2.5, y - shift, label[2], va="center", fontsize=8, color=TEXT_SECONDARY)
        n_test = int(sub["n_test"].max())
        n_train = int(sub["n_train_seen"].max())
        ax.set_title(f"{DATASET_TITLES.get(dataset, dataset)}  (train {n_train}, test {n_test})", fontsize=10, color=TEXT_PRIMARY, loc="left")
        ax.set_xlabel("Training data fed to memory (%)", fontsize=9, color=TEXT_SECONDARY)
        ax.set_xticks(sorted(sub["train_pct"].unique()))
        ax.set_xlim(-5, 112)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))  # Whole-percent ticks
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0][0].set_ylabel("Held-out test accuracy", fontsize=9, color=TEXT_SECONDARY)

    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 2), frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY, handlelength=4)
    fig.suptitle("Test accuracy vs. training data accumulated in memory",
                 fontsize=11, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06 + 0.04 * math.ceil(len(labels) / 2), 1, 0.95))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor=SURFACE)
    print(f"Saved figure to {output_path}")


def main(args: Arguments):
    df = load_runs(args.results_directory)
    if df.empty:
        print(f"No summary.json files found under {args.results_directory}")
        return
    csv_path = os.path.splitext(args.output_path)[0] + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved table to {csv_path}\n")

    table = df.pivot_table(index=["dataset", "run"], columns="train_pct", values="test_accuracy")
    table.columns = [f"{c}%" for c in table.columns]
    print((table * 100).round(1).to_string())
    print()
    plot(df[~df["run"].apply(lambda run: any(x in run for x in args.exclude))], args.output_path)


if __name__ == "__main__":
    main(Arguments().parse_args())
