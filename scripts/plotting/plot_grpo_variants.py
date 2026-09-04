"""Plot seed-42 curves for standard GRPO and its on-policy variants."""

from pathlib import Path
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


RUNS = {
    "Standard GRPO": Path("experiments/on_policy/standard/metrics_seed42.jsonl"),
    "GRPO constant": Path("experiments/on_policy/grpo_constant/metrics_seed42.jsonl"),
    "Dr. GRPO": Path("experiments/on_policy/dr_grpo/metrics_seed42.jsonl"),
    "RFT": Path("experiments/on_policy/rft/metrics_seed42.jsonl"),
    "MaxRL": Path("experiments/on_policy/maxrl/metrics_seed42.jsonl"),
}

COLORS = {
    "Standard GRPO": "#147d92",
    "GRPO constant": "#d05a45",
    "Dr. GRPO": "#6b5ca5",
    "RFT": "#c28a25",
    "MaxRL": "#39805a",
}


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def points(rows: list[dict], field: str) -> tuple[list[int], list[float]]:
    selected = [row for row in rows if field in row and row[field] is not None]
    return (
        [int(row["step"]) for row in selected],
        [float(row[field]) for row in selected],
    )


def plot_series(ax, rows: list[dict], field: str, color: str, validation: bool = False):
    steps, values = points(rows, field)
    if not steps:
        return
    ax.plot(
        steps,
        values,
        color=color,
        linestyle="--" if validation else "-",
        linewidth=1.7,
        marker="o" if validation else None,
        markersize=3.2 if validation else None,
        markevery=1 if validation else None,
        alpha=0.95,
    )


def main() -> None:
    output_path = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "experiments/figures/on_policy/variants_seed42.png"
    )
    data = {name: load_rows(path) for name, path in RUNS.items()}

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dfe7ea",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), constrained_layout=False)
    panels = [
        ("Rewards", "mean_reward", "val_reward", "reward"),
        ("Format reward", "mean_format_reward", "val_format_reward", "reward"),
        ("Policy-gradient loss", "loss", None, "loss"),
        ("Gradient norm", "grad_norm", None, "norm"),
        ("Token entropy", "token_entropy", None, "entropy"),
        ("Validation response length", None, "val_avg_response_length", "tokens"),
    ]

    for ax, (title, train_field, validation_field, ylabel) in zip(axes.flat, panels):
        for name, rows in data.items():
            color = COLORS[name]
            if train_field is not None:
                plot_series(ax, rows, train_field, color, validation=False)
            if validation_field is not None:
                plot_series(ax, rows, validation_field, color, validation=True)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("rollout step")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, 200)
        if title in {"Rewards", "Format reward"}:
            ax.set_ylim(0, 1)

    method_handles = [
        Line2D([0], [0], color=COLORS[name], linewidth=2, label=name)
        for name in RUNS
    ]
    style_handles = [
        Line2D([0], [0], color="#344149", linewidth=1.8, label="train", linestyle="-"),
        Line2D([0], [0], color="#344149", linewidth=1.8, label="validation", linestyle="--", marker="o", markersize=3),
    ]
    fig.legend(
        handles=method_handles + style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "On-policy GRPO variants on GSM8K (seed 42)",
        fontsize=15,
        fontweight="bold",
        y=1.035,
    )
    fig.text(
        0.5,
        0.995,
        "200 rollout steps; validation every 10 steps; constant-normalized runs use Z = 131072",
        ha="center",
        va="top",
        fontsize=9,
        color="#66737d",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(output_path)


if __name__ == "__main__":
    main()
