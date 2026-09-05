"""绘制 seed 42 的 learning-rate sweep 结果。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = {
    5e-6: Path("experiments/learning_rate_sweep/lr_5e-6/metrics_seed42.jsonl"),
    1e-5: Path("experiments/on_policy/standard/metrics_seed42.jsonl"),
    2e-5: Path("experiments/learning_rate_sweep/lr_2e-5/metrics_seed42.jsonl"),
}


def load_last_row(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    return rows[-1]


def main() -> None:
    output_path = Path(
        "experiments/figures/learning_rate_sweep_seed42.png"
    )
    rows = {learning_rate: load_last_row(path) for learning_rate, path in RUNS.items()}
    learning_rates = list(RUNS)
    labels = ["5e-6", "1e-5", "2e-5"]

    metrics = [
        ("Validation reward", "val_reward", (0, 0.6)),
        ("Validation format reward", "val_format_reward", (0.8, 1.0)),
        ("Validation response length", "val_avg_response_length", None),
    ]
    colors = ["#147d92", "#d05a45", "#39805a"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, (title, field, ylim) in zip(axes, metrics):
        values = [rows[learning_rate][field] for learning_rate in learning_rates]
        ax.plot(
            labels,
            values,
            color="#344149",
            linewidth=1.5,
            marker="o",
            markersize=6,
        )
        for label, value, color in zip(labels, values, colors):
            ax.scatter(label, value, color=color, s=46, zorder=3)
            ax.annotate(
                f"{value:.3f}" if "length" not in title else f"{value:.1f}",
                (label, value),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("learning rate")
        ax.grid(True, color="#dfe7ea", linewidth=0.7, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ylim is not None:
            ax.set_ylim(*ylim)

    fig.suptitle(
        "Learning-rate sweep on GSM8K (seed 42)",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(output_path)


if __name__ == "__main__":
    main()
