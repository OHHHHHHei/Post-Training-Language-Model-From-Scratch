"""绘制 seed 42 的 Prompt ablation 结果。"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


RUNS = {
    "Question-only": Path(
        "experiments/prompt_ablation/question_only/metrics_seed42.jsonl"
    ),
    "R1 zero-shot": Path("experiments/on_policy/standard/metrics_seed42.jsonl"),
    "R1 three-shot": Path(
        "experiments/prompt_ablation/three_shot/metrics_seed42.jsonl"
    ),
}

COLORS = {
    "Question-only": "#147d92",
    "R1 zero-shot": "#d05a45",
    "R1 three-shot": "#39805a",
}


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def plot_series(ax, rows: list[dict], field: str, color: str, validation: bool):
    selected = [row for row in rows if field in row and row[field] is not None]
    if not selected:
        return
    ax.plot(
        [row["step"] for row in selected],
        [row[field] for row in selected],
        color=color,
        linestyle="--" if validation else "-",
        linewidth=1.7,
        marker="o" if validation else None,
        markersize=3.0 if validation else None,
    )


def main() -> None:
    output_path = Path("experiments/figures/prompt_ablation_seed42.png")
    data = {name: load_rows(path) for name, path in RUNS.items()}

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), constrained_layout=True)
    panels = [
        ("Reward", "mean_reward", "val_reward", (0, 0.7)),
        ("Format reward", "mean_format_reward", "val_format_reward", (0, 1.05)),
        ("Response length", None, "val_avg_response_length", None),
        ("Token entropy", "token_entropy", None, None),
    ]
    for ax, (title, train_field, validation_field, ylim) in zip(axes.flat, panels):
        for name, rows in data.items():
            if train_field is not None:
                plot_series(ax, rows, train_field, COLORS[name], validation=False)
            if validation_field is not None:
                plot_series(ax, rows, validation_field, COLORS[name], validation=True)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("rollout step")
        ax.grid(True, color="#dfe7ea", linewidth=0.7, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(1, 200)
        if ylim is not None:
            ax.set_ylim(*ylim)

    method_handles = [
        Line2D([0], [0], color=COLORS[name], linewidth=2, label=name)
        for name in RUNS
    ]
    style_handles = [
        Line2D([0], [0], color="#344149", linewidth=1.8, label="train"),
        Line2D(
            [0],
            [0],
            color="#344149",
            linewidth=1.8,
            linestyle="--",
            marker="o",
            markersize=3,
            label="validation",
        ),
    ]
    fig.legend(
        handles=method_handles + style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "Prompt ablation on GSM8K (seed 42)",
        fontsize=14,
        fontweight="bold",
        y=1.08,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(output_path)


if __name__ == "__main__":
    main()
