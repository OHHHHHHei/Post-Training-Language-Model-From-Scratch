"""绘制 seed 42 的 off-policy 训练曲线和最终验证指标。"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


RUNS = {
    "Naive": Path("experiments/off_policy/naive/metrics_seed42.jsonl"),
    "No clip": Path("experiments/off_policy/noclip/metrics_seed42.jsonl"),
    "Token clip": Path("experiments/off_policy/clip/metrics_seed42.jsonl"),
    "GSPO": Path("experiments/off_policy/gspo/metrics_seed42.jsonl"),
}

COLORS = {
    "Naive": "#147d92",
    "No clip": "#d05a45",
    "Token clip": "#6b5ca5",
    "GSPO": "#39805a",
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


def plot_series(
    ax,
    rows: list[dict],
    field: str,
    color: str,
    validation: bool = False,
) -> None:
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


def configure_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#dfe7ea", linewidth=0.7, alpha=0.8)
    ax.set_xlim(1, 200)
    ax.set_xlabel("rollout step")


def save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(output_path)


def plot_curves(data: dict[str, list[dict]], output_path: Path) -> None:
    panels = [
        ("Reward", "mean_reward", "val_reward", "reward"),
        ("Format reward", "mean_format_reward", "val_format_reward", "reward"),
        ("Policy-gradient loss", "loss", None, "loss"),
        ("Gradient norm", "grad_norm", None, "norm"),
        ("Token entropy", "token_entropy", None, "entropy"),
        ("Validation response length", None, "val_avg_response_length", "tokens"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    for ax, (title, train_field, validation_field, ylabel) in zip(axes.flat, panels):
        for name, rows in data.items():
            color = COLORS[name]
            if train_field is not None:
                plot_series(ax, rows, train_field, color)
            if validation_field is not None:
                plot_series(ax, rows, validation_field, color, validation=True)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        configure_axes(ax)
        if ylabel == "reward":
            ax.set_ylim(0, 1)
        if title == "Policy-gradient loss":
            ax.set_yscale("symlog", linthresh=1e-3)
        if title == "Gradient norm":
            ax.set_yscale("symlog", linthresh=1.0)

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
        bbox_to_anchor=(0.5, 0.995),
        ncol=6,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, output_path)


def plot_final_metrics(data: dict[str, list[dict]], output_path: Path) -> None:
    names = list(RUNS)
    final_rows = {name: rows[-1] for name, rows in data.items()}
    panels = [
        ("Validation reward", "val_reward", (0, 1)),
        ("Validation format reward", "val_format_reward", (0, 1)),
        ("Validation response length", "val_avg_response_length", None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (ylabel, field, limits) in zip(axes, panels):
        values = [float(final_rows[name][field]) for name in names]
        bars = ax.bar(
            names,
            values,
            color=[COLORS[name] for name in names],
            width=0.68,
        )
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#dfe7ea", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        if limits is not None:
            ax.set_ylim(*limits)
        offset = max(values) * 0.025 if max(values) else 0.02
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    save_figure(fig, output_path)


def main() -> None:
    curve_path = Path(
        "experiments/figures/off_policy/variants_seed42.png"
    )
    final_path = Path(
        "experiments/figures/off_policy/final_seed42.png"
    )
    data = {name: load_rows(path) for name, path in RUNS.items()}
    plot_curves(data, curve_path)
    plot_final_metrics(data, final_path)


if __name__ == "__main__":
    main()
