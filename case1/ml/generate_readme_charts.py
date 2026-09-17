"""Generate reproducible README charts from checked project artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "case1" / "output"
ASSETS_DIR = ROOT_DIR / "docs" / "assets"
METRICS_JSON = OUTPUT_DIR / "classifier_training_metrics.json"
DETECTIONS_CSV = OUTPUT_DIR / "all_fields_detections.csv"

GREEN = "#2f855a"
LIGHT_GREEN = "#68d391"
BLUE = "#3182ce"
ORANGE = "#dd6b20"
RED = "#c53030"
GRAY = "#718096"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_training_curves(metrics: dict) -> None:
    history = pd.DataFrame(metrics["history"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)

    axes[0].plot(history["epoch"], history["train_loss"], color=ORANGE, linewidth=2.2)
    axes[0].scatter(history["epoch"], history["train_loss"], color=ORANGE, s=18)
    axes[0].set(title="Снижение ошибки обучения", xlabel="Эпоха", ylabel="Train loss")

    axes[1].plot(
        history["epoch"],
        history["train_species_acc"],
        label="Train species accuracy",
        color=BLUE,
        linewidth=2,
    )
    axes[1].plot(
        history["epoch"],
        history["val_species_f1"],
        label="Validation species F1",
        color=GREEN,
        linewidth=2,
    )
    axes[1].plot(
        history["epoch"],
        history["val_stage_f1"],
        label="Validation stage F1",
        color=LIGHT_GREEN,
        linewidth=2,
    )
    axes[1].set(
        title="Качество на train и validation",
        xlabel="Эпоха",
        ylabel="Значение метрики",
        ylim=(0.75, 1.02),
    )
    axes[1].legend(frameon=False, loc="lower right")

    fig.suptitle(
        f"EfficientNet-B0 · {metrics['epochs']} эпох · Focal Loss",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(ASSETS_DIR / "training_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_class_recall(metrics: dict) -> None:
    recall = metrics["test_metrics"]["species_recall_per_class"]
    labels = ["Бодяк", "Вьюнок", "Пырей", "Пшеница / фон"]
    keys = ["field_thistle", "field_bindweed", "couch_grass", "crop_wheat"]
    values = [recall[key] * 100 for key in keys]
    min_value = min(values)
    colors = [RED if value == min_value else GREEN for value in values]

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    bars = ax.barh(labels, values, color=colors, height=0.62)
    ax.invert_yaxis()
    ax.set(xlim=(0, 105), xlabel="Recall на локальном test split, %")
    ax.set_title("Полнота распознавания по классам", fontsize=15, fontweight="bold")
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=6)
    ax.text(
        0,
        -0.9,
        "Красным выделен класс с минимальным Recall: главный текущий риск — вьюнок.",
        color=GRAY,
        fontsize=10,
    )
    fig.savefig(ASSETS_DIR / "class_recall.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_field_overview(detections: pd.DataFrame) -> None:
    species_labels = {
        "crop_wheat": "Пшеница / фон",
        "couch_grass": "Пырей",
        "field_bindweed": "Вьюнок",
        "field_thistle": "Бодяк",
        "unknown": "Не определено",
    }
    counts = detections["species"].value_counts()
    ordered_keys = [key for key in species_labels if key in counts]
    labels = [species_labels[key] for key in ordered_keys]
    values = [int(counts[key]) for key in ordered_keys]
    review = detections["review_required"].astype(str).str.lower().eq("true")
    review_counts = [int((~review).sum()), int(review.sum())]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    bars = axes[0].bar(labels, values, color=[BLUE, GREEN, ORANGE, RED, GRAY][: len(labels)])
    axes[0].set(title="Результат прогона 5 DJI-снимков", ylabel="Количество объектов")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].bar_label(bars, padding=4)

    axes[1].pie(
        review_counts,
        labels=["Автоматический результат", "Ручная проверка"],
        autopct="%1.1f%%",
        colors=[GREEN, ORANGE],
        startangle=90,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
    )
    axes[1].set_title("Human-in-the-loop safety gate")
    fig.suptitle(
        f"Полевой demo-run · {len(detections)} candidate detections",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(ASSETS_DIR / "field_demo_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    detections = pd.read_csv(DETECTIONS_CSV)
    configure_style()
    save_training_curves(metrics)
    save_class_recall(metrics)
    save_field_overview(detections)
    print(f"Generated 3 charts in {ASSETS_DIR}")


if __name__ == "__main__":
    main()
