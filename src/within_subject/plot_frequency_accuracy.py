"""Generate the Within-Subject EEGNet frequency-band accuracy chart and CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.data_loader import PROJECT_ROOT

SUBJECT_SCORES: Dict[str, Dict[str, List[float]]] = {
    "Delta\n1-4 Hz": {
        "Overt": [90.38, 89.10, 77.50, 88.75, 85.62, 92.95, 92.50, 91.88, 88.75, 98.72],
        "Mimed": [82.69, 72.50, 97.44, 85.26, 94.38, 93.59, 96.15, 75.64, 90.00, 100.00],
    },
    "Theta\n4-8 Hz": {
        "Overt": [94.87, 95.00, 75.00, 87.80, 78.85, 93.12, 92.50, 97.44, 98.75, 98.12],
        "Mimed": [95.51, 82.50, 95.00, 92.31, 100.00, 93.59, 96.25, 86.88, 99.39, 94.08],
    },
    "Alpha\n8-12 Hz": {
        "Overt": [93.12, 91.88, 72.50, 76.25, 76.28, 97.50, 87.50, 95.62, 97.50, 100.00],
        "Mimed": [99.38, 78.75, 96.25, 94.87, 100.00, 90.00, 98.12, 80.00, 99.39, 96.79],
    },
    "Beta\n12-30 Hz": {
        "Overt": [97.44, 91.25, 75.00, 77.50, 78.12, 99.38, 86.54, 96.79, 96.15, 99.38],
        "Mimed": [100.00, 82.50, 98.12, 95.62, 100.00, 99.36, 98.75, 91.25, 98.72, 100.00],
    },
    "Gamma\n30-80 Hz": {
        "Overt": [100.00, 95.62, 74.38, 93.59, 83.97, 100.00, 88.75, 99.36, 95.00, 100.00],
        "Mimed": [100.00, 93.12, 100.00, 99.36, 100.00, 98.03, 100.00, 89.10, 100.00, 99.36],
    },
    "Broadband\n0.5-45 Hz": {
        "Overt": [86.88, 89.74, 76.25, 88.75, 82.69, 94.38, 89.38, 91.25, 90.38, 96.79],
        "Mimed": [94.38, 92.95, 71.88, 80.62, 84.62, 98.12, 90.62, 95.00, 96.15, 98.08],
    },
}


def generate_frequency_accuracy_outputs(output_dir: Path) -> Tuple[Path, Path]:
    """Save the Within-Subject EEGNet frequency accuracy CSV and PNG bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    bands = list(SUBJECT_SCORES.keys())

    rows = []
    for band in bands:
        band_clean = band.replace("\n", " ")
        for mode in ("Overt", "Mimed"):
            for subject_slot, score in enumerate(SUBJECT_SCORES[band][mode]):
                rows.append(
                    {
                        "band": band_clean,
                        "mode": mode,
                        "subject_slot": subject_slot,
                        "accuracy_percent": round(float(score), 2),
                    }
                )

    df = pd.DataFrame(rows)
    csv_path = output_dir / "within_subject_eegnet_frequency_scores.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(14, 7.875), dpi=180)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("white")

    x = np.arange(len(bands))
    width = 0.34
    for mode, shift, color in (("Overt", -width / 2, "#17607a"), ("Mimed", width / 2, "#d88554")):
        means = [float(np.mean(SUBJECT_SCORES[b][mode])) for b in bands]
        sds = [float(np.std(SUBJECT_SCORES[b][mode], ddof=1)) for b in bands]
        bars = ax.bar(x + shift, means, width, label=mode, color=color, zorder=3)
        ax.errorbar(x + shift, means, yerr=sds, fmt="none", ecolor="#263746", capsize=5, lw=1.6, zorder=4)
        for bar, mean, sd in zip(bars, means, sds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + sd + 1.6,
                f"{mean:.1f}%\n±{sd:.1f}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                weight="bold",
                color="#18354a",
            )

    ax.set_ylim(0, 119)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Test accuracy (%)", fontsize=15, weight="bold")
    ax.set_xticks(x, bands)
    ax.tick_params(axis="x", length=0, pad=12)
    ax.grid(axis="y", color="#d9e1e8", linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=14)
    fig.suptitle(
        "Within-subject EEGNet accuracy by frequency band",
        x=0.07,
        y=0.96,
        ha="left",
        fontsize=22,
        weight="bold",
        color="#163448",
    )
    fig.text(
        0.07,
        0.90,
        "Bars: mean across 10 subject scores per mode   |   Error bars: sample SD",
        fontsize=12.5,
        color="#526779",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.14)

    png_path = output_dir / "within_subject_eegnet_frequency_accuracy.png"
    fig.savefig(png_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return csv_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Within-Subject EEGNet frequency accuracy chart and CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "within_subject",
    )
    args = parser.parse_args()
    csv_path, png_path = generate_frequency_accuracy_outputs(args.output_dir)
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved PNG to: {png_path}")


if __name__ == "__main__":
    main()
