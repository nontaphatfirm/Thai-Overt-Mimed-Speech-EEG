"""Generate Before vs. After 5-step EEG preprocessing comparison figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.utils.data_loader import (
    ACTIVE_END_SAMPLE,
    ACTIVE_END_SEC,
    ACTIVE_START_SAMPLE,
    ACTIVE_START_SEC,
    DEFAULT_DATASET_DIR,
    PROJECT_ROOT,
    preprocess_single_trial_waveform,
    resolve_word_npz_path,
)


def plot_preprocessing_before_after(
    output_path: Path,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    word_id: int = 1,
    trial_idx: int = 0,
    channel_name: str = "C3",
) -> Path:
    """Create and save the raw vs. preprocessed EEG comparison figure."""
    npz_path = resolve_word_npz_path("overt", word_id, dataset_dir=dataset_dir)
    with np.load(npz_path, allow_pickle=True) as data:
        raw_adc = data["eeg"].astype(np.float32)
        scale = data["scale"].astype(np.float32)
        fs = int(data["fs"])
        channels = [str(c) for c in data["channels"].tolist()]

    ch_idx = channels.index(channel_name)
    raw_trace = raw_adc[trial_idx, :, ch_idx]
    t_full = np.arange(raw_trace.shape[0]) / float(fs)

    sos_drift = butter(2, 3.0, btype="low", fs=fs, output="sos")
    drift_trace = sosfiltfilt(sos_drift, raw_trace)

    eeg_uv = raw_adc[trial_idx : trial_idx + 1] * scale[trial_idx : trial_idx + 1]
    clean_windowed = preprocess_single_trial_waveform(
        eeg_uv,
        fs=fs,
        l_freq=4.0,
        h_freq=30.0,
        start_sample=ACTIVE_START_SAMPLE,
        end_sample=ACTIVE_END_SAMPLE,
    )[0, :, ch_idx]
    t_active = np.linspace(ACTIVE_START_SEC, ACTIVE_END_SEC, clean_windowed.shape[0], endpoint=False)

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Segoe UI", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax_raw, ax_clean) = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=200)
    fig.suptitle(
        f"EEG Signal Before vs. After 5-Step Preprocessing (Overt Word {word_id:02d}, Channel {channel_name})",
        fontsize=14.5,
        fontweight="bold",
        color="#1B2A4A",
        y=0.97,
    )

    # Left panel: Raw EEG
    ax_raw.axvspan(0.0, ACTIVE_START_SEC, color="#FADBD8", alpha=0.35, label="Trimmed Window (Outside 0.2–1.8 s)")
    ax_raw.axvspan(ACTIVE_END_SEC, 2.0, color="#FADBD8", alpha=0.35)
    ax_raw.axvline(ACTIVE_START_SEC, color="#C0392B", linestyle=":", linewidth=1.2)
    ax_raw.axvline(ACTIVE_END_SEC, color="#C0392B", linestyle=":", linewidth=1.2)
    ax_raw.plot(t_full, raw_trace, color="#566573", linewidth=0.8, alpha=0.85, label="Raw EEG (Unscaled ADC + High-Freq Noise)")
    ax_raw.plot(t_full, drift_trace, color="#E74C3C", linestyle="--", linewidth=2.2, label="Slow Baseline Drift (<3 Hz)")
    ax_raw.set_title("Before: Raw Unscaled EEG (0.0–2.0 s)", fontsize=12.5, fontweight="bold", color="#1B2A4A", pad=10)
    ax_raw.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    ax_raw.set_ylabel("Raw Amplitude (ADC Counts)", fontsize=11, fontweight="bold", color="#A93226")
    ax_raw.set_xlim(0.0, 2.0)
    ax_raw.grid(True, linestyle="--", alpha=0.35)
    ax_raw.spines["top"].set_visible(False)
    ax_raw.spines["right"].set_visible(False)
    ax_raw.legend(loc="lower left", frameon=True, fontsize=9.5)

    # Right panel: Preprocessed EEG
    ax_clean.axhspan(-2.0, 2.0, color="#D6EAF8", alpha=0.35, label="Standardized Range (±2 SD)")
    ax_clean.axhline(0.0, color="#34495E", linestyle="--", linewidth=1.1, label="Zero Baseline (Mean = 0)")
    ax_clean.plot(t_active, clean_windowed, color="#1F618D", linewidth=1.5, label="Preprocessed EEG (Scale + CAR + 4–30 Hz + Z-score)")
    ax_clean.set_title("After: Preprocessed EEG (Active Window 0.2–1.8 s)", fontsize=12.5, fontweight="bold", color="#1B2A4A", pad=10)
    ax_clean.set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    ax_clean.set_ylabel("Standardized Amplitude (Z-score)", fontsize=11, fontweight="bold", color="#1F618D")
    ax_clean.set_xlim(ACTIVE_START_SEC, ACTIVE_END_SEC)
    ax_clean.grid(True, linestyle="--", alpha=0.35)
    ax_clean.spines["top"].set_visible(False)
    ax_clean.spines["right"].set_visible(False)
    ax_clean.legend(loc="lower left", frameon=True, fontsize=9.5)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Before vs. After EEG preprocessing figure.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "within_subject" / "eeg_preprocessing_before_after.png",
        help="Output PNG path.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()
    saved = plot_preprocessing_before_after(output_path=args.output, dataset_dir=args.dataset_dir)
    print(f"Saved preprocessing figure to: {saved}")


if __name__ == "__main__":
    main()
