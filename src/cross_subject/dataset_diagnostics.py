"""Cross-Subject Dataset Diagnostics (Recording-Group Confound & Slot Specificity).

Quantifies the three core dataset factors limiting zero-calibration cross-subject generalization:
  1. Recording-Group Bias: Same-half word pairs (`same_half=True`) vs. Cross-half word pairs (`same_half=False`).
  2. Recording-Group Confound on Unseen Words & Unseen Slot:
     Trains a recording-half classifier (first 5 vs. last 5 words) on 8 words of 9 source slots
     and evaluates on the 2 held-out words of the 10th unseen slot.
  3. High Slot Specificity & Orthogonal Cross-Slot Word Contrasts:
     Evaluates 10-way nearest-centroid slot identification across separate word blocks
     and compares within-slot vs. cross-slot word contrast cosine similarity.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.cross_subject.loso_evaluation import extract_cross_subject_bandpower
from src.utils.data_loader import (
    ACTIVE_END_SAMPLE,
    ACTIVE_START_SAMPLE,
    DEFAULT_DATASET_DIR,
    N_SUBJECTS_PER_COHORT,
    PROJECT_ROOT,
    SAMPLING_RATE_HZ,
    TRIAL_PER_SUBJECT,
    resolve_word_npz_path,
)

DIAGNOSTIC_BANDS: Tuple[Tuple[float, float], ...] = (
    (1.0, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 30.0),
    (30.0, 45.0),
)
DIAGNOSTIC_ARMS: Dict[str, List[int]] = {
    "low": [1, 2, 3],
    "gamma": [4],
    "delta": [0],
}


def run_recording_half_control(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> Dict[str, object]:
    """Predict recording half (first 5 vs. last 5 words) on 2 unseen words of an unseen 10th slot."""
    words = list(range(11, 21))
    slots = np.repeat(np.arange(N_SUBJECTS_PER_COHORT), TRIAL_PER_SUBJECT)
    report: Dict[str, object] = {
        "task": "classify first-five (words 11-15) vs last-five (words 16-20) recording group",
        "test": "unseen subject slot and two unseen word IDs (one per group)",
        "results": {},
    }

    for mode in ("mime", "overt"):
        loaded = {
            w: extract_cross_subject_bandpower(word_id=w, mode=mode, dataset_dir=dataset_dir)
            for w in words
        }
        for band_label, band_slice in (("4-30 Hz", slice(0, 3)), ("30-45 Hz", slice(3, 4))):
            x_by_word = {w: loaded[w][:, band_slice, :].reshape(2000, -1) for w in words}
            pair_results = []
            for first_word, second_word in zip(range(11, 16), range(16, 21)):
                held_words = {first_word, second_word}
                train_words = [w for w in words if w not in held_words]
                scores = []
                for held_subject in range(N_SUBJECTS_PER_COHORT):
                    x_train = np.concatenate([x_by_word[w][slots != held_subject] for w in train_words])
                    y_train = np.concatenate([np.full(1800, w >= 16, dtype=np.int64) for w in train_words])
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
                    clf.fit(x_train, y_train)
                    x_test = np.concatenate([x_by_word[w][slots == held_subject] for w in (first_word, second_word)])
                    y_test = np.repeat([0, 1], TRIAL_PER_SUBJECT)
                    scores.append(float(balanced_accuracy_score(y_test, clf.predict(x_test))))
                pair_results.append(
                    {
                        "test_words": [first_word, second_word],
                        "folds": scores,
                        "mean": float(np.mean(scores)),
                    }
                )

            subject_mean = np.mean([row["folds"] for row in pair_results], axis=0)
            key = f"{mode} | full | {band_label}"
            report["results"][key] = {
                "test_word_pairs": pair_results,
                "per_subject_mean": subject_mean.tolist(),
                "mean": float(subject_mean.mean()),
                "sd_subject": float(subject_mean.std(ddof=1)),
                "min_subject": float(subject_mean.min()),
                "above_chance_subjects": int((subject_mean > 0.5).sum()),
            }
    return report


def _extract_five_bands(
    word_id: int,
    mode: str,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> np.ndarray:
    """Extract 5-band (Delta, Theta, Alpha, Beta, Gamma) Hann FFT log-bandpower."""
    path = resolve_word_npz_path(mode=mode, word_id=word_id, dataset_dir=dataset_dir)
    parts: List[np.ndarray] = []
    with np.load(path, allow_pickle=False) as data:
        scale = data["scale"].astype(np.float32)
        raw = data["eeg"]
        for start in range(0, 2000, 100):
            x = raw[start : start + 100].astype(np.float32) * scale
            x -= x.mean(axis=2, keepdims=True)
            x = x[:, ACTIVE_START_SAMPLE:ACTIVE_END_SAMPLE, :].copy()
            x -= x.mean(axis=1, keepdims=True)
            x *= np.hanning(x.shape[1]).astype(np.float32)[None, :, None]
            power = np.abs(rfft(x, axis=1, workers=1)) ** 2
            hz = rfftfreq(x.shape[1], 1.0 / float(SAMPLING_RATE_HZ))
            bp = np.stack(
                [power[:, (hz >= lo) & (hz < hi), :].mean(axis=1) for lo, hi in DIAGNOSTIC_BANDS],
                axis=1,
            )
            parts.append(np.log10(bp + 1e-12).astype(np.float32))
    return np.concatenate(parts, axis=0)


def _nearest_subject_centroid_accuracy(train_block: np.ndarray, test_block: np.ndarray) -> np.ndarray:
    """10-way subject identification accuracy from a separate word block's subject centroids."""
    mean = train_block.mean(axis=0)
    sd = train_block.std(axis=0) + 1e-6
    centroids = ((train_block - mean) / sd).reshape(10, 200, -1).mean(axis=1)
    samples = (test_block - mean) / sd
    distances = ((samples[:, None] - centroids[None]) ** 2).sum(axis=2)
    predictions = distances.argmin(axis=1)
    true_slots = np.repeat(np.arange(10), 200)
    return (predictions == true_slots).reshape(10, 200).mean(axis=1)


def _summarize_subject_signature_pair(
    mode: str,
    word_a: int,
    word_b: int,
    arm_name: str,
    band_indices: List[int],
    cache: Dict[Tuple[str, int], np.ndarray],
) -> Dict[str, object]:
    """Compute subject identification accuracy and within- vs cross-subject word contrast cosines."""
    x = [cache[(mode, w)][:, band_indices, :].reshape(2000, -1) for w in (word_a, word_b)]
    forward_id = _nearest_subject_centroid_accuracy(x[0], x[1])
    reverse_id = _nearest_subject_centroid_accuracy(x[1], x[0])

    person_means = np.stack([v.reshape(10, 200, -1).mean(axis=1) for v in x])
    pooled = np.concatenate(x).reshape(2, 10, 200, -1)
    residual = pooled - pooled.mean(axis=2, keepdims=True)
    sd = residual.reshape(-1, residual.shape[-1]).std(axis=0) + 1e-6

    contrast = (person_means[1] - person_means[0]) / sd
    early_contrast = (pooled[1, :, :100].mean(axis=1) - pooled[0, :, :100].mean(axis=1)) / sd
    late_contrast = (pooled[1, :, 100:].mean(axis=1) - pooled[0, :, 100:].mean(axis=1)) / sd

    within_cosine = (early_contrast * late_contrast).sum(axis=1) / (
        np.linalg.norm(early_contrast, axis=1) * np.linalg.norm(late_contrast, axis=1) + 1e-12
    )
    unit_contrast = contrast / (np.linalg.norm(contrast, axis=1, keepdims=True) + 1e-12)
    cosine_matrix = unit_contrast @ unit_contrast.T
    pairwise_cosines = cosine_matrix[np.triu_indices(10, k=1)]

    return {
        "mode": mode,
        "words": [word_a, word_b],
        "band": arm_name,
        "subject_identity_chance": 0.1,
        "identity_cross_word_a_to_b_mean": float(forward_id.mean()),
        "identity_cross_word_b_to_a_mean": float(reverse_id.mean()),
        "contrast_pairwise_cosine_mean": float(pairwise_cosines.mean()),
        "contrast_pairwise_cosine_median": float(np.median(pairwise_cosines)),
        "contrast_pairwise_positive_count": int((pairwise_cosines > 0).sum()),
        "contrast_pairwise_count": int(len(pairwise_cosines)),
        "contrast_first_last_within_person_cosine_mean": float(within_cosine.mean()),
    }


def run_subject_signature_diagnostics(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Run subject identification and contrast vector cosine analysis across all 80 same-day pairs."""
    results: List[Dict[str, object]] = []
    for mode in ("mime", "overt"):
        for cohort in (1, 2):
            words = list(range(1, 11)) if cohort == 1 else list(range(11, 21))
            cache = {(mode, w): _extract_five_bands(w, mode, dataset_dir=dataset_dir) for w in words}
            for half_words in (words[:5], words[5:]):
                for word_a, word_b in combinations(half_words, 2):
                    for arm_name, band_indices in DIAGNOSTIC_ARMS.items():
                        results.append(
                            _summarize_subject_signature_pair(
                                mode=mode,
                                word_a=word_a,
                                word_b=word_b,
                                arm_name=arm_name,
                                band_indices=band_indices,
                                cache=cache,
                            )
                        )

    payload = {
        "method": "CAR, 0.2-1.8 s Hann log bandpower across 80 same-half pairs",
        "results": results,
    }
    table = pd.DataFrame(
        [
            {
                "mode": r["mode"],
                "cohort": 1 if r["words"][0] <= 10 else 2,
                "word_0": r["words"][0],
                "word_1": r["words"][1],
                "arm": r["band"],
                "identity_ab": r["identity_cross_word_a_to_b_mean"],
                "identity_ba": r["identity_cross_word_b_to_a_mean"],
                "contrast_cosine_mean": r["contrast_pairwise_cosine_mean"],
                "within_person_cosine_mean": r["contrast_first_last_within_person_cosine_mean"],
                "contrast_positive_fraction": r["contrast_pairwise_positive_count"] / 45.0,
            }
            for r in results
        ]
    )
    return payload, table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Cross-Subject Dataset Diagnostics (Recording-half confound & Subject specificity)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "cross_subject",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("1. Running Recording-Half Confound Control (Unseen Subject & Unseen Words)...")
    half_report = run_recording_half_control(dataset_dir=args.dataset_dir)
    half_path = args.output_dir / "recording_half_control.json"
    half_path.write_text(json.dumps(half_report, indent=2), encoding="utf-8")
    for key, metrics in half_report["results"].items():
        print(f"   {key:24s} -> {metrics['mean'] * 100:.2f}% ± {metrics['sd_subject'] * 100:.2f}%")

    print("\n2. Running Subject Specificity & Contrast Vector Cosine Diagnostics...")
    sig_payload, sig_table = run_subject_signature_diagnostics(dataset_dir=args.dataset_dir)
    sig_json_path = args.output_dir / "subject_signature.json"
    sig_csv_path = args.output_dir / "all_pair_diagnostics.csv"
    sig_json_path.write_text(json.dumps(sig_payload, indent=2), encoding="utf-8")
    sig_table.to_csv(sig_csv_path, index=False)

    summary = (
        sig_table.groupby("arm")[
            ["identity_ab", "identity_ba", "contrast_cosine_mean", "within_person_cosine_mean"]
        ]
        .agg(["mean", "median"])
        .round(3)
    )
    print(summary.to_string())
    print(f"\nSaved diagnostics to:\n  - {half_path}\n  - {sig_json_path}\n  - {sig_csv_path}")


if __name__ == "__main__":
    main()
