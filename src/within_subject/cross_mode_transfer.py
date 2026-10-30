"""Within-Subject Cross-Mode Transfer Evaluation (Overt <-> Mimed).

Evaluates within-subject decoding across same-mode and cross-mode conditions
using a chronological split (first 160 trials/word train -> final 40 trials/word test):
  - Overt -> Overt (same-mode reference)
  - Overt -> Mimed (cross-mode transfer)
  - Mimed -> Mimed (same-mode reference)
  - Mimed -> Overt (cross-mode transfer)

Methodological notes:
  - Uses CAR, 0.2-1.8 s Hann FFT log-bandpower, train-only StandardScaler, and
    LogisticRegression(C=0.1) rather than EEGNet.
  - Consecutive 200-row groups are treated as provisional participant slots.
  - For the 4-word subset [5, 6, 7, 8], word 5 belongs to the first 5-word recording
    group and words 6-8 to the second 5-word recording group (personal communication
    with the dataset research team); block/session state remains coupled with word ID.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.cross_subject.loso_evaluation import extract_cross_subject_bandpower
from src.utils.data_loader import (
    DEFAULT_DATASET_DIR,
    N_SUBJECTS_PER_COHORT,
    PROJECT_ROOT,
    TRIAL_PER_SUBJECT,
)

BAND_SLICES: Dict[str, slice] = {
    "4-30 Hz": slice(0, 3),
    "30-45 Hz": slice(3, 4),
}

TRANSFER_CONDITIONS: Tuple[Tuple[str, str], ...] = (
    ("overt", "overt"),
    ("overt", "mime"),
    ("mime", "mime"),
    ("mime", "overt"),
)

N_TRAIN_PER_WORD: int = 160
N_TEST_PER_WORD: int = 40


def build_classifier() -> Pipeline:
    """Train-only StandardScaler + L2 LogisticRegression(C=0.1)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=1000, random_state=42),
    )


def make_subject_chronological_split(
    loaded: Dict[str, Dict[int, np.ndarray]],
    words: Sequence[int],
    slot: int,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split each word's 200 trials for `slot` into first 160 train and final 40 test."""
    train_x, train_y, test_x, test_y = [], [], [], []
    start = slot * TRIAL_PER_SUBJECT
    stop = start + TRIAL_PER_SUBJECT
    for class_id, word in enumerate(words):
        x = loaded[mode][word][start:stop]
        train_x.append(x[:N_TRAIN_PER_WORD])
        train_y.append(np.full(N_TRAIN_PER_WORD, class_id, dtype=np.int64))
        test_x.append(x[N_TRAIN_PER_WORD:])
        test_y.append(np.full(N_TEST_PER_WORD, class_id, dtype=np.int64))
    return (
        np.concatenate(train_x, axis=0),
        np.concatenate(train_y, axis=0),
        np.concatenate(test_x, axis=0),
        np.concatenate(test_y, axis=0),
    )


def evaluate_cross_mode_words(
    words: Sequence[int],
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Run all 4 transfer directions across 10 provisional participant slots."""
    loaded: Dict[str, Dict[int, np.ndarray]] = {"overt": {}, "mime": {}}
    for mode in ("overt", "mime"):
        for w in words:
            loaded[mode][w] = extract_cross_subject_bandpower(
                word_id=w,
                mode=mode,
                dataset_dir=dataset_dir,
            )

    chance = 1.0 / float(len(words))
    fold_rows: List[Dict[str, object]] = []

    for slot in range(N_SUBJECTS_PER_COHORT):
        by_mode = {
            mode: make_subject_chronological_split(loaded, words, slot, mode)
            for mode in ("overt", "mime")
        }
        for band_name, band_slice in BAND_SLICES.items():
            for train_mode, test_mode in TRANSFER_CONDITIONS:
                x_train, y_train, _, _ = by_mode[train_mode]
                _, _, x_test, y_test = by_mode[test_mode]
                xb_train = x_train[:, band_slice, :].reshape(len(y_train), -1)
                xb_test = x_test[:, band_slice, :].reshape(len(y_test), -1)

                clf = build_classifier()
                clf.fit(xb_train, y_train)
                pred = clf.predict(xb_test)

                bacc = float(balanced_accuracy_score(y_test, pred))
                acc = float(accuracy_score(y_test, pred))
                fold_rows.append(
                    {
                        "subject_slot": slot,
                        "train_mode": train_mode,
                        "test_mode": test_mode,
                        "direction": f"{train_mode}->{test_mode}",
                        "band": band_name,
                        "balanced_accuracy": bacc,
                        "accuracy": acc,
                        "accuracy_pct": round(bacc * 100.0, 2),
                        "chance": chance,
                        "n_train": len(y_train),
                        "n_test": len(y_test),
                        "classes": len(words),
                    }
                )

    folds_df = pd.DataFrame(fold_rows)
    summary_records: List[Dict[str, object]] = []
    for (band_name, direction), grp in folds_df.groupby(["band", "direction"], sort=False):
        vals = grp["balanced_accuracy"].to_numpy(dtype=float) * 100.0
        summary_records.append(
            {
                "band": band_name,
                "direction": direction,
                "n_classes": len(words),
                "chance_pct": round(chance * 100.0, 2),
                "mean_pct": round(float(vals.mean()), 2),
                "sd_pct": round(float(vals.std(ddof=1)), 2),
                "min_pct": round(float(vals.min()), 2),
                "max_pct": round(float(vals.max()), 2),
            }
        )

    summary_payload = {
        "task": "within-subject word decoding, same-mode and cross-mode",
        "words": list(words),
        "classes": len(words),
        "chance_pct": round(chance * 100.0, 2),
        "n_train_per_slot": N_TRAIN_PER_WORD * len(words),
        "n_test_per_slot": N_TEST_PER_WORD * len(words),
        "protocol": (
            "first 160 train and last 40 test trials per word and slot; "
            "CAR, 0.2-1.8 s Hann FFT log-bandpower, train-only StandardScaler, "
            "LogisticRegression(C=0.1)"
        ),
        "subject_slots": "provisional consecutive 200-row groups; source participant IDs absent",
        "limitation": (
            "chronological split within contiguous 200-trial word blocks; for words 5-8, word 5 "
            "belongs to the first 5-word recording group and words 6-8 to the second (personal "
            "communication with the dataset research team)"
        ),
        "summary": summary_records,
    }
    return folds_df, summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Within-Subject Cross-Mode Transfer evaluation.")
    parser.add_argument("--words", type=int, nargs="+", default=[5, 6, 7, 8])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "within_subject",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds_df, summary_payload = evaluate_cross_mode_words(
        words=args.words,
        dataset_dir=args.dataset_dir,
    )

    folds_path = args.output_dir / "cross_mode_4class_folds.csv"
    summary_path = args.output_dir / "cross_mode_4class_summary.json"
    folds_df.to_csv(folds_path, index=False)
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Within-Subject Cross-Mode Transfer Summary ===")
    summary_df = pd.DataFrame(summary_payload["summary"])
    print(summary_df.to_string(index=False))
    print(f"\nSaved folds to: {folds_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
