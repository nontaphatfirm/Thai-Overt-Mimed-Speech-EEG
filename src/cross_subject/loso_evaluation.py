"""Cross-Subject Strict LOSO Evaluation and Few-Shot Calibration Pipeline.

Evaluates 180 binary word pairs within each 10-word participant cohort (Cohort 1: words 1-10;
Cohort 2: words 11-20) in Overt and Mimed modes using 10-fold Leave-One-Subject-Slot-Out (LOSO):
  - 80 same-recording-half pairs (`same_half=True`: words 1-5, 6-10, 11-15, or 16-20)
  - 100 cross-recording-half pairs (`same_half=False`: first-five vs. last-five words)

Methodological notes:
  - Released NPZ files do not include explicit participant IDs; consecutive 200-row slices
    are treated as provisional participant slots (10 slots per cohort).
  - `zero_low_logistic` (4-30 Hz) and `zero_high_spatial_lda` (30-45 Hz) fit all scalers
    and classifiers strictly on the 9 source slots without access to target-slot data.
  - `target_first20` fits a new classifier from scratch on only the target slot's first 20
    trials/word (30-45 Hz for mime, 4-30 Hz for overt) and tests on the remaining 180
    trials/word of the same blocks (personalized within-block baseline, no source subjects).
  - `opposite_target_only_{low,high}` fits from scratch on 20 evenly spaced trials/word
    (indices 0, 10, ..., 190) from the target slot's opposite mode and tests on all 200
    trials/word in the target mode.
  - `source_plus_opposite_{low,high}` fits a sample-weighted classifier on the 9 source
    slots (target mode) combined with the 20 opposite-mode calibration trials/word.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.data_loader import (
    ACTIVE_END_SAMPLE,
    ACTIVE_START_SAMPLE,
    DEFAULT_DATASET_DIR,
    N_SUBJECTS_PER_COHORT,
    PROJECT_ROOT,
    SAMPLING_RATE_HZ,
    TRIAL_PER_SUBJECT,
    VOCABULARY_METADATA,
    is_same_recording_half,
    resolve_word_npz_path,
)

CROSS_SUBJECT_BANDS: Tuple[Tuple[float, float], ...] = (
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 30.0),
    (30.0, 45.0),
)

SUBJECT_GROUPS: np.ndarray = np.tile(np.repeat(np.arange(N_SUBJECTS_PER_COHORT), TRIAL_PER_SUBJECT), 2)
PAIR_LABELS: np.ndarray = np.repeat([0, 1], N_SUBJECTS_PER_COHORT * TRIAL_PER_SUBJECT)


def word_pair_category(word_a: int, word_b: int) -> str:
    """Return descriptive semantic/recording category for a word pair."""
    pair = (word_a, word_b)
    if pair == (1, 2):
        return "opposite_action"
    if 7 <= word_a < word_b <= 10:
        return "emotion"
    if 16 <= word_a < word_b <= 20:
        return "body_part"
    if pair == (12, 13):
        return "opposite_response"
    if pair == (3, 4):
        return "elimination"
    if pair == (5, 6):
        return "hunger_thirst"
    return "other"


def extract_cross_subject_bandpower(
    word_id: int,
    mode: str,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> np.ndarray:
    """Extract CAR + 0.2-1.8s Hann-FFT log10 bandpower of shape `(2000, 4, 32)`."""
    path = resolve_word_npz_path(mode=mode, word_id=word_id, dataset_dir=dataset_dir)
    parts: List[np.ndarray] = []
    with np.load(path, allow_pickle=False) as data:
        scale = data["scale"].astype(np.float32)
        raw = data["eeg"]
        for start in range(0, 2000, 100):
            x = raw[start : start + 100].astype(np.float32) * scale
            x -= x.mean(axis=2, keepdims=True)
            segment = x[:, ACTIVE_START_SAMPLE:ACTIVE_END_SAMPLE, :].copy()
            segment -= segment.mean(axis=1, keepdims=True)
            n_times = segment.shape[1]
            segment *= np.hanning(n_times).astype(np.float32)[None, :, None]
            power = np.abs(rfft(segment, axis=1, workers=1)) ** 2
            freqs = rfftfreq(n_times, 1.0 / float(SAMPLING_RATE_HZ))
            bp = np.stack(
                [power[:, (freqs >= f_lo) & (freqs < f_hi), :].mean(axis=1) for f_lo, f_hi in CROSS_SUBJECT_BANDS],
                axis=1,
            )
            parts.append(np.log10(bp + 1e-12).astype(np.float32))
    return np.concatenate(parts, axis=0)


def build_logistic_pipeline() -> Pipeline:
    """Train-only StandardScaler + L2 Logistic Regression (C=0.1)."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))


def build_spatial_lda_pipeline() -> Pipeline:
    """Train-only StandardScaler + Ledoit-Wolf Shrinkage LDA."""
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def evaluate_pair_all_arms(
    word_a: int,
    word_b: int,
    mode: str,
    cohort: int,
    features_by_mode: Dict[str, Dict[int, np.ndarray]],
) -> List[Dict[str, object]]:
    """Evaluate zero-calibration and few-shot calibration arms on a word pair across 10 LOSO folds."""
    opposite_mode = "overt" if mode == "mime" else "mime"
    data_target = features_by_mode[mode]
    data_opposite = features_by_mode[opposite_mode]

    x_low = np.concatenate([data_target[word_a][:, :3], data_target[word_b][:, :3]]).reshape(4000, -1)
    x_high_raw = np.concatenate([data_target[word_a][:, 3:4], data_target[word_b][:, 3:4]])
    x_high_spatial = (x_high_raw - x_high_raw.mean(axis=2, keepdims=True)).reshape(4000, -1)
    x_high = x_high_raw.reshape(4000, -1)

    x_opp_low = np.concatenate([data_opposite[word_a][:, :3], data_opposite[word_b][:, :3]]).reshape(4000, -1)
    x_opp_high = np.concatenate([data_opposite[word_a][:, 3:4], data_opposite[word_b][:, 3:4]]).reshape(4000, -1)
    x_within_cal = x_high if mode == "mime" else x_low

    cohort_label = "1-10" if cohort == 1 else "11-20"
    same_half = is_same_recording_half(word_a, word_b)
    w_type = word_pair_category(word_a, word_b)
    w0_th = str(VOCABULARY_METADATA[word_a]["thai"])
    w1_th = str(VOCABULARY_METADATA[word_b]["thai"])

    fold_rows: List[Dict[str, object]] = []
    for slot in range(N_SUBJECTS_PER_COHORT):
        source_mask = SUBJECT_GROUPS != slot
        source_idx = np.flatnonzero(source_mask)
        target_idx = np.concatenate(
            [
                np.arange(cls * 2000 + slot * 200, cls * 2000 + slot * 200 + 200)
                for cls in range(2)
            ]
        )

        cal_first20_idx = np.concatenate([target_idx[:20], target_idx[200:220]])
        rem_test_idx = np.concatenate([target_idx[20:200], target_idx[220:400]])

        standard_arms = [
            ("zero_low_logistic", x_low, source_idx, target_idx, build_logistic_pipeline),
            ("zero_high_spatial_lda", x_high_spatial, source_idx, target_idx, build_spatial_lda_pipeline),
            ("target_first20", x_within_cal, cal_first20_idx, rem_test_idx, build_logistic_pipeline),
        ]
        for arm_name, feat_mat, tr_idx, te_idx, builder in standard_arms:
            clf = builder().fit(feat_mat[tr_idx], PAIR_LABELS[tr_idx])
            pred = clf.predict(feat_mat[te_idx])
            score_pp = float((pred == PAIR_LABELS[te_idx]).mean() * 100.0)
            fold_rows.append(
                {
                    "mode": mode,
                    "cohort": cohort_label,
                    "word_0": word_a,
                    "word_1": word_b,
                    "word_0_th": w0_th,
                    "word_1_th": w1_th,
                    "same_half": same_half,
                    "word_type": w_type,
                    "arm": arm_name,
                    "slot": slot,
                    "score_pp": score_pp,
                }
            )

        opp_cal_idx = np.r_[
            slot * 200 + np.arange(0, 200, 10),
            2000 + slot * 200 + np.arange(0, 200, 10),
        ]
        for band_suffix, x_targ_band, x_opp_band in (
            ("low", x_low, x_opp_low),
            ("high", x_high, x_opp_high),
        ):
            xs, ys = x_targ_band[source_idx], PAIR_LABELS[source_idx]
            xc, yc = x_opp_band[opp_cal_idx], PAIR_LABELS[opp_cal_idx]
            xt, yt = x_targ_band[target_idx], PAIR_LABELS[target_idx]

            clf_opp = build_logistic_pipeline().fit(xc, yc)
            pred_opp = clf_opp.predict(xt)
            fold_rows.append(
                {
                    "mode": mode,
                    "cohort": cohort_label,
                    "word_0": word_a,
                    "word_1": word_b,
                    "word_0_th": w0_th,
                    "word_1_th": w1_th,
                    "same_half": same_half,
                    "word_type": w_type,
                    "arm": f"opposite_target_only_{band_suffix}",
                    "slot": slot,
                    "score_pp": float((pred_opp == yt).mean() * 100.0),
                }
            )

            both_x = np.concatenate([xs, xc], axis=0)
            both_y = np.concatenate([ys, yc], axis=0)
            weights = np.r_[np.full(len(ys), 0.5), np.full(len(yc), 45.0)]
            clf_both = build_logistic_pipeline()
            clf_both.fit(both_x, both_y, logisticregression__sample_weight=weights)
            pred_both = clf_both.predict(xt)
            fold_rows.append(
                {
                    "mode": mode,
                    "cohort": cohort_label,
                    "word_0": word_a,
                    "word_1": word_b,
                    "word_0_th": w0_th,
                    "word_1_th": w1_th,
                    "same_half": same_half,
                    "word_type": w_type,
                    "arm": f"source_plus_opposite_{band_suffix}",
                    "slot": slot,
                    "score_pp": float((pred_both == yt).mean() * 100.0),
                }
            )

    return fold_rows


def summarize_pair_results(folds_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate fold-level accuracies into per-pair and overall summaries."""
    key_cols = ["mode", "cohort", "word_0", "word_1", "arm"]
    meta_cols = ["word_0_th", "word_1_th", "same_half", "word_type"]

    summary = (
        folds_df.groupby(key_cols, sort=True)
        .agg(
            **{c: (c, "first") for c in meta_cols},
            mean_pp=("score_pp", "mean"),
            loso_sd_pp=("score_pp", "std"),
            min_loso_fold_pp=("score_pp", "min"),
            max_loso_fold_pp=("score_pp", "max"),
            folds_above_chance=("score_pp", lambda s: int((s > 50.0).sum())),
        )
        .reset_index()
    )
    wide = folds_df.pivot(index=key_cols, columns="slot", values="score_pp")
    wide.columns = [f"fold_{int(c)}_pp" for c in wide.columns]
    pair_summary = summary.merge(wide.reset_index(), on=key_cols, validate="one_to_one")

    agg_keys = ["arm", "same_half"]
    pair_group = pair_summary.groupby(agg_keys, dropna=False, sort=True)
    out = pair_group.agg(
        n_pairs=("mean_pp", "size"),
        avg_pair_mean_pp=("mean_pp", "mean"),
        sd_between_pair_means_pp=("mean_pp", "std"),
        min_pair_mean_pp=("mean_pp", "min"),
        max_pair_mean_pp=("mean_pp", "max"),
        avg_within_pair_loso_sd_pp=("loso_sd_pp", "mean"),
        min_within_pair_loso_sd_pp=("loso_sd_pp", "min"),
        max_within_pair_loso_sd_pp=("loso_sd_pp", "max"),
    ).reset_index()

    fold_group = folds_df.groupby(agg_keys, dropna=False, sort=True).agg(
        n_fold_scores=("score_pp", "size"),
        sd_all_fold_scores_pp=("score_pp", "std"),
        min_loso_fold_pp=("score_pp", "min"),
        max_loso_fold_pp=("score_pp", "max"),
    ).reset_index()

    aggregate_df = out.merge(fold_group, on=agg_keys, validate="one_to_one")
    return pair_summary, aggregate_df


def run_all_pair_loso(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    cohorts: Sequence[int] = (1, 2),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run strict LOSO evaluation and few-shot calibration across all specified cohorts."""
    all_folds: List[Dict[str, object]] = []
    for cohort in cohorts:
        words = list(range(1, 11)) if cohort == 1 else list(range(11, 21))
        features_by_mode: Dict[str, Dict[int, np.ndarray]] = {"mime": {}, "overt": {}}
        for mode in ("mime", "overt"):
            for w in words:
                features_by_mode[mode][w] = extract_cross_subject_bandpower(
                    word_id=w,
                    mode=mode,
                    dataset_dir=dataset_dir,
                )
            print(f"Loaded Cohort {cohort} ({mode}): words {words[0]}..{words[-1]}")

        for word_a, word_b in combinations(words, 2):
            for mode in ("mime", "overt"):
                all_folds.extend(
                    evaluate_pair_all_arms(
                        word_a=word_a,
                        word_b=word_b,
                        mode=mode,
                        cohort=cohort,
                        features_by_mode=features_by_mode,
                    )
                )

    folds_df = pd.DataFrame(all_folds)
    pair_summary_df, aggregate_df = summarize_pair_results(folds_df)
    return folds_df, pair_summary_df, aggregate_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Cross-Subject Strict LOSO and Few-Shot Calibration evaluation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "cross_subject",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, pair_summary_df, aggregate_df = run_all_pair_loso(dataset_dir=args.dataset_dir)

    pairs_path = args.output_dir / "all_pair_loso_summary.csv"
    agg_path = args.output_dir / "aggregate_all_cohorts.csv"
    pair_summary_df.to_csv(pairs_path, index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(agg_path, index=False, encoding="utf-8-sig")

    print("\n=== Cross-Subject Strict LOSO & Few-Shot Summary ===")
    print(aggregate_df.to_string(index=False))
    print(f"\nSaved per-pair summary to: {pairs_path}")
    print(f"Saved aggregate summary to: {agg_path}")


if __name__ == "__main__":
    main()
