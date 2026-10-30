"""Within-subject 4-class EEGNet frequency-band evaluation and Picard ICA + ICLabel pipeline.

Protocol & Methodological Scope:
  - 4 Target Classes: Words [5, 6, 7, 8] (Chance = 25.0%) across 10 provisional 200-row participant slots.
  - Chronological split per word block: 64% Train, 16% Validation, 20% Test.
  - Evaluates 6 frequency bands: Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-12 Hz),
    Beta (12-30 Hz), Gamma (30-80 Hz), and Broadband (0.5-45 Hz).
  - Exploratory ICA branch fits 25-component Picard Extended Infomax ICA + MNE-ICLabel
    per subject block, rejecting eye blink, heart beat, line/channel noise, muscle (>30%),
    or low-brain (<=40%) components before re-evaluating EEGNet in the 0.5-30 Hz band.
  - Interpretation limit: Because trials for each word are stored in a contiguous 200-trial
    block (and word 5 vs. words 6-8 belong to different 5-word recording groups; personal
    communication with the dataset research team), high within-subject accuracy reflects
    within-block discriminability and does not prove isolated semantic neural decoding or
    complete EMG removal.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mne.preprocessing import ICA
from mne_icalabel import label_components
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from src.utils.data_loader import (
    DEFAULT_DATASET_DIR,
    PROJECT_ROOT,
    resolve_word_npz_path,
)

DEFAULT_WORDS: List[int] = [5, 6, 7, 8]
DEFAULT_SUBJECTS: List[int] = list(range(10))

FREQUENCY_BANDS: Dict[str, Tuple[float, float]] = {
    "Delta (1-4 Hz)": (1.0, 4.0),
    "Theta (4-8 Hz)": (4.0, 8.0),
    "Alpha (8-12 Hz)": (8.0, 12.0),
    "Beta (12-30 Hz)": (12.0, 30.0),
    "Gamma (30-80 Hz)": (30.0, 80.0),
    "Broadband (0.5-45 Hz)": (0.5, 45.0),
}


class EEGNet(nn.Module):
    """Compact Convolutional Network for EEG-based BCIs (Lawhern et al., 2018)."""

    def __init__(
        self,
        n_classes: int = 4,
        chans: int = 32,
        samples: int = 1024,
        f1: int = 8,
        d: int = 2,
        kern_length: int = 256,
        dropout_rate: float = 0.25,
    ) -> None:
        super().__init__()
        f2 = f1 * d

        self.conv1 = nn.Conv2d(1, f1, (1, kern_length), padding=(0, kern_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.depthwise = nn.Conv2d(f1, f2, (chans, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f2)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        self.separable = nn.Conv2d(f2, f2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        out_samples = samples // 32
        self.fc = nn.Linear(f2 * out_samples, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.drop1(self.pool1(self.act1(self.bn2(self.depthwise(self.bn1(self.conv1(x)))))))
        x = self.drop2(self.pool2(self.act2(self.bn3(self.separable(x)))))
        x = x.flatten(start_dim=1)
        return self.fc(x)


def load_word_epochs(
    word_id: int,
    modality: str,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    apply_baseline: bool = True,
) -> Tuple[mne.EpochsArray, int]:
    """Load a word `.npz` file into an `mne.EpochsArray` with CAR re-referencing."""
    path = resolve_word_npz_path(mode=modality, word_id=word_id, dataset_dir=dataset_dir)
    with np.load(path, allow_pickle=True) as data:
        ch_names = [str(c) for c in data["channels"]]
        sfreq = int(data["fs"])
        eeg_volts = data["eeg"].astype(np.float32) * data["scale"].astype(np.float32) * 1e-6

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    epochs = mne.EpochsArray(
        data=np.transpose(eeg_volts, (0, 2, 1)),
        info=info,
        tmin=0.0,
        verbose=False,
    )
    epochs.set_montage("standard_1020", on_missing="ignore")
    epochs.set_eeg_reference("average", verbose=False)
    if apply_baseline:
        epochs.apply_baseline((0.0, 0.2), verbose=False)
    return epochs, sfreq


def drop_bad_trials(trials: np.ndarray) -> np.ndarray:
    """Reject flat or extreme peak-to-peak outlier trials."""
    ptp = np.ptp(trials, axis=-1).max(axis=1)
    mask = (ptp > 1e-6) & (ptp < np.median(ptp) + 2.5 * np.std(ptp))
    return trials[mask]


def build_within_subject_splits(
    word_epochs: Dict[int, mne.EpochsArray],
    words: Sequence[int],
    subject_ids: Sequence[int],
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Create chronological 64% Train / 16% Val / 20% Test splits per subject and word."""
    subject_data = {}
    for s in subject_ids:
        s_tr, s_v, s_te = [], [], []
        for w in words:
            raw_trials = word_epochs[w].get_data(copy=True)[s * 200 : (s + 1) * 200]
            clean_trials = drop_bad_trials(raw_trials)
            n_tr = int(len(clean_trials) * 0.64)
            n_v = int(len(clean_trials) * 0.16)
            s_tr.append(clean_trials[:n_tr])
            s_v.append(clean_trials[n_tr : n_tr + n_v])
            s_te.append(clean_trials[n_tr + n_v :])

        n_tr_min = min(len(x) for x in s_tr)
        n_v_min = min(len(x) for x in s_v)
        n_te_min = min(len(x) for x in s_te)

        x_tr = np.concatenate([x[:n_tr_min] for x in s_tr], axis=0)
        y_tr = np.concatenate([np.full(n_tr_min, i) for i in range(len(words))])
        x_v = np.concatenate([x[:n_v_min] for x in s_v], axis=0)
        y_v = np.concatenate([np.full(n_v_min, i) for i in range(len(words))])
        x_te = np.concatenate([x[:n_te_min] for x in s_te], axis=0)
        y_te = np.concatenate([np.full(n_te_min, i) for i in range(len(words))])

        x_tr = (x_tr - x_tr.mean(axis=-1, keepdims=True)) / (x_tr.std(axis=-1, keepdims=True) + 1e-8)
        x_v = (x_v - x_v.mean(axis=-1, keepdims=True)) / (x_v.std(axis=-1, keepdims=True) + 1e-8)
        x_te = (x_te - x_te.mean(axis=-1, keepdims=True)) / (x_te.std(axis=-1, keepdims=True) + 1e-8)

        subject_data[s] = (x_tr, y_tr, x_v, y_v, x_te, y_te)
    return subject_data


def train_and_evaluate_eegnet(
    subject_data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    n_classes: int,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[int, float]:
    """Train an independent EEGNet model for each subject and return test accuracies (%)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    accuracies: Dict[int, float] = {}
    for s, (x_tr, y_tr, _, _, x_te, y_te) in subject_data.items():
        train_loader = DataLoader(
            TensorDataset(torch.tensor(x_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(x_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long)),
            batch_size=batch_size,
            shuffle=False,
        )

        model = EEGNet(n_classes=n_classes, chans=x_tr.shape[1], samples=x_tr.shape[2]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                preds.extend(model(x_batch.to(device)).argmax(dim=1).cpu().numpy())
                trues.extend(y_batch.numpy())

        accuracies[s] = float(accuracy_score(trues, preds) * 100.0)
    return accuracies


def evaluate_frequency_band(
    modality: str,
    l_freq: float,
    h_freq: float,
    words: Sequence[int] = DEFAULT_WORDS,
    subject_ids: Sequence[int] = DEFAULT_SUBJECTS,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    epochs: int = 100,
    batch_size: int = 32,
) -> Dict[int, float]:
    """Filter epochs to `[l_freq, h_freq]` Hz (plus 50 Hz notch if applicable) and evaluate EEGNet."""
    filtered_epochs: Dict[int, mne.EpochsArray] = {}
    for w in words:
        ep, _ = load_word_epochs(w, modality=modality, dataset_dir=dataset_dir, apply_baseline=True)
        ep.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            method="iir",
            iir_params=dict(order=4, ftype="butter"),
            verbose=False,
        )
        if l_freq < 50.0 < h_freq:
            ep.filter(
                l_freq=51.0,
                h_freq=49.0,
                method="iir",
                iir_params=dict(order=4, ftype="butter"),
                verbose=False,
            )
        filtered_epochs[w] = ep

    splits = build_within_subject_splits(filtered_epochs, words=words, subject_ids=subject_ids)
    return train_and_evaluate_eegnet(splits, n_classes=len(words), epochs=epochs, batch_size=batch_size)


def run_ica_iclabel_evaluation(
    modality: str = "overt",
    words: Sequence[int] = DEFAULT_WORDS,
    subject_ids: Sequence[int] = DEFAULT_SUBJECTS,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    epochs: int = 100,
    batch_size: int = 32,
) -> Tuple[Dict[int, float], pd.DataFrame]:
    """Fit 25-component Picard ICA + ICLabel per subject, remove artifacts, and evaluate EEGNet."""
    raw_word_epochs = {
        w: load_word_epochs(w, modality=modality, dataset_dir=dataset_dir, apply_baseline=False)[0]
        for w in words
    }

    subject_icas: Dict[int, ICA] = {}
    ic_summary_rows = []

    for s in subject_ids:
        sub_eps = mne.concatenate_epochs(
            [raw_word_epochs[w][s * 200 : (s + 1) * 200] for w in words],
            verbose=False,
        )
        sub_for_ica = sub_eps.copy().filter(
            1.0,
            100.0,
            method="iir",
            iir_params=dict(order=4, ftype="butter"),
            verbose=False,
        )
        if sub_for_ica.info["sfreq"] > 250:
            sub_for_ica.resample(250.0, verbose=False)

        ica = ICA(
            n_components=25,
            method="picard",
            fit_params=dict(ortho=False, extended=True),
            random_state=42,
        )
        ica.fit(sub_for_ica, decim=2, verbose=False)

        ic_labels = label_components(sub_for_ica, ica, method="iclabel")
        labels = ic_labels["labels"]
        scores = ica.labels_scores_

        exclude_idx = [
            i
            for i, (lbl, sc) in enumerate(zip(labels, scores))
            if (lbl in ["eye blink", "heart beat", "line noise", "channel noise"])
            or (sc[1] > 0.30)
            or (sc[0] <= 0.40)
        ]
        if len(exclude_idx) == len(labels):
            exclude_idx.remove(int(np.argmax(scores[:, 0])))

        ica.exclude = exclude_idx
        subject_icas[s] = ica
        ic_summary_rows.append(
            {
                "subject": s,
                "excluded_ics": len(exclude_idx),
                "retained_ics": len(labels) - len(exclude_idx),
                "mean_brain_prob_pct": float(scores[:, 0].mean() * 100.0),
                "mean_muscle_prob_pct": float(scores[:, 1].mean() * 100.0),
            }
        )

    cleaned_splits = {}
    for s in subject_ids:
        s_tr, s_v, s_te = [], [], []
        ica_s = subject_icas[s]
        for w in words:
            sub_raw = raw_word_epochs[w][s * 200 : (s + 1) * 200].copy()
            ep_clean = ica_s.apply(sub_raw, verbose=False)
            ep_clean.apply_baseline((0.0, 0.2), verbose=False)
            ep_clean.filter(
                0.5,
                30.0,
                method="iir",
                iir_params=dict(order=4, ftype="butter"),
                verbose=False,
            )
            s_x = drop_bad_trials(ep_clean.get_data(copy=True))
            n_tr = int(len(s_x) * 0.64)
            n_v = int(len(s_x) * 0.16)
            s_tr.append(s_x[:n_tr])
            s_v.append(s_x[n_tr : n_tr + n_v])
            s_te.append(s_x[n_tr + n_v :])

        n_tr_min = min(len(x) for x in s_tr)
        n_v_min = min(len(x) for x in s_v)
        n_te_min = min(len(x) for x in s_te)

        x_tr = np.concatenate([x[:n_tr_min] for x in s_tr], axis=0)
        y_tr = np.concatenate([np.full(n_tr_min, i) for i in range(len(words))])
        x_v = np.concatenate([x[:n_v_min] for x in s_v], axis=0)
        y_v = np.concatenate([np.full(n_v_min, i) for i in range(len(words))])
        x_te = np.concatenate([x[:n_te_min] for x in s_te], axis=0)
        y_te = np.concatenate([np.full(n_te_min, i) for i in range(len(words))])

        x_tr = (x_tr - x_tr.mean(axis=-1, keepdims=True)) / (x_tr.std(axis=-1, keepdims=True) + 1e-8)
        x_v = (x_v - x_v.mean(axis=-1, keepdims=True)) / (x_v.std(axis=-1, keepdims=True) + 1e-8)
        x_te = (x_te - x_te.mean(axis=-1, keepdims=True)) / (x_te.std(axis=-1, keepdims=True) + 1e-8)

        cleaned_splits[s] = (x_tr, y_tr, x_v, y_v, x_te, y_te)

    clean_accs = train_and_evaluate_eegnet(
        cleaned_splits,
        n_classes=len(words),
        epochs=epochs,
        batch_size=batch_size,
    )
    return clean_accs, pd.DataFrame(ic_summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Within-Subject 4-class EEGNet across frequency bands and Picard ICA cleaning."
    )
    parser.add_argument("--modality", choices=["overt", "mime", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--run-ica", action="store_true", help="Run 25-component Picard ICA + ICLabel evaluation.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()

    modalities = ["overt", "mime"] if args.modality == "both" else [args.modality]
    for mod in modalities:
        print(f"\n==================== Modality: {mod.upper()} ====================")
        for band_name, (f_low, f_high) in FREQUENCY_BANDS.items():
            accs = evaluate_frequency_band(
                modality=mod,
                l_freq=f_low,
                h_freq=f_high,
                dataset_dir=args.dataset_dir,
                epochs=args.epochs,
            )
            vals = list(accs.values())
            print(f"{band_name:22s} -> Mean: {np.mean(vals):6.2f}% ± {np.std(vals, ddof=1):5.2f}%")

        if args.run_ica:
            print(f"\n--- Picard ICA + ICLabel ({mod.upper()}) ---")
            clean_accs, ic_df = run_ica_iclabel_evaluation(
                modality=mod,
                dataset_dir=args.dataset_dir,
                epochs=args.epochs,
            )
            vals = list(clean_accs.values())
            print(ic_df.to_string(index=False))
            print(f"Post-ICA Cleaned EEGNet -> Mean: {np.mean(vals):6.2f}% ± {np.std(vals, ddof=1):5.2f}%")


if __name__ == "__main__":
    main()
