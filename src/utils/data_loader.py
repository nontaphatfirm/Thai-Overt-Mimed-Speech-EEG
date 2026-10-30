"""Dataset loading, 5-step EEG preprocessing, and vocabulary metadata.

Preprocessing steps:
  1. Scale Restoration: eeg.astype(np.float32) * scale (microvolts)
  2. Spatial Re-referencing: Common Average Reference (CAR)
  3. Frequency Selection: Butterworth bandpass filter or Hann FFT log-bandpower
  4. Time Windowing: Active speech window 0.2-1.8 s (samples 102:922 at 512 Hz)
  5. Signal Normalization: Per-trial Z-score or train-only StandardScaler
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "Thai Overt-Mimed Speech EEG Dataset"

SAMPLING_RATE_HZ: int = 512
TRIAL_SAMPLES: int = 1024
TRIAL_PER_SUBJECT: int = 200
N_SUBJECTS_PER_COHORT: int = 10
ACTIVE_START_SEC: float = 0.2
ACTIVE_END_SEC: float = 1.8
ACTIVE_START_SAMPLE: int = round(ACTIVE_START_SEC * SAMPLING_RATE_HZ)  # 102
ACTIVE_END_SAMPLE: int = round(ACTIVE_END_SEC * SAMPLING_RATE_HZ)      # 922

# 20-word vocabulary metadata across Set 1 (words 1-10) and Set 2 (words 11-20).
# Dataset reference: https://data.mendeley.com/datasets/yzts9t69r2/1
# Recording-group split (first 5 vs. last 5 words per cohort) per personal communication with the research team.
VOCABULARY_METADATA: Dict[int, Dict[str, object]] = {
    1:  {"thai": "เปิดไฟ",  "english": "Turn on light",  "category": "Daily Life / Environment", "cohort": "Set 1 (word_01-10)", "group": 1},
    2:  {"thai": "ปิดไฟ",   "english": "Turn off light", "category": "Daily Life / Environment", "cohort": "Set 1 (word_01-10)", "group": 1},
    3:  {"thai": "อุจจาระ", "english": "Defecate",       "category": "Excretion / Physiology",   "cohort": "Set 1 (word_01-10)", "group": 1},
    4:  {"thai": "ปัสสาวะ", "english": "Urinate",        "category": "Excretion / Physiology",   "cohort": "Set 1 (word_01-10)", "group": 1},
    5:  {"thai": "หิวข้าว", "english": "Hungry",         "category": "Basic Needs (Food)",       "cohort": "Set 1 (word_01-10)", "group": 1},
    6:  {"thai": "หิวน้ำ",  "english": "Thirsty",        "category": "Basic Needs (Water)",      "cohort": "Set 1 (word_01-10)", "group": 2},
    7:  {"thai": "เศร้า",   "english": "Sad",            "category": "Emotion",                  "cohort": "Set 1 (word_01-10)", "group": 2},
    8:  {"thai": "กลัว",    "english": "Scared",         "category": "Emotion",                  "cohort": "Set 1 (word_01-10)", "group": 2},
    9:  {"thai": "โกรธ",    "english": "Angry",          "category": "Emotion",                  "cohort": "Set 1 (word_01-10)", "group": 2},
    10: {"thai": "ดีใจ",    "english": "Happy",          "category": "Emotion",                  "cohort": "Set 1 (word_01-10)", "group": 2},
    11: {"thai": "เช็ดตัว", "english": "Sponge bath",    "category": "Nursing / Daily Care",     "cohort": "Set 2 (word_11-20)", "group": 1},
    12: {"thai": "ใช่",     "english": "Yes",            "category": "Affirmation",              "cohort": "Set 2 (word_11-20)", "group": 1},
    13: {"thai": "ไม่",     "english": "No",             "category": "Negation",                 "cohort": "Set 2 (word_11-20)", "group": 1},
    14: {"thai": "หมอ",     "english": "Doctor",         "category": "Medical Request",          "cohort": "Set 2 (word_11-20)", "group": 1},
    15: {"thai": "ปวด",     "english": "Pain",           "category": "Symptom",                  "cohort": "Set 2 (word_11-20)", "group": 1},
    16: {"thai": "หัว",     "english": "Head",           "category": "Body Part",                "cohort": "Set 2 (word_11-20)", "group": 2},
    17: {"thai": "ขา",      "english": "Leg",            "category": "Body Part",                "cohort": "Set 2 (word_11-20)", "group": 2},
    18: {"thai": "แขน",     "english": "Arm",            "category": "Body Part",                "cohort": "Set 2 (word_11-20)", "group": 2},
    19: {"thai": "หลัง",    "english": "Back",           "category": "Body Part",                "cohort": "Set 2 (word_11-20)", "group": 2},
    20: {"thai": "คอ",      "english": "Neck",           "category": "Body Part",                "cohort": "Set 2 (word_11-20)", "group": 2},
}


def resolve_word_npz_path(
    mode: str,
    word_id: int,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> Path:
    """Return path to the `.npz` file for a given speech mode and word ID."""
    mode_clean = mode.strip().lower()
    if mode_clean not in {"overt", "mime"}:
        raise ValueError(f"Unsupported mode '{mode}'. Expected 'overt' or 'mime'.")
    path = dataset_dir / mode_clean / f"{mode_clean}_data_word_{word_id:02d}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Download the dataset from https://data.mendeley.com/datasets/yzts9t69r2/1 "
            f"and place 'overt/' and 'mime/' inside '{dataset_dir}'."
        )
    return path


def load_word_npz(
    mode: str,
    word_id: int,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> Tuple[np.ndarray, int, List[str]]:
    """Load a word `.npz` file and restore microvolt scale."""
    path = resolve_word_npz_path(mode=mode, word_id=word_id, dataset_dir=dataset_dir)
    with np.load(path, allow_pickle=True) as data:
        eeg_uv = data["eeg"].astype(np.float32) * data["scale"].astype(np.float32)
        fs = int(data["fs"])
        channels = [str(ch) for ch in data["channels"].tolist()]
    return eeg_uv, fs, channels


def recording_half(word_id: int) -> str:
    """Return 'first_half' (words 1-5, 11-15) or 'second_half' (words 6-10, 16-20)."""
    offset = (int(word_id) - 1) % 10
    return "first_half" if offset < 5 else "second_half"


def is_same_recording_half(word_a: int, word_b: int) -> bool:
    """Return True if both words belong to the same 5-word recording group."""
    return recording_half(word_a) == recording_half(word_b)


def preprocess_single_trial_waveform(
    eeg_uv: np.ndarray,
    fs: int = SAMPLING_RATE_HZ,
    l_freq: float = 4.0,
    h_freq: float = 30.0,
    start_sample: int = ACTIVE_START_SAMPLE,
    end_sample: int = ACTIVE_END_SAMPLE,
) -> np.ndarray:
    """Apply CAR, Butterworth bandpass filter, 0.2-1.8s windowing, and per-trial Z-score."""
    car = eeg_uv - eeg_uv.mean(axis=2, keepdims=True)
    sos = butter(4, [l_freq, h_freq], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, car, axis=1).astype(np.float32)
    windowed = filtered[:, start_sample:end_sample, :]
    mean = windowed.mean(axis=1, keepdims=True)
    std = windowed.std(axis=1, keepdims=True) + 1e-8
    return (windowed - mean) / std
