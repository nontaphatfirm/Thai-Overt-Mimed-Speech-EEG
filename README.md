# Thai Overt & Mimed Speech EEG Analysis

EEG signal processing and word-decoding pipeline for **Overt Speech** (vocalized) and **Mimed Speech** (silent articulation) in Thai. This repository covers signal preprocessing, exploratory within-subject decoding (`EEGNet`), artifact decomposition (`Picard ICA + ICLabel`), within-subject cross-mode transfer (`Overt ↔ Mimed`), strict cross-subject Leave-One-Subject-Slot-Out (`LOSO`) evaluation, dataset confound diagnostics, and few-shot calibration.

> Developed as part of the **[Brain Code Camp (BCC)](https://braincode101.github.io/)** program.  
> 📝 **Blog Post on Medium (Thai)**: [Thai Overt & Mimed Speech EEG](https://medium.com/@nontaphatfirm/thai-overt-mimed-speech-eeg-3207c127b06e)

---

## Dataset & Methodological Scope

- **Dataset Reference**: [Thai Overt-Mimed Speech EEG Dataset (Mendeley Data)](https://data.mendeley.com/datasets/yzts9t69r2/1)
- **Released Files Overview**: 20 participants (2 cohorts × 10 participants), 20 Thai words (10 words/cohort; split into two 5-word recording groups per cohort, per [Mendeley Data](https://data.mendeley.com/datasets/yzts9t69r2/1) and personal communication with the dataset research team), 2 modes (`overt`, `mime`), 32 channels (10-20 system), `512 Hz`, `2.0 s` per trial (`1,024 samples`), 200 contiguous trials/word/mode/participant (`80,000 trials` across 40 `.npz` files).

Download the dataset from the link above and place the `overt/` and `mime/` directories under `Thai Overt-Mimed Speech EEG Dataset/` in the project root (ignored by `.gitignore`):

```text
Thai Overt-Mimed Speech EEG Dataset/
├── overt/
│   └── overt_data_word_01.npz ... overt_data_word_20.npz
└── mime/
    └── mime_data_word_01.npz  ... mime_data_word_20.npz
```

| Cohort | Recording Group | Word IDs | Vocabulary (Thai / English) |
| :--- | :--- | :--- | :--- |
| **Set 1** (10 Slots) | **First Half (Group 1)** | `word_01` – `word_05` | 1. เปิดไฟ (Turn on light), 2. ปิดไฟ (Turn off light), 3. อุจจาระ (Defecate), 4. ปัสสาวะ (Urinate), 5. หิวข้าว (Hungry) |
| **Set 1** (10 Slots) | **Second Half (Group 2)** | `word_06` – `word_10` | 6. หิวน้ำ (Thirsty), 7. เศร้า (Sad), 8. กลัว (Scared), 9. โกรธ (Angry), 10. ดีใจ (Happy) |
| **Set 2** (10 Slots) | **First Half (Group 1)** | `word_11` – `word_15` | 11. เช็ดตัว (Sponge bath), 12. ใช่ (Yes), 13. ไม่ (No), 14. หมอ (Doctor), 15. ปวด (Pain) |
| **Set 2** (10 Slots) | **Second Half (Group 2)** | `word_16` – `word_20` | 16. หัว (Head), 17. ขา (Leg), 18. แขน (Arm), 19. หลัง (Back), 20. คอ (Neck) |

### Data & Analysis Notes
1. **Provisional Participant Slots**: The released `.npz` files do not contain explicit participant, session, or trial IDs. Following repository convention, consecutive 200-trial blocks (`0:200`, `200:400`, ..., `1800:2000`) within each 2,000-trial file are treated as 10 **provisional participant slots** per cohort, assuming consistent slot ordering across files within the same cohort.
2. **Vocabulary & Recording-Group Convention**: According to the public dataset description ([Mendeley Data](https://data.mendeley.com/datasets/yzts9t69r2/1)) and personal communication with the dataset research team, the 20 words are divided into two 10-word cohorts (`word_01..word_10` and `word_11..word_20`), with the first 5 words (`1–5`, `11–15`) and last 5 words (`6–10`, `16–20`) in each cohort belonging to separate recording groups (note that earlier exploratory within-subject notebooks labeled `word_05..word_08` as `ปวด, หัว, ขา, แขน`).
3. **Trial Quality**: Across all 80,000 trials in the released `.npz` files, 121 flat trials were identified (including 104 consecutive zeroed trials in `overt_data_word_18.npz`, slot 1). Excluding pairs involving `word_18` changes mean same-half `4–30 Hz` LOSO accuracy from `51.44%` to `51.51%`.

---

## Project Structure

```text
.
├── .gitignore                                             # Excludes dataset, backup/, and cache files
├── LICENSE                                                # Project license
├── README.md                                              # Project documentation
├── requirements.txt                                       # Python dependencies
├── Slide/                                                 # Presentation deck
│   └── Final Thai Overt-Mimed Speech EEG Present.pdf
├── notebooks/                                             # Interactive Jupyter notebook
│   └── Within_Subject_Frequency_and_ICA.ipynb             # Within-subject PSD, EEGNet, and Picard ICA + ICLabel
├── src/                                                   # Source code modules
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py                                 # NPZ loading, uV scaling, CAR, 0.2-1.8s windowing, vocabulary
│   │   └── plot_preprocessing.py                          # Raw vs. preprocessed EEG waveform comparison plot
│   ├── within_subject/
│   │   ├── __init__.py
│   │   ├── eegnet_frequency_ica.py                        # 4-class within-subject EEGNet & Picard ICA + ICLabel
│   │   ├── plot_frequency_accuracy.py                     # Within-subject EEGNet frequency-band accuracy plot & CSV
│   │   └── cross_mode_transfer.py                         # Within-subject Overt <-> Mimed cross-mode decoding
│   └── cross_subject/
│       ├── __init__.py
│       ├── loso_evaluation.py                             # Strict 10-fold LOSO & few-shot calibration across 180 pairs
│       └── dataset_diagnostics.py                         # Recording-group control & slot specificity diagnostics
└── results/                                               # Reproducible experimental outputs
    ├── within_subject/
    │   ├── eeg_preprocessing_before_after.png
    │   ├── within_subject_eegnet_frequency_accuracy.png
    │   ├── within_subject_eegnet_frequency_scores.csv
    │   ├── cross_mode_4class_folds.csv
    │   └── cross_mode_4class_summary.json
    └── cross_subject/
        ├── aggregate_all_cohorts.csv
        ├── all_pair_loso_summary.csv
        ├── all_pair_diagnostics.csv
        ├── recording_half_control.json
        └── subject_signature.json
```

---

## Installation & Usage

```bash
pip install -r requirements.txt
```

| Experiment / Task | Command | Output Files |
| :--- | :--- | :--- |
| **5-Step EEG Preprocessing Plot** | `python -m src.utils.plot_preprocessing` | `results/within_subject/eeg_preprocessing_before_after.png` |
| **Within-Subject EEGNet & ICA** | `python -m src.within_subject.eegnet_frequency_ica --run-ica` | Console report (or run `notebooks/Within_Subject_Frequency_and_ICA.ipynb`) |
| **Within-Subject Frequency Plot & CSV** | `python -m src.within_subject.plot_frequency_accuracy` | `results/within_subject/within_subject_eegnet_frequency_scores.csv`<br>`results/within_subject/within_subject_eegnet_frequency_accuracy.png` |
| **Within-Subject Cross-Mode Transfer** | `python -m src.within_subject.cross_mode_transfer` | `results/within_subject/cross_mode_4class_summary.json`<br>`results/within_subject/cross_mode_4class_folds.csv` |
| **Cross-Subject Strict LOSO & Few-Shot** | `python -m src.cross_subject.loso_evaluation` | `results/cross_subject/aggregate_all_cohorts.csv`<br>`results/cross_subject/all_pair_loso_summary.csv` |
| **Cross-Subject Dataset Diagnostics** | `python -m src.cross_subject.dataset_diagnostics` | `results/cross_subject/recording_half_control.json`<br>`results/cross_subject/subject_signature.json`<br>`results/cross_subject/all_pair_diagnostics.csv` |

---

## Experimental Results & Scientific Interpretation

### 1. EEG Preprocessing Pipeline
1. **Scale Restoration**: Multiply raw ADC counts by per-trial `scale` to restore microvolts ($\mu V$).
2. **Spatial Re-referencing**: Common Average Reference (`CAR`) across 32 channels.
3. **Frequency Selection**: Butterworth bandpass filtering (time-domain `EEGNet`) or Hann-windowed FFT log-bandpower (linear/LDA models).
4. **Time Windowing**: Crop active speech interval `0.2–1.8 s` (`samples 102:922`).
5. **Signal Normalization**: Per-trial Z-score (within-subject `EEGNet`) or train-only `StandardScaler` (cross-mode and strict LOSO).

### 2. Model Selection & Hyperparameter Strategy
- **Within-Subject Decoding (`EEGNet`)**:
  - **Architecture Choice**: `EEGNet` (Lawhern et al., 2018), a compact CNN specialized for raw EEG time-series using depthwise and separable convolutions to learn spatial and temporal filters with minimal parameters.
  - **Hyperparameters**: Temporal filters $F_1 = 8$, depth multiplier $D = 2$ ($F_2 = 16$), temporal kernel length $= 256$ (500 ms at 512 Hz), dropout $= 0.25$.
  - **Training Setup**: Adam optimizer ($\text{lr} = 10^{-3}$), cross-entropy loss, batch size $= 32$, 100 epochs, per-trial Z-score normalization.
- **Cross-Mode & Cross-Subject LOSO (Linear Baselines)**:
  - **Model Choice**: Linear models on Hann-windowed FFT log-bandpower features were selected over deep networks to avoid overfitting to subject-specific signatures, ensure reproducible evaluation across 180 word pairs $\times$ 10 folds, and provide transparent baselines.
  - **Hyperparameters & Regularization**:
    - **Logistic Regression ($C=0.1$)**: L2-regularized logistic regression with tuned $C=0.1$ to penalize high weights across correlated multi-channel bandpower features.
    - **Shrinkage LDA (`shrinkage="auto"`)**: Linear Discriminant Analysis with Ledoit-Wolf automatic covariance shrinkage to stabilize ill-conditioned spatial covariance matrices across 32 EEG channels.

### 3. Exploratory Within-Subject Evaluation (`words = [5, 6, 7, 8]`)
- **Frequency-Band `EEGNet` (4 Classes, Chance = 25%, Chronological 64% Train / 16% Val / 20% Test per Word Block)**:

| Frequency Band | Overt (Mean ± SD) | Mimed (Mean ± SD) |
| :--- | :---: | :---: |
| **Delta (`1–4 Hz`)** | `89.6 ± 5.5%` | `88.8 ± 9.4%` |
| **Theta (`4–8 Hz`)** | `91.1 ± 8.2%` | `93.6 ± 5.3%` |
| **Alpha (`8–12 Hz`)** | `88.8 ± 10.2%` | `93.4 ± 7.9%` |
| **Beta (`12–30 Hz`)** | `89.8 ± 9.7%` | `96.4 ± 5.6%` |
| **Gamma (`30–80 Hz`)** | `93.1 ± 8.5%` | `97.9 ± 3.8%` |
| **Broadband (`0.5–45 Hz`)** | `88.6 ± 5.8%` | `90.2 ± 8.6%` |

- **Artifact Decomposition (`Picard ICA + ICLabel`)**: Fitting 25-component Picard ICA per participant slot and rejecting ocular, cardiac, line/channel noise, and muscle (`>30%`) or low-brain (`<=40%`) components leaves within-subject accuracy high (`~87–90%`).
- **Within-Subject Cross-Mode Transfer (4 Classes `words = [5, 6, 7, 8]`, Chance = 25%)**:  
  Evaluated using `CAR + 0.2–1.8 s Hann FFT log-bandpower + train-only StandardScaler + LogisticRegression(C=0.1)` with a chronological `160 train / 40 test` split per word and slot:

| Band | Overt → Overt | Overt → Mimed | Mimed → Mimed | Mimed → Overt |
| :--- | :---: | :---: | :---: | :---: |
| **`4–30 Hz`** | `93.12 ± 7.24%` | `58.38 ± 12.00%` | `96.31 ± 6.34%` | `73.38 ± 17.05%` |
| **`30–45 Hz`** | `92.63 ± 7.84%` | `55.75 ± 13.49%` | `96.38 ± 7.77%` | `73.56 ± 15.66%` |

> **Interpretation Limit (Within-Subject)**: Because trials for each word are stored in a single contiguous 200-trial block (and `word_05` belongs to the first 5-word recording group while `words 06–08` belong to the second; personal communication with the dataset research team), high within-subject accuracy and post-ICA retention reflect **within-block and recording-group discriminability** rather than isolated semantic neural representations or guaranteed complete EMG removal. Similarly, the descriptive asymmetry between `Mimed → Overt` and `Overt → Mimed` cannot be attributed to a single neural mechanism without unconfounded cross-session data.

### 4. Cross-Subject Strict LOSO, Confound Diagnostics & Few-Shot Calibration
- **Zero-Calibration Strict LOSO (80 Same-Half Word Pairs, Chance = 50%)**:  
  Evaluated across all 80 word pairs within the same 5-word recording group (`same_half=True`), fitting all scalers and classifiers strictly on the 9 training slots:

| Pipeline (Zero-Calibration) | Mean ± SD Across 80 Pairs | Pair Mean Range | Avg Within-Pair Fold SD | Individual Fold Range |
| :--- | :---: | :---: | :---: | :---: |
| **`4–30 Hz + Logistic Regression`** | **`51.44 ± 4.57%`** | `38.60–62.72%` | `11.40%` | `0.75–99.00%` |
| **`30–45 Hz + Spatial Shrinkage LDA`** | **`52.50 ± 4.89%`** | `42.82–67.72%` | `13.19%` | `0.00–100.00%` |

> **Interpretation Limit (Zero-Calibration LOSO)**: Under the tested bandpower pipelines, zero-calibration cross-subject word decoding across same-group word pairs remains near the 50% chance level with high inter-slot variance. This shows that the tested pipelines do not generalize reliably to unseen participant slots on this dataset; it does not prove that no speech-related brain signal exists or that no future architecture could succeed.

- **Empirical Dataset Diagnostics**:
  1. **Same-Half vs. Cross-Half Pair Comparison**: With `4–30 Hz + Logistic Regression`, the 80 same-half pairs average **`51.44 ± 4.57%`** (avg within-pair fold SD `11.40%`), whereas the 100 cross-half pairs (`same_half=False`, comparing first-five vs. last-five words) average **`73.07 ± 8.31%`** (avg within-pair fold SD `26.05%`). Elevated cross-half scores therefore reflect **recording-group differences** rather than word-level generalization.
  2. **Recording-Group Control (`words 11–20`)**: Training a `4–30 Hz` classifier on 8 words from 9 slots to predict whether a trial comes from the first 5 words (`11–15`) or last 5 words (`16–20`) on **2 held-out words of an unseen 10th slot** yields **`82.91 ± 17.79%`** (Mimed) and **`74.41 ± 21.19%`** (Overt), confirming a transferable recording-group signal across words and slots.
  3. **Retrospective Slot Fingerprint & Contrast Cosine Diagnostic (`80 same-half pairs, 4–30 Hz`)**: Nearest-centroid 10-way slot identification trained on one word block and tested on another reaches **`82.6%`** (chance `10%`, reflecting participant or recording-state differences). Meanwhile, word contrast vectors (Word B minus Word A) are consistent between the first 100 and last 100 trials within the same slot (`mean cosine = 0.702`) but near-orthogonal across different slots (`mean pairwise cosine = 0.010`, positive in `51.2%` of slot pairs). *Note: Contrast cosine is a retrospective diagnostic computed using both words' labels, not a deployable decoding model.*

- **Few-Shot Calibration (`20 labeled trials/word`, 80 Same-Half Word Pairs)**:

| Calibration Arm | Training Data | Test Set | Mean ± SD Across 80 Pairs | Avg Within-Pair Fold SD | Min Fold |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **`zero_low_logistic`** | 9 source slots only (`4–30 Hz`) | All 200 trials/word of target slot | `51.44 ± 4.57%` | `11.40%` | `0.75%` |
| **`source_plus_opposite_low`** | 9 source slots (target mode) + 20 evenly spaced trials/word from target slot's opposite mode (`4–30 Hz`, weighted) | All 200 trials/word of target slot | **`77.80 ± 10.87%`** | `15.45%` | `7.75%` |
| **`opposite_target_only_low`** | 20 evenly spaced trials/word (`0, 10, ..., 190`) from target slot's opposite mode only (`4–30 Hz`, trained from scratch) | All 200 trials/word of target slot | **`77.39 ± 10.95%`** | `16.06%` | `12.00%` |
| **`target_first20`** | First 20 trials/word from target slot's same block only (`30–45 Hz` Mimed, `4–30 Hz` Overt, trained from scratch) | Remaining 180 trials/word of the **same block** | **`86.13 ± 6.83%`** | `12.53%` | `34.44%` |

> **Interpretation Limit (Few-Shot Calibration)**:
> - `target_first20` (`86.13%`) trains from scratch on the target slot's first 20 trials/word without using the 9 source slots; because calibration and testing come from the same contiguous 200-trial block, within-block continuity contributes to this score.
> - `opposite_target_only_low` (`77.39%`) and `source_plus_opposite_low` (`77.80%`) evaluate across separate mode blocks, with calibration trials sampled uniformly across the opposite-mode block. Adding the 9 source slots changes mean accuracy by only **`+0.41` percentage points** (`77.39% → 77.80%`), providing no consistent evidence that cross-subject source data improves performance over target-slot calibration alone.

---

## Author & Resources

- **Author**: **[Nontapat Auetrongjit (Firm)](https://github.com/nontaphatfirm)**  
  Computer Engineering and Digital Technology (CEDT), Chulalongkorn University  
  Participant, **[Brain Code Camp (BCC)](https://braincode101.github.io/)**
- **Blog Post on Medium (Thai)**: [Thai Overt & Mimed Speech EEG](https://medium.com/@nontaphatfirm/thai-overt-mimed-speech-eeg-3207c127b06e)
- **Presentation Deck**: [`Slide/Final Thai Overt-Mimed Speech EEG Present.pdf`](Slide/Final%20Thai%20Overt-Mimed%20Speech%20EEG%20Present.pdf)

---
