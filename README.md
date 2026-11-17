# Pneumonia Detection from Chest X-Rays — Baseline CNN vs. Transfer Learning

A PyTorch project that trains a **from-scratch convolutional neural network** and two **pretrained
transfer-learning models** on the [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
dataset, classifying chest X-rays as `NORMAL` or `PNEUMONIA`, and compares how their results differ.

Coursework for **Neural Networks and Deep Learning**.

---

## Assignment brief

> Pneumonia Detection from chest xrays... implement using normal cnn and transfer learning model...
> dataset from kaggle.. manually or using api key.. upload in github as a proof project

How each requirement is met:

| Requirement | Where |
|---|---|
| Normal CNN (from scratch) | `BaselineCNN` — Section 4 of the notebook |
| Transfer learning model | `ConvNeXtTinyCXR` and `EfficientNetB0CXR` — Sections 5 and 6 |
| Kaggle dataset, manual download | Section 1 — no API key needed; the notebook auto-locates the archive |
| Uploaded to GitHub as proof | This repository |

---

## Models evaluated

Five arms, all trained and scored through the same pipeline:

| # | Arm | Pretrained weights | Trainable parameters |
|---|---|---|---|
| 1 | Baseline CNN | none — trained from scratch | 1,206,370 |
| 2 | ConvNeXt-Tiny + custom head | ImageNet (stage 4 and head trained) | 15,570,818 |
| 3 | EfficientNet-B0 + custom head | ImageNet (last blocks and head trained) | 1,293,618 |
| 4 | Ensemble of arms 2 and 3 | soft vote, equal weights | 16,864,436 |
| 5 | Ensemble of arms 1, 2 and 3 | soft vote, equal weights | 18,070,806 |

**Baseline CNN (from scratch).** Four blocks, each holding two 3×3 convolutions with BatchNorm and ReLU,
followed by 2×2 max-pooling. Channels go 3 → 32 → 64 → 128 → 256. Global average pooling, dropout, then
`Linear(256, 128)` → `Linear(128, 2)`. Capacity is deliberately close to EfficientNet-B0's trainable count,
so the comparison isolates *whether the features are pretrained* rather than model size.

**ConvNeXt-Tiny.** ImageNet-pretrained backbone; everything before `features[6]` is frozen and pinned to
evaluation mode, so the last downsample, stage 4 and the new head are the only trainable parts. Head:
LayerNorm → Dropout → `Linear(768, 128)` → GELU → Dropout → `Linear(128, 2)`.

**EfficientNet-B0.** ImageNet-pretrained; blocks before `features[7]` are frozen and pinned to evaluation
mode, which keeps the frozen BatchNorm statistics from drifting. Head: Dropout → `Linear(1280, 128)` →
ReLU → Dropout → `Linear(128, 2)`.

---

## Dataset

**[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)**
— Kermany et al. Paediatric chest X-rays from a single hospital, labelled `NORMAL` / `PNEUMONIA`.

| Split | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| Official `train/` | 1,341 | 3,875 | 5,216 |
| Official `test/` | 234 | 390 | 624 |

**The dataset is not included in this repository.** Download it manually:

1. Open the Kaggle page linked above and sign in (a free account is enough).
2. Click **Download** (top-right) — you get `archive.zip`, about 1.2 GB. It may also be named
   `chest-xray-pneumonia.zip`.
3. Place the zip **in the same folder as the notebook**.
4. Run Section 1 of the notebook. It extracts to `data/` and finds `train/` and `test/` whether the
   archive stores them at the top level or one level deeper.

Mirror if the Kaggle page is unavailable: https://data.mendeley.com/datasets/rscbjbr9sj/3

The provided `val/` folder is **not used** — it contains only 16 images. Validation is re-derived from the
training data instead (see below).

---

## Methodology

Details that change the numbers, and the reasoning behind each:

- **Patient-level split.** Several images exist per patient, and the filename encodes the patient id
  (`person137_bacteria_650.jpeg`, `NORMAL2-IM-0382-0001.jpeg`). The 5,216 training images come from 2,846
  patients, and the split uses `StratifiedGroupKFold`, so no patient appears on both sides — validation
  holds 746 images from 406 patients, with 0 patients shared. An image-level split leaks the same child
  into train and validation and inflates validation accuracy. The notebook asserts that the two patient
  sets are disjoint and that the validation class balance is preserved (192 NORMAL / 554 PNEUMONIA).
- **Fixed input pipeline.** 224×224, converted to 3 channels, normalised with ImageNet mean/std — identical
  for all five arms, so differences come from the architecture and not from preprocessing.
- **Augmentation.** Random horizontal flip (p=0.3) and ±10° rotation, training split only. Validation and
  test use the plain resize/normalise transform.
- **Class imbalance.** The training data is 74% PNEUMONIA, so a `WeightedRandomSampler` draws balanced
  batches. The loss stays unweighted, and every reported metric is computed on the natural test
  distribution.
- **Threshold from validation, not 0.5.** Because the sampler shifts the output distribution, a hardcoded
  0.5 cut makes every model flag far too many images as pneumonia. Each arm's threshold is instead chosen on
  the validation split by maximising Youden's J (sensitivity + specificity − 1) and then applied unchanged
  to the test set. The test data plays no part in choosing it. ROC-AUC is reported as the threshold-free
  measure of ranking quality.
- **Optimisation.** Adam with `ReduceLROnPlateau` (factor 0.5, patience 2), cross-entropy loss, batch size
  32. Learning rate 2e-4 for the transfer models over 10 epochs; 1e-3 for the baseline over 30 epochs,
  since random initialisation has no pretrained features to fine-tune. The best checkpoint by validation
  loss is restored before evaluation.
- **Reproducibility.** `SEED = 42` for Python, NumPy and PyTorch; deterministic cuDNN; a seeded generator
  for the sampler; and a `seed_worker` function so the augmentation workers do not replay identical random
  transforms.

---

## Results

One full run on an NVIDIA RTX 4050 Laptop GPU, 624-image test set, each arm scored at the threshold
selected on its validation split. Training took about 31 minutes in total (37–39 s per epoch).

| Arm | Accuracy | Precision | Recall (PNEUMONIA) | Specificity | F1 | ROC-AUC | Threshold | Trainable params |
|---|---|---|---|---|---|---|---|---|
| **Baseline CNN** | **0.8349** | **0.7959** | 0.9897 | **0.5769** | **0.8823** | 0.9389 | 0.764 | 1,206,370 |
| ConvNeXt-Tiny | 0.8237 | 0.7834 | 0.9923 | 0.5427 | 0.8756 | 0.9686 | 0.492 | 15,570,818 |
| EfficientNet-B0 | 0.8013 | 0.7598 | 0.9974 | 0.4744 | 0.8625 | 0.9572 | 0.115 | 1,293,618 |
| Ensemble (ConvNeXt + EfficientNet) | 0.8189 | 0.7764 | 0.9974 | 0.5214 | 0.8732 | **0.9701** | 0.361 | 16,864,436 |
| Ensemble (all three) | 0.8285 | 0.7859 | 0.9974 | 0.5470 | 0.8791 | 0.9695 | 0.427 | 18,070,806 |

Paired exact McNemar test against the from-scratch baseline (624 identical test images):

| Arm | Right only by this arm | Right only by Baseline CNN | p-value |
|---|---|---|---|
| ConvNeXt-Tiny | 19 | 26 | 0.371 |
| EfficientNet-B0 | 14 | 35 | **0.004** |
| Ensemble (ConvNeXt + EfficientNet) | 15 | 25 | 0.154 |

Prediction disagreement between arms, which is what determines whether averaging can help:

| Pair | Disagreement | Cohen's kappa |
|---|---|---|
| Baseline CNN vs ConvNeXt-Tiny | 7.21% | 0.787 |
| Baseline CNN vs EfficientNet-B0 | 7.85% | 0.756 |
| ConvNeXt-Tiny vs EfficientNet-B0 | 6.73% | 0.785 |

### What the numbers say

**The accuracy ranking and the ROC-AUC ranking disagree, and that is the main finding.** The from-scratch
baseline has the highest test accuracy (0.835) and the highest specificity (0.577), but the *lowest* ROC-AUC
(0.939). The two pretrained models rank images better — ConvNeXt at 0.969 and the ConvNeXt+EfficientNet
ensemble at 0.970 — yet convert that into slightly lower accuracy at their tuned thresholds. The baseline
does not lose to transfer learning here; it wins on the metric most people quote and loses on the metric
that measures ranking quality.

**At this sample size the gap is mostly noise.** ConvNeXt versus the baseline is 19 images right for
ConvNeXt against 26 right for the baseline, p = 0.371 — statistically indistinguishable. The one real
separation is EfficientNet-B0, which is *significantly worse* than the baseline (p = 0.004) despite being
purpose-built for this kind of transfer.

**Transfer learning still bought something, just not accuracy.** ConvNeXt reached 0.987 validation accuracy
within 10 epochs at 2e-4; the baseline needed 30 epochs at 1e-3 and peaked at 0.988 on epoch 24. On
a larger dataset, or with stronger augmentation, the pretrained features would be expected to pull ahead —
here 5,216 training images with a strong class prior are enough for a plain CNN to catch up.

**The ensembles do not beat their best member.** The ConvNeXt+EfficientNet ensemble has the best AUC
(0.970) but the all-three ensemble is worse than the plain baseline on accuracy. The disagreement table
explains why: the arms disagree on only 6.7–7.9% of test images, with kappa around 0.78. When members are
right and wrong on nearly the same images, averaging cannot cancel anything. Adding the baseline CNN to the
ensemble adds almost no diversity, because it already agrees with the others.

**Specificity is the weak point everywhere.** Every arm has PNEUMONIA recall near or above 0.99 while
NORMAL recall sits between 0.47 and 0.58. In other words 99–123 of the 234 healthy patients are still
flagged as pneumonia. That is the expected behaviour of these models on this test split, and it is the
number to look at before quoting accuracy.

### Figures

Extracted from the executed notebook's saved outputs into `figures/`:

| Figure | File | What it shows |
|---|---|---|
| ROC curves, all five arms | `figures/roc_comparison.png` | Threshold-free ranking quality — ConvNeXt and the ensembles sit above the baseline |
| Confusion matrices, three base models | `figures/confusion_matrices.png` | Where the errors land; makes the specificity cost visible |
| Training and validation curves, three base models | `figures/training_curves.png` | The baseline needs 30 epochs to match what ConvNeXt does in 10 |

![ROC curves for all five arms](figures/roc_comparison.png)

![Confusion matrices for the three base models](figures/confusion_matrices.png)

![Training and validation curves for the three base models](figures/training_curves.png)

---

## How to run

**Requirements.** Python 3.10+ with:

| Package | Version used |
|---|---|
| `torch` | 2.11.0+cu130 |
| `torchvision` | 0.26.0+cu130 |
| `scikit-learn` | 1.9.1 |
| `numpy` | 2.5.4 |
| `pandas` | 3.1.1 |
| `matplotlib` | 3.11.2 |
| `scipy` | 1.17.1 |

Install the matching pair of PyTorch wheels from the same index, otherwise `torchvision` will fail to load
with `RuntimeError: operator torchvision::nms does not exist`:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install scikit-learn numpy pandas matplotlib scipy
```

**Steps.**

1. Download the dataset and place the zip next to the notebook (see [Dataset](#dataset)).
2. Open `pneumonia_detection_convnext_efficientnet_ensemble.ipynb`.
3. Run all cells top to bottom. The first run of Sections 5 and 6 downloads ImageNet weights (~110 MB for
   ConvNeXt-Tiny, ~21 MB for EfficientNet-B0); the baseline CNN downloads nothing.
4. Results appear in Section 10. The three plots there are the same ones saved in `figures/`.

**Runtime.** About 31 minutes end to end on a laptop RTX 4050, at 37–39 s per epoch: roughly 19 min for the
30-epoch baseline, 6 min for ConvNeXt-Tiny and 6 min for EfficientNet-B0 over 10 epochs each.

On CPU everything still runs, but many times slower, and the baseline CNN is hit hardest because it needs the
most epochs. Lowering `EPOCHS_BASELINE` is a reasonable trade, as long as you say so when reporting results —
an under-trained baseline makes transfer learning look better than it is.

---

## Repository layout

```
Pytorch/
├── README.md                                            ← this file
├── .gitignore
├── pneumonia_detection_convnext_efficientnet_ensemble.ipynb   ← the project, with saved outputs
├── figures/                                             ← plots extracted from the run
│   ├── roc_comparison.png
│   ├── confusion_matrices.png
│   └── training_curves.png
├── data/                                                ← extracted dataset, git-ignored (1.2 GB)
└── … other coursework notebooks (unrelated)
```

This repository also holds other, unrelated coursework notebooks (`NNDL&NLP/`, `Old/`). Only the pneumonia
notebook and the files listed above belong to this project.

---

## Limitations

- **Single-hospital paediatric data.** The model would not transfer to adult or multi-site radiographs
  without re-training and re-validation.
- **Image-level labels.** Splitting is patient-aware, but the label is not a patient-level outcome, and a
  per-patient assessment would be the clinically meaningful unit.
- **Small validation set.** Thresholds are tuned on roughly 780 images, so the operating point itself
  carries uncertainty.
- **Threshold selection.** Youden's J treats sensitivity and specificity as equally important. In a screening
  context a higher-recall operating point would be preferable; the notebook's per-class metrics make that
  adjustment possible, but does not perform it.
- **Research exercise, not a diagnostic tool.** This project is a deep-learning coursework comparison. It is
  not a medical device and must not be used to inform patient care.

---

## Author

Pranam — Neural Networks and Deep Learning coursework.

Dataset credit: Kermany, Goldbaum, Cai, Valentim, Liang and Baxter, *Identifying Medical Diagnoses and
Treatable Diseases by Image-Based Deep Learning*, Cell 172(5), 2018. Redistributed on Kaggle; see that page
for the licence terms that apply to the images, which are not included in this repository.