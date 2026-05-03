# CNN-Based Traffic Sign Recognition with Cross-Domain Generalization Analysis

This repository contains the source code and results for a CNN-based traffic sign classification project. The project uses GTSRB as the main clean traffic sign dataset and evaluates cross-domain generalization on GTSDB real-road crops and Synset Signset Germany synthetic traffic sign images.

The main research question is:

> Can a model that performs well on clean cropped traffic sign images also generalize to real-road crops and synthetic traffic sign variations?

---

## 1. Project Goal

Traffic Sign Recognition (TSR) is an important computer vision task for driver assistance systems and autonomous driving. This project builds and compares CNN-based traffic sign classifiers for 43 German traffic sign classes.

The project focuses on:

- training a custom baseline CNN,
- fine-tuning an improved EfficientNet-B0 model,
- analyzing domain shift between clean and real-road images,
- improving generalization through mixed-domain training,
- evaluating the final model with learning curves, confusion matrices, class-wise analysis, and failure cases,
- and demonstrating inference through a lightweight Gradio demo.

---

## 2. Datasets

The original datasets are not included in this repository because of file size. Please download them from the links below.

### GTSRB - German Traffic Sign Recognition Benchmark

GTSRB is used as the main training and in-domain evaluation dataset.

- 43 traffic sign classes
- 51,839 images
- clean, centered, cropped traffic sign images
- used for baseline training and main classification evaluation

Dataset link:  
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

### GTSDB - German Traffic Sign Detection Benchmark

GTSDB is used for real-road domain adaptation and external evaluation.

- real-road traffic sign images
- bounding-box crops are generated from annotations
- used to analyze domain shift from clean GTSRB images to real-road crops

Dataset link:  
https://www.kaggle.com/datasets/safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb

### Synset Signset Germany

Synset Signset Germany is used for synthetic-domain evaluation.

- synthetic traffic sign images
- only labels corresponding to GTSRB classes 0-42 are used
- 5% is used for training in SynsetMix experiments
- 95% is used for synthetic-domain evaluation

Dataset link:  
https://synset.de/datasets/synset-signset-ger/

---

## 3. Models

| Model | Description |
|---|---|
| Baseline CNN | Custom 3-block CNN trained from scratch |
| EfficientNet-B0 | ImageNet-pretrained EfficientNet-B0 fine-tuned for 43 classes |
| EfficientNet Robust | EfficientNet-B0 with stronger augmentation |
| EfficientNet Mixed | EfficientNet-B0 trained with GTSRB + GTSDB |
| EfficientNet Final | GTSRB + GTSDB mixed model evaluated on GTSRB, GTSDB, and Synset |
| SynsetMix | EfficientNet-B0 trained with GTSRB + GTSDB + 5% Synset |
| SynsetMix v2 | SynsetMix with 96x96 input resolution and batch size 32 |
| SynsetMix v3 | v2 with horizontal flip removed to reduce direction-related label noise |

---

## 4. Main Results

| Experiment | GTSRB | GTSDB | Synset |
|---|---:|---:|---:|
| Baseline CNN | 96.9% | 4.2% | - |
| EfficientNet-B0 | 95.2% | 3.8% | - |
| EfficientNet Robust | 87.0% | 2.1% | - |
| EfficientNet Mixed | 95.1% | 50.6% | - |
| EfficientNet Final | 94.9% | 58.4% | 56.2% |
| EfficientNet SynsetMix | 94.8% | 59.2% | 69.5% |
| SynsetMix v3 | 99.40% | 99.88%* | 91.14% |

\* The GTSDB v3 score is measured on GTSDB Full/reference evaluation. Since part of GTSDB was used during mixed-domain training, this score should not be interpreted as a pure unseen external-test accuracy.

---

## 5. Key Findings

1. GTSRB-only models achieved high in-domain accuracy but failed to generalize to GTSDB.
2. Strong augmentation alone did not solve the domain-shift problem.
3. Adding real-road GTSDB crops improved real-road generalization.
4. Adding a small portion of Synset improved synthetic-domain performance.
5. Increasing input resolution from 48x48 to 96x96 helped preserve fine visual details.
6. Removing horizontal flip in v3 reduced semantic inconsistency for direction-sensitive signs.

---

## 6. Repository Structure

```text
cnn-traffic-sign-recognition/
├── README.md
├── requirements.txt
├── config.py
├── dataset.py
├── train.py
├── evaluate.py
├── external_test.py
├── model_baseline.py
├── model_improved.py
├── main.py
├── setup.py
├── demo_v3_clean.py
│
├── scripts/
│   ├── train_mixed.py
│   ├── train_final.py
│   ├── train_synset_mix.py
│   ├── train_synset_mix_v2.py
│   ├── train_synset_mix_v3.py
│   ├── finetune_gtsdb.py
│   └── finetune_synset.py
│
├── results/
│   ├── curves/
│   ├── confusion_matrices/
│   └── failure_cases/
│
├── reports/
│   └── Project1_Final_Report.pdf
│
└── presentation/
    └── final_presentation.pptx
```

---

## 7. Environment Setup

This project was implemented with Python and PyTorch.

Recommended Python version:

```text
Python 3.10 or higher
```

Clone the repository:

```bash
git clone https://github.com/<your-username>/cnn-traffic-sign-recognition.git
cd cnn-traffic-sign-recognition
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 8. Dataset Preparation

### Option 1: Automatic setup

Run the setup script first:

```bash
python setup.py
```

The setup script checks or downloads the required datasets and creates the GTSDB crop CSV file.

It prepares:

- GTSRB dataset path,
- GTSDB dataset path,
- GTSDB cropped images,
- `gtsdb_crops.csv`,
- Synset parquet file.

### Option 2: Manual setup

If dataset paths are different in your environment, edit `config.py`.

Example:

```python
GTSDB_ROOT = "/path/to/gtsdb"
GTSDB_CSV = f"{GTSDB_ROOT}/gtsdb_crops.csv"
OUTPUT_DIR = "./outputs"
```

The GTSDB training scripts expect a CSV file with this format:

```text
Path,ClassId
gtsdb_crops/00/example.png,0
gtsdb_crops/01/example.png,1
```

If `gtsdb_crops.csv` does not exist, run:

```bash
python setup.py
```

If the GTSDB crop-generation script is stored in `scripts/`, run:

```bash
PYTHONPATH=. python scripts/make_gtsdb_crops.py
```

For Synset, the scripts expect the parquet file path used in the code:

```text
/home/work/AI/data/synset-signset-germany/cycles/validation.parquet
```

If your Synset file is stored somewhere else, update the `SYNSET_PARQUET` path inside the Synset training scripts.

---

## 9. Training Instructions

For reproducibility, run the scripts in the order below. Some later scripts depend on files or checkpoints created by earlier steps.

### Step 1: Baseline CNN, EfficientNet-B0, and Robust Augmentation

```bash
python main.py
```

This script trains and evaluates:

- Baseline CNN,
- EfficientNet-B0,
- EfficientNet-B0 with robust augmentation.

It also saves learning curves, confusion matrices, and failure-case examples in the `outputs/` directory.

Expected outputs include:

```text
outputs/curve_baseline.png
outputs/curve_efficientnet.png
outputs/curve_efficientnet_robust.png
outputs/confusion_baseline.png
outputs/confusion_efficientnet.png
outputs/confusion_efficientnet_robust.png
outputs/failures_baseline.png
outputs/failures_efficientnet.png
outputs/failures_efficientnet_robust.png
```

### Step 2: Mixed-domain training with GTSRB + GTSDB

If `train_mixed.py` is in the repository root:

```bash
python train_mixed.py
```

If it is inside the `scripts/` folder:

```bash
PYTHONPATH=. python scripts/train_mixed.py
```

This experiment adds real-road GTSDB crops to the GTSRB training data.

Expected outputs include:

```text
outputs/best_efficientnet_mixed.pth
outputs/curve_efficientnet_mixed.png
outputs/confusion_efficientnet_mixed.png
outputs/confusion_efficientnet_mixed_GTSDB_external.png
```

### Step 3: Final mixed model and Synset evaluation

```bash
PYTHONPATH=. python scripts/train_final.py
```

If `train_final.py` is in the repository root:

```bash
python train_final.py
```

This script trains on GTSRB + GTSDB and evaluates on:

- GTSRB test set,
- GTSDB hold-out / full reference set,
- Synset Signset Germany.

Expected outputs include:

```text
outputs/best_efficientnet_final.pth
outputs/curve_efficientnet_final.png
```

### Step 4: SynsetMix training

```bash
PYTHONPATH=. python scripts/train_synset_mix.py
```

If `train_synset_mix.py` is in the repository root:

```bash
python train_synset_mix.py
```

This script trains with:

```text
GTSRB + GTSDB + 5% Synset
```

and evaluates on:

```text
GTSRB, GTSDB Full, Synset 95%
```

Expected outputs include:

```text
outputs/best_efficientnet_synset_mix.pth
outputs/curve_efficientnet_synset_mix.png
```

### Step 5: SynsetMix v2

```bash
PYTHONPATH=. python scripts/train_synset_mix_v2.py
```

If `train_synset_mix_v2.py` is in the repository root:

```bash
python train_synset_mix_v2.py
```

This version uses:

- input size: 96x96,
- batch size: 32,
- SynsetMix training data,
- horizontal flip still included in part of the augmentation pipeline.

Expected outputs include:

```text
outputs/best_efficientnet_v2.pth
outputs/curve_v2.png
outputs/cm_v2_synset.png
outputs/cm_v2_synset.npy
```

### Step 6: SynsetMix v3

```bash
PYTHONPATH=. python scripts/train_synset_mix_v3.py
```

If `train_synset_mix_v3.py` is in the repository root:

```bash
python train_synset_mix_v3.py
```

This is the final version. It uses:

- input size: 96x96,
- batch size: 32,
- GTSRB + GTSDB + 5% Synset,
- no horizontal flip augmentation.

Expected outputs include:

```text
outputs/best_efficientnet_v3.pth
outputs/curve_v3.png
outputs/cm_v3_synset.png
outputs/cm_v3_synset.npy
```

Final v3 evaluation example:

```text
GTSRB Test Acc: 99.40%
GTSDB Full Acc: 99.88%
Synset Test Acc: 91.14%
```

Important note: The GTSDB Full score is a reference/adaptation score because some GTSDB images were used during mixed-domain training. It should not be interpreted as a pure unseen external-test score.

---

## 10. Evaluation Instructions

Most training scripts automatically evaluate the model after training.

The evaluation outputs include:

- final test accuracy,
- classification report,
- confusion matrix,
- failure-case visualization,
- learning curve.

The main evaluation functions are implemented in:

```text
evaluate.py
```

`full_evaluate()` computes test accuracy, classification report, and confusion matrix.

`show_failure_cases()` saves misclassified examples.

Generated outputs are saved to:

```text
outputs/
```

For final submission, selected output images were copied into:

```text
results/curves/
results/confusion_matrices/
results/failure_cases/
```

---

## 11. Inference Demo

Run the Gradio demo:

```bash
python demo_v3_clean.py
```

The demo compares:

- Baseline CNN,
- Final V2 EfficientNet-B0,
- Final V3 EfficientNet-B0.

The demo returns the top-5 predicted classes with confidence scores.

Important note:

This project implements a classification model, not a detection model. Therefore, the input image should be a cropped traffic sign image, not a full road-scene image.

If the default Gradio port is already in use, run:

```bash
GRADIO_SERVER_PORT=7861 python demo_v3_clean.py
```

or edit the port inside `demo_v3_clean.py`.

---

## 12. Expected Output Files

After running the full pipeline, the following files may be created:

```text
outputs/best_baseline.pth
outputs/best_efficientnet.pth
outputs/best_efficientnet_robust.pth
outputs/best_efficientnet_mixed.pth
outputs/best_efficientnet_final.pth
outputs/best_efficientnet_synset_mix.pth
outputs/best_efficientnet_v2.pth
outputs/best_efficientnet_v3.pth

outputs/curve_baseline.png
outputs/curve_efficientnet.png
outputs/curve_efficientnet_robust.png
outputs/curve_efficientnet_mixed.png
outputs/curve_efficientnet_final.png
outputs/curve_efficientnet_synset_mix.png
outputs/curve_v2.png
outputs/curve_v3.png

outputs/confusion_baseline.png
outputs/confusion_efficientnet.png
outputs/confusion_efficientnet_robust.png
outputs/confusion_efficientnet_mixed.png
outputs/cm_v2_synset.png
outputs/cm_v3_synset.png

outputs/failures_baseline.png
outputs/failures_efficientnet.png
outputs/failures_efficientnet_robust.png
```

Large checkpoint files are not included in this repository. If needed, download them separately and place them in:

```text
outputs/
```

---

## 13. Inference Checkpoint Note

The Gradio demo expects trained checkpoint files such as:

```text
outputs/best_baseline.pth
outputs/best_efficientnet_v2.pth
outputs/best_efficientnet_v3.pth
```

If these files are not included in the repository, train the corresponding models first or place downloaded checkpoints into the `outputs/` directory.

---

## 14. References

[1] J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel, "The German Traffic Sign Recognition Benchmark: A multi-class classification competition," in Proc. IEEE International Joint Conference on Neural Networks (IJCNN), 2011.

[2] S. Houben, J. Stallkamp, J. Salmen, M. Schlipsing, and C. Igel, "Detection of Traffic Signs in Real-World Images: The German Traffic Sign Detection Benchmark," in Proc. IEEE International Joint Conference on Neural Networks (IJCNN), 2013.

[3] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in Proc. International Conference on Machine Learning (ICML), 2019.

[4] Kaggle, "GTSRB - German Traffic Sign Recognition Benchmark," Kaggle Dataset.  
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

[5] Kaggle, "German Traffic Sign Detection Benchmark - GTSDB," Kaggle Dataset.  
https://www.kaggle.com/datasets/safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb

[6] Synset, "Synset Signset Germany," Synset Dataset.  
https://synset.de/datasets/synset-signset-ger/
