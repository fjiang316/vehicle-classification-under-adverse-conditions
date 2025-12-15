# Execution Instruction Guide for Foggy Distortion Experiment

All files in this folders are related to foggy condition (distortion) related experiment in this project. Please follow the guideline below if you want to run the code.

Dataset can be downloaded from: https://drive.google.com/file/d/1kGqC6a4gG2upr7aw-FOwlAgkAq5rsjfn/view?usp=sharing

# Vehicle Classification under foggy Conditions

This repository contains two experimental pipelines for **vehicle type classification under foggy conditions**:

1. **foggy dehazing + classification pipeline** (`run_foggy_final.py`)
2. **Foggy-condition fine-tuning pipeline** (`foggy_finetune.py`)

Both scripts are designed to work with a subset of vehicle classes and pretrained CNN models (e.g., MobileNetV2, InceptionV3).

---

## 1. Environment Requirements

* Python >= 3.8
* TensorFlow / Keras
* OpenCV (`cv2`)
* NumPy
* Pandas
* Matplotlib

Recommended installation:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib
```

---

## 2. Dataset Directory Structure Requirements

### 2.1 Foggy Dataset Structure (`run_foggy_final.py`)

#### Required directory structure:

```
raw_data/
├── bicycle/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── boat/
├── bus/
├── car/
├── helicopter/
├── motorcycle/
└── truck/
```

Each subfolder name **must exactly match** the following list:

```
bicycle, boat, bus, car, helicopter, motorcycle, truck
```

The script will automatically create an enhanced dataset with the same structure:

```
test_dehazed/
├── bicycle/
├── boat/
├── bus/
├── car/
├── helicopter/
├── motorcycle/
└── truck/
```

---

### 2.2 Foggy Dataset Structure (`foggy_finetune.py`)

The foggy pipeline **fine-tunes a pretrained model** using labeled foggy images.

#### Required directory structure:

```
foggy_dataset/
├── train/
│   ├── bicycle/
│   ├── boat/
│   ├── bus/
│   ├── car/
│   ├── helicopter/
│   ├── motorcycle/
│   └── truck/
└── val/
    ├── bicycle/
    ├── boat/
    ├── bus/
    ├── car/
    ├── helicopter/
    ├── motorcycle/
    └── truck/
```

> ⚠️ Important:
>
> * Training and validation **must have identical class subfolders**
> * Folder names are treated as class labels by `flow_from_directory()`

---

## 3. Pretrained Models

The following pretrained models are expected:

```
saved_models/
├── mobilenet2.h5
└── InceptionV3.h5
```

* `run_foggy_final.py` uses **both models** for ensemble prediction

---

## 4. How to Run

### 4.1 Dehazed Enhancement & Evaluation

Edit paths if needed:

```python
TEST_DIR = "foggy/"
ENHANCED_DIR = "foggy_enhanced/"
```

Run:

```bash
python run_foggy_final.py
```

#### Output:

* Enhanced images saved to `bad_light_enhanced/`
* CSV results:

  * `experiment1_baseline.csv`
  * `experiment2_enhanced.csv`
* Printed comparison of:

  * Overall accuracy
  * Per-class accuracy

---

### 4.2 Foggy Condition Fine-Tuning

Edit dataset and model paths:

```python
FOGGY_TRAIN_DIR = ".../foggy_dataset/train"
FOGGY_VAL_DIR   = ".../foggy_dataset/val"
```

Run:

```bash
python foggy_finetune.py
```

#### What this script does:

1. Loads a pretrained MobileNetV2 model
2. Freezes most backbone layers
3. Fine-tunes the top layers on foggy images
4. Saves a new model:

```
mobilenet2_finetuned_foggy.h5
```

---

## 5. Experimental Purpose

* **Low-light pipeline**: tests whether classical image enhancement (MSR, CLAHE, Gamma) improves inference accuracy
* **Foggy fine-tuning**: addresses domain shift by adapting CNN weights to fog-degraded inputs

These two scripts support a **comparative study between preprocessing-based and learning-based robustness methods**.

---

## 6. Notes for GitHub Submission

If the dataset is too large to upload:

* Upload **directory structure only** (empty folders)
* Provide a download link via:

  * Google Drive
  * OneDrive
  * Kaggle Dataset

Mention in README:

> "Due to size limitations, datasets are provided via external link."

---

## 7. Contact / Notes

This code is intended for academic experiments (ECE / CV coursework).
Paths are written explicitly for clarity and may need adjustment per system.
