# Vehicle Classification Under Low-Light Conditions

This repository contains experiments comparing two approaches for improving vehicle classification accuracy on severely degraded nighttime images:
- **Experiment 2**: Enhanced preprocessing with pretrained models
- **Experiment 3**: Fine-tuned models on degraded images

## Dataset

Download the dataset here: https://drive.google.com/drive/u/0/folders/1m0AXwPMiRXJRXr7Ho5PGOJAOAmLyG743?q=sharedwith:public%20parent:1m0AXwPMiRXJRXr7Ho5PGOJAOAmLyG743

**Directory Structure:**
```
├── bad_light/                          # Original degraded nighttime images
│   ├── bicycle/
│   ├── boat/
│   ├── bus/
│   ├── car/
│   ├── helicopter/
│   ├── motorcycle/
│   └── truck/
├── bad_light_proper_enhancement/       # Enhanced images (generated)
├── saved_models/                       # Pretrained models
│   ├── mobilenet2.h5
│   └── InceptionV3.h5
└── finetuned_models/                   # Fine-tuned models
    └── mobilenet_transfer_learning_final.h5
```

## Scripts

### 1. `run_bad_light_final.py`
Creates enhanced dataset by reversing day-to-night degradation pipeline.

**What it does:**
- Removes vignette effects
- Applies sharpening and denoising
- Corrects white balance
- Brightens images
- Saves enhanced images to `bad_light_proper_enhancement/`

**Run:**
```bash
python run_bad_light_final.py
```

### 2. `fine_tune.py` 
Evaluates both experiments without regenerating data or retraining models.

**What it does:**
- **Experiment 2**: Tests pretrained models on enhanced images
- **Experiment 3**: Tests fine-tuned models on original degraded images
- Generates comparison statistics

**Run:**
```bash
python fine_tune.py
```

### 3. `run_two_experiments.py`
Alternative evaluation script with ensemble prediction support.

**Run:**
```bash
python run_two_experiments.py
```

## Quick Start

1. **Generate enhanced images:**
```bash
   python run_bad_light_final.py
```

2. **Run experiments:**
```bash
   python fine_tune.py
```

3. **View results:**
   - `results_experiment2_enhanced_preprocessing.csv`
   - `results_experiment3_finetuned.csv`

## Results

The scripts automatically compare:
- Overall accuracy for both experiments
- Per-class accuracy breakdown
- Statistical significance of differences

Results show which approach (preprocessing vs fine-tuning) performs better for vehicle classification under severe lighting degradation.
