#!/usr/bin/env python3
"""
Direct Evaluation Script for Experiments 2 & 3
- Experiment 2: Enhanced preprocessing with pretrained models
- Experiment 3: Fine-tuned models on original bad_light images
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
import pandas as pd
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories
TEST_DIR_ORIGINAL = "bad_light/"  # Original dark images
TEST_DIR_ENHANCED = "bad_light_proper_enhancement/"  # Already enhanced images

# Model Paths
PRETRAINED_MODELS = {
    'mobilenet': 'saved_models/mobilenet2.h5',
    'inception': 'saved_models/InceptionV3.h5',
}

# Updated with your actual fine-tuned model names
FINETUNED_MODELS = {
    'mobilenet_ft': 'finetuned_models/mobilenet_transfer_learning_final.h5',
    # Add other fine-tuned models if you have them:
    # 'inception_ft': 'finetuned_models/inception_transfer_learning_final.h5',
}

# Class Configuration
MODEL_CLASSES = [
    'Ambulance', 'Barge', 'Bicycle', 'Boat', 'Bus', 'Car', 
    'Cart', 'Caterpillar', 'Helicopter', 'Limousine', 
    'Motorcycle', 'Segway', 'Snowmobile', 'Tank', 'Taxi', 
    'Truck', 'Van'
]

YOUR_CLASSES = ['bicycle', 'boat', 'bus', 'car', 'helicopter', 'motorcycle', 'truck']

CLASS_MAPPING = {
    'bicycle': 'Bicycle',
    'boat': 'Boat',
    'bus': 'Bus',
    'car': 'Car',
    'helicopter': 'Helicopter',
    'motorcycle': 'Motorcycle',
    'truck': 'Truck',
}

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models_from_dict(model_dict):
    """Load models from a dictionary of paths"""
    models = []
    model_names = []
    
    for name, path in model_dict.items():
        if os.path.exists(path):
            try:
                print(f"  Loading {name}...")
                model = load_model(path)
                models.append(model)
                model_names.append(name)
                print(f"  ✅ {name}")
            except Exception as e:
                print(f"  ❌ {name}: {e}")
        else:
            print(f"  ⚠️  {name}: file not found at {path}")
    
    return models, model_names


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def ensemble_predictions(models, img):
    """Ensemble predictions from multiple models"""
    predictions = [model.predict(img, verbose=0) for model in models]
    predictions = np.array(predictions)
    summed = np.sum(predictions, axis=0)
    class_index = np.argmax(summed, axis=1)[0]
    confidence = summed[0][class_index] / len(models)
    return class_index, confidence


def single_model_prediction(model, img):
    """Single model prediction (for when only one model is available)"""
    prediction = model.predict(img, verbose=0)
    class_index = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][class_index]
    return class_index, confidence


def run_evaluation(models, data_dir, experiment_name):
    """Run predictions on a dataset"""
    print(f"\n{'='*70}")
    print(f"{experiment_name}")
    print("="*70)
    
    all_results = []
    correct = 0
    total = 0
    
    class_stats = {}
    
    # Determine if we should use ensemble or single model
    use_ensemble = len(models) > 1
    print(f"Using {'ensemble' if use_ensemble else 'single model'} prediction")
    
    for class_name in YOUR_CLASSES:
        class_path = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_path):
            print(f"  ⚠️  Skipping {class_name}/ (not found)")
            continue
        
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            print(f"  ⚠️  Skipping {class_name}/ (no images)")
            continue
        
        print(f"\n  {class_name}/ ({len(images)} images)")
        
        class_correct = 0
        
        for i, img_file in enumerate(images):
            if i > 0 and i % 20 == 0:
                print(f"    {i}/{len(images)}...")
            
            img_path = os.path.join(class_path, img_file)
            
            try:
                img = plt.imread(img_path)
                img_resized = cv2.resize(img, (224, 224))
                img_batch = img_resized[np.newaxis, ...].astype(np.float32)
                
                # Use appropriate prediction method
                if use_ensemble:
                    pred_idx, confidence = ensemble_predictions(models, img_batch)
                else:
                    pred_idx, confidence = single_model_prediction(models[0], img_batch)
                
                predicted_model_class = MODEL_CLASSES[pred_idx]
                
                # Map to your class names
                predicted_your_class = None
                for your_cls, model_cls in CLASS_MAPPING.items():
                    if model_cls == predicted_model_class:
                        predicted_your_class = your_cls
                        break
                
                if predicted_your_class is None:
                    predicted_your_class = 'unknown'
                
                is_correct = (predicted_your_class.lower() == class_name.lower())
                if is_correct:
                    class_correct += 1
                    correct += 1
                
                total += 1
                
                all_results.append({
                    'true_class': class_name,
                    'filename': img_file,
                    'predicted_class': predicted_your_class,
                    'confidence': confidence,
                    'correct': is_correct
                })
                
            except Exception as e:
                print(f"    ❌ Error processing {img_file}: {e}")
                continue
        
        class_acc = (class_correct / len(images)) * 100 if images else 0
        class_stats[class_name] = {
            'correct': class_correct,
            'total': len(images),
            'accuracy': class_acc
        }
        print(f"    ✓ {class_correct}/{len(images)} = {class_acc:.1f}%")
    
    df = pd.DataFrame(all_results)
    overall_acc = (correct / total) * 100 if total > 0 else 0
    
    print(f"\n  Overall: {correct}/{total} = {overall_acc:.2f}%")
    
    return df, overall_acc, class_stats


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Run Experiment 2 and Experiment 3 evaluations
    """
    print("\n" + "="*70)
    print("DIRECT EVALUATION: EXPERIMENTS 2 & 3")
    print("="*70)
    
    # ========================================================================
    # EXPERIMENT 2: Enhanced Preprocessing + Pretrained Models
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: ENHANCED PREPROCESSING")
    print("="*70)
    print("Dataset: Enhanced images (already generated)")
    print("Models:  Pretrained models (no fine-tuning)")
    
    # Check if enhanced directory exists
    if not os.path.exists(TEST_DIR_ENHANCED):
        print(f"\n❌ ERROR: Enhanced directory not found: {TEST_DIR_ENHANCED}")
        print("Please run the enhancement script first to generate enhanced images.")
        exp2_success = False
    else:
        # Load pretrained models
        print("\nLoading pretrained models...")
        pretrained_models, pretrained_names = load_models_from_dict(PRETRAINED_MODELS)
        
        if not pretrained_models:
            print("❌ No pretrained models loaded! Cannot run Experiment 2.")
            exp2_success = False
        else:
            print(f"\n✅ Loaded {len(pretrained_models)} models: {', '.join(pretrained_names)}")
            
            # Run evaluation
            results_exp2, acc_exp2, stats_exp2 = run_evaluation(
                pretrained_models, 
                TEST_DIR_ENHANCED, 
                "EXPERIMENT 2: Enhanced Preprocessing"
            )
            
            # Save results
            results_exp2.to_csv('results_experiment2_enhanced_preprocessing.csv', index=False)
            print(f"\n📊 Results saved: results_experiment2_enhanced_preprocessing.csv")
            exp2_success = True
    
    # ========================================================================
    # EXPERIMENT 3: Fine-tuned Models + Original Images
    # ========================================================================
    print("\n\n" + "="*70)
    print("EXPERIMENT 3: FINE-TUNED MODELS")
    print("="*70)
    print("Dataset: Original bad_light images (no preprocessing)")
    print("Models:  Fine-tuned models")
    
    # Load fine-tuned models
    print("\nLoading fine-tuned models...")
    finetuned_models, finetuned_names = load_models_from_dict(FINETUNED_MODELS)
    
    if not finetuned_models:
        print("❌ No fine-tuned models loaded! Cannot run Experiment 3.")
        print("Please check that fine-tuned models exist at the specified paths.")
        exp3_success = False
    else:
        print(f"\n✅ Loaded {len(finetuned_models)} models: {', '.join(finetuned_names)}")
        
        # Run evaluation
        results_exp3, acc_exp3, stats_exp3 = run_evaluation(
            finetuned_models, 
            TEST_DIR_ORIGINAL, 
            "EXPERIMENT 3: Fine-tuned Models"
        )
        
        # Save results
        results_exp3.to_csv('results_experiment3_finetuned.csv', index=False)
        print(f"\n📊 Results saved: results_experiment3_finetuned.csv")
        exp3_success = True
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    if exp2_success and exp3_success:
        print(f"\n\n{'='*70}")
        print("EXPERIMENT COMPARISON")
        print("="*70)
        
        print(f"\n  Experiment 2 (Enhanced Preprocessing): {acc_exp2:6.2f}%")
        print(f"  Experiment 3 (Fine-tuned Models):      {acc_exp3:6.2f}%")
        print(f"  {'─'*50}")
        
        difference = acc_exp3 - acc_exp2
        print(f"  Difference (Exp3 - Exp2):              {difference:+6.2f}%")
        
        if difference > 5:
            print(f"\n  ✅ Fine-tuning significantly outperforms preprocessing!")
        elif difference < -5:
            print(f"\n  ✅ Preprocessing significantly outperforms fine-tuning!")
        else:
            print(f"\n  ⚖️  Both approaches yield similar results")
        
        # Per-class comparison
        print(f"\n📈 Per-Class Comparison:")
        print("-"*70)
        print(f"{'Class':<12} {'Exp2 (Preproc)':<15} {'Exp3 (Finetune)':<15} {'Difference':<12}")
        print("-"*70)
        
        for class_name in YOUR_CLASSES:
            if class_name in stats_exp2 and class_name in stats_exp3:
                acc2 = stats_exp2[class_name]['accuracy']
                acc3 = stats_exp3[class_name]['accuracy']
                diff = acc3 - acc2
                
                indicator = "✅" if abs(diff) > 10 else "  "
                print(f"{class_name:<12} {acc2:5.1f}% {indicator:>7}   {acc3:5.1f}% {indicator:>7}   {diff:+6.1f}%")
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE!")
    print("="*70)
    
    if exp2_success:
        print(f"\n📁 Enhanced images: {TEST_DIR_ENHANCED}")
        print(f"📊 Exp2 results: results_experiment2_enhanced_preprocessing.csv")
    
    if exp3_success:
        print(f"📁 Original images: {TEST_DIR_ORIGINAL}")
        print(f"📊 Exp3 results: results_experiment3_finetuned.csv")


if __name__ == "__main__":
    main()