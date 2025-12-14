#!/usr/bin/env python3
"""
Properly Calibrated Enhancement for VERY DARK Images
Enhanced with improved reversal of day-to-night degradation
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
TEST_DIR = "bad_light/"
ENHANCED_DIR = "bad_light_proper_enhancement/"

MODEL_PATHS = {
    'mobilenet': 'saved_models/mobilenet2.h5',
    'inception': 'saved_models/InceptionV3.h5',
}

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
# ENHANCEMENT FUNCTIONS - IMPROVED REVERSAL
# ============================================================================

def invert_bad_light_v2(img):
    """
    Reverse the A + D degradation pipeline more accurately
    
    Your degradation order:
    1. apply_dim (0.55x brightness)
    2. apply_relight (white balance + another dim)
    3. apply_relight_local (noise + blur + vignette)
    
    Reversal order (reverse of above):
    1. Remove vignette
    2. Sharpen (undo blur)
    3. Denoise (reduce noise)
    4. Undo white balance shift
    5. Brighten (undo double dimming)
    """
    
    # -----------------------------
    # Step 1: REMOVE VIGNETTE
    # -----------------------------
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    center_x, center_y = w / 2, h / 2
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.max(dist)
    
    # Inverse vignette (strength=0.45 in degradation)
    vignette_mask = 1 - 0.45 * (dist / max_dist)
    vignette_mask = np.clip(vignette_mask, 0.3, 1.0)  # prevent division by very small numbers
    vignette_mask = np.dstack([vignette_mask] * 3)
    
    devignetted = img.astype(np.float32) / vignette_mask
    devignetted = np.clip(devignetted, 0, 255).astype(np.uint8)
    
    # -----------------------------
    # Step 2: SHARPEN (undo GaussianBlur)
    # -----------------------------
    # Unsharp mask is better than simple sharpening kernel
    gaussian_blur = cv2.GaussianBlur(devignetted, (5, 5), 0)
    sharpened = cv2.addWeighted(devignetted, 1.5, gaussian_blur, -0.5, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # -----------------------------
    # Step 3: DENOISE (reduce luminance + chrominance noise)
    # -----------------------------
    # Your degradation adds sigma=18 luminance, sigma=8 chrominance
    denoised = cv2.fastNlMeansDenoisingColored(
        sharpened, None,
        h=12,              # increased for sigma=18 luminance noise
        hColor=10,         # for sigma=8 chrominance noise
        templateWindowSize=7,
        searchWindowSize=21
    )
    
    # -----------------------------
    # Step 4: UNDO WHITE BALANCE SHIFT
    # -----------------------------
    # Your apply_relight normalizes channels by their means
    # We need to reverse this
    img_float = denoised.astype(np.float32)
    
    # Estimate the white balance that was applied
    # Original: scale = gray / avg_channel
    # To reverse: divide by scale (multiply by avg_channel / gray)
    avg_channel = img_float.mean(axis=(0,1))
    gray = avg_channel.mean()
    
    # Reverse the scaling
    if gray > 1:  # avoid division by zero
        reverse_scale = avg_channel / gray
        # Apply conservative reversal to avoid over-correction
        reverse_scale = np.clip(reverse_scale, 0.8, 1.2)
        img_float *= reverse_scale
    
    wb_corrected = np.clip(img_float, 0, 255).astype(np.uint8)
    
    # -----------------------------
    # Step 5: BRIGHTEN (undo double dimming)
    # -----------------------------
    # Your pipeline applies 0.55x twice effectively:
    # - apply_dim: 0.55x
    # - apply_relight: includes another dim at 0.55x
    # Total dimming ≈ 0.55 * 0.55 = 0.3025
    
    # So we need to brighten by ~1/0.3025 = 3.3x
    # But let's be conservative to avoid oversaturation
    brighten_factor = 2.5
    
    brightened = wb_corrected.astype(np.float32) * brighten_factor
    brightened = np.clip(brightened, 0, 255).astype(np.uint8)
    
    # -----------------------------
    # Step 6: ADAPTIVE HISTOGRAM EQUALIZATION (optional)
    # -----------------------------
    # Only if still too dark
    gray_check = cv2.cvtColor(brightened, cv2.COLOR_RGB2GRAY)
    if np.mean(gray_check) < 80:
        lab = cv2.cvtColor(brightened, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Moderate CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        brightened = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    
    return brightened


def invert_bad_light_aggressive(img):
    """
    More aggressive reversal for very dark images
    Use this if invert_bad_light_v2 still leaves images too dark
    """
    
    # Start with v2
    enhanced = invert_bad_light_v2(img)
    
    # Additional aggressive enhancement if still dark
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    
    if brightness < 60:
        # Apply logarithmic brightening
        img_float = enhanced.astype(np.float32)
        c = 255 / np.log(1 + np.max(img_float))
        enhanced = c * np.log(1 + img_float)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # Strong CLAHE
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    
    return enhanced


def enhance_image(img):
    """
    MAIN ENHANCEMENT - Improved reversal of day-to-night degradation
    
    Automatically chooses between standard and aggressive enhancement
    based on image brightness.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    
    # Use aggressive enhancement for very dark images
    if brightness < 50:
        return invert_bad_light_aggressive(img)
    else:
        return invert_bad_light_v2(img)


# ============================================================================
# SAVE ENHANCED DATASET
# ============================================================================

def save_enhanced_dataset(input_dir, output_dir, classes):
    """
    Process entire dataset with improved enhancement method
    """
    print(f"\n{'='*70}")
    print("CREATING ENHANCED DATASET")
    print("="*70)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    brightness_stats = []
    
    for class_name in classes:
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        
        if not os.path.exists(input_class_dir):
            continue
        
        Path(output_class_dir).mkdir(parents=True, exist_ok=True)
        
        images = [f for f in os.listdir(input_class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            continue
        
        print(f"\n  {class_name}/ ({len(images)} images)")
        
        for i, img_file in enumerate(images):
            if i > 0 and i % 20 == 0:
                print(f"    {i}/{len(images)}...")
            
            input_path = os.path.join(input_class_dir, img_file)
            img = cv2.imread(input_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Track original brightness
            gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness_before = np.mean(gray_orig)
            
            # Apply enhancement
            enhanced = enhance_image(img_rgb)
            
            # Track enhanced brightness
            gray_enh = cv2.cvtColor(cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
            brightness_after = np.mean(gray_enh)
            
            brightness_stats.append({
                'class': class_name,
                'file': img_file,
                'before': brightness_before,
                'after': brightness_after,
                'gain': brightness_after - brightness_before
            })
            
            # Save
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            output_path = os.path.join(output_class_dir, img_file)
            cv2.imwrite(output_path, enhanced_bgr)
    
    # Show enhancement statistics
    print(f"\n{'='*70}")
    print("ENHANCEMENT STATISTICS")
    print("="*70)
    
    df_stats = pd.DataFrame(brightness_stats)
    avg_before = df_stats['before'].mean()
    avg_after = df_stats['after'].mean()
    avg_gain = df_stats['gain'].mean()
    
    print(f"\nBrightness transformation:")
    print(f"  Before enhancement: {avg_before:.1f}/255")
    print(f"  After enhancement:  {avg_after:.1f}/255")
    print(f"  Average gain:       +{avg_gain:.1f}")
    
    print(f"\nPer-class statistics:")
    for class_name in classes:
        class_df = df_stats[df_stats['class'] == class_name]
        if len(class_df) > 0:
            print(f"  {class_name:12s}: {class_df['before'].mean():.1f} → {class_df['after'].mean():.1f} (+{class_df['gain'].mean():.1f})")
    
    print(f"\n✅ Enhanced dataset created: {output_dir}/")
    
    # Save statistics
    stats_path = os.path.join(output_dir, 'enhancement_statistics.csv')
    df_stats.to_csv(stats_path, index=False)
    print(f"📊 Statistics saved: {stats_path}")
    
    return df_stats


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def load_models():
    """Load pre-trained models"""
    models = []
    print("\n" + "="*70)
    print("LOADING MODELS")
    print("="*70)
    
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                print(f"  Loading {name}...")
                model = load_model(path)
                models.append(model)
                print(f"  ✅ {name}")
            except Exception as e:
                print(f"  ❌ {name}: {e}")
        else:
            print(f"  ⚠️  {name}: file not found")
    
    return models


def ensemble_predictions(models, img):
    """Ensemble predictions from multiple models"""
    predictions = [model.predict(img, verbose=0) for model in models]
    predictions = np.array(predictions)
    summed = np.sum(predictions, axis=0)
    class_index = np.argmax(summed, axis=1)[0]
    confidence = summed[0][class_index] / len(models)
    return class_index, confidence


def run_predictions(models, data_dir, exp_name):
    """Run predictions on a dataset"""
    print(f"\n{'='*70}")
    print(f"{exp_name}")
    print("="*70)
    
    all_results = []
    correct = 0
    total = 0
    
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
                
                pred_idx, confidence = ensemble_predictions(models, img_batch)
                predicted_model_class = MODEL_CLASSES[pred_idx]
                
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
        print(f"    ✓ {class_correct}/{len(images)} = {class_acc:.1f}%")
    
    df = pd.DataFrame(all_results)
    overall_acc = (correct / total) * 100 if total > 0 else 0
    
    print(f"\nOverall: {correct}/{total} = {overall_acc:.2f}%")
    
    return df, overall_acc


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Full pipeline: enhance dataset, evaluate baseline and enhanced performance
    """
    print("\n" + "="*70)
    print("ENHANCED PREPROCESSING PIPELINE")
    print("="*70)
    
    # Step 1: Create enhanced dataset
    save_enhanced_dataset(TEST_DIR, ENHANCED_DIR, YOUR_CLASSES)
    
    # Step 2: Load models
    models = load_models()
    if not models:
        print("❌ No models loaded! Skipping evaluation.")
        return
    
    # Step 3: Baseline evaluation
    print("\n" + "="*70)
    print("BASELINE EVALUATION")
    print("="*70)
    results_baseline, acc_baseline = run_predictions(
        models, TEST_DIR, "BASELINE (No Enhancement)"
    )
    results_baseline.to_csv('results_baseline.csv', index=False)
    
    # Step 4: Enhanced evaluation
    results_enhanced, acc_enhanced = run_predictions(
        models, ENHANCED_DIR, "ENHANCED"
    )
    results_enhanced.to_csv('results_enhanced.csv', index=False)
    
    # Step 5: Comparison
    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON")
    print("="*70)
    
    improvement = acc_enhanced - acc_baseline
    
    print(f"\n  Baseline:   {acc_baseline:6.2f}%")
    print(f"  Enhanced:   {acc_enhanced:6.2f}%")
    print(f"  {'─'*30}")
    print(f"  Change:     {improvement:+6.2f}%")
    
    if improvement > 5:
        print(f"\n  ✅ SIGNIFICANT IMPROVEMENT! Enhancement works!")
    elif improvement > 2:
        print(f"\n  ✓ Moderate improvement")
    elif improvement > 0:
        print(f"\n  ✓ Small improvement")
    else:
        print(f"\n  ⚠️  No improvement - enhancement doesn't help")
    
    # Per-class comparison
    print(f"\n📈 Per-Class Comparison:")
    print("-"*70)
    
    for class_name in YOUR_CLASSES:
        base_class = results_baseline[results_baseline['true_class'] == class_name]
        enh_class = results_enhanced[results_enhanced['true_class'] == class_name]
        
        if len(base_class) > 0 and len(enh_class) > 0:
            base_acc = (base_class['correct'].sum() / len(base_class)) * 100
            enh_acc = (enh_class['correct'].sum() / len(enh_class)) * 100
            change = enh_acc - base_acc
            
            indicator = "✅" if change > 5 else ("❌" if change < -5 else "  ")
            
            print(f"  {class_name:12s}: {base_acc:5.1f}% → {enh_acc:5.1f}% ({indicator} {change:+5.1f}%)")
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print("="*70)
    print(f"\n📁 Enhanced images: {ENHANCED_DIR}/")
    print(f"📊 Baseline results: results_baseline.csv")
    print(f"📊 Enhanced results: results_enhanced.csv")


if __name__ == "__main__":
    main()