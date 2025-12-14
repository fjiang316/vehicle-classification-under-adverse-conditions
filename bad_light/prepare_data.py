#!/usr/bin/env python3
"""
Data Preparation Script
Splits dataset into train/val/test sets automatically
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

# Source directory (where your 280 images per class are)
SOURCE_DIR = "bad_light/"  # Change this to your actual data directory

# Output directories
TRAIN_DIR = "train/"
VAL_DIR = "val/"
TEST_DIR = "test/"

# Split ratios
TRAIN_RATIO = 0.70  # 70% for training (196 images)
VAL_RATIO = 0.15    # 15% for validation (42 images)
TEST_RATIO = 0.15   # 15% for testing (42 images)

# Classes
YOUR_CLASSES = ['Bicycle', 'Boat', 'Bus', 'Car', 'Helicopter', 'Motorcycle', 'Truck']

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# DATA SPLITTING FUNCTION
# ============================================================================

def split_dataset(source_dir, train_dir, val_dir, test_dir, 
                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                  classes=None, seed=42):
    """
    Split dataset into train/val/test sets
    
    Args:
        source_dir: Source directory containing class folders
        train_dir: Output directory for training data
        val_dir: Output directory for validation data
        test_dir: Output directory for test data
        train_ratio: Proportion for training (default 0.7)
        val_ratio: Proportion for validation (default 0.15)
        test_ratio: Proportion for test (default 0.15)
        classes: List of class names (if None, use all subdirectories)
        seed: Random seed for reproducibility
    """
    
    # Set random seed
    random.seed(seed)
    
    # Create output directories
    for output_dir in [train_dir, val_dir, test_dir]:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get classes
    if classes is None:
        classes = [d for d in os.listdir(source_dir) 
                  if os.path.isdir(os.path.join(source_dir, d))]
    
    print("\n" + "="*70)
    print("SPLITTING DATASET")
    print("="*70)
    print(f"Source: {source_dir}")
    print(f"Train: {train_ratio*100:.0f}% → {train_dir}")
    print(f"Val:   {val_ratio*100:.0f}% → {val_dir}")
    print(f"Test:  {test_ratio*100:.0f}% → {test_dir}")
    print(f"Classes: {classes}")
    print(f"Random seed: {seed}")
    
    # Statistics
    total_stats = {
        'total': 0,
        'train': 0,
        'val': 0,
        'test': 0
    }
    
    class_stats = []
    
    # Process each class
    for class_name in classes:
        class_source_dir = os.path.join(source_dir, class_name)
        
        if not os.path.exists(class_source_dir):
            print(f"\n⚠️  Warning: {class_name}/ not found in {source_dir}, skipping...")
            continue
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        all_images = [f for f in os.listdir(class_source_dir) 
                     if any(f.endswith(ext) for ext in image_extensions)]
        
        if len(all_images) == 0:
            print(f"\n⚠️  Warning: No images found in {class_name}/, skipping...")
            continue
        
        # Shuffle images
        random.shuffle(all_images)
        
        # Calculate split sizes
        total_images = len(all_images)
        train_size = int(total_images * train_ratio)
        val_size = int(total_images * val_ratio)
        test_size = total_images - train_size - val_size  # Remainder goes to test
        
        # Split images
        train_images = all_images[:train_size]
        val_images = all_images[train_size:train_size + val_size]
        test_images = all_images[train_size + val_size:]
        
        # Create class directories
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        
        Path(train_class_dir).mkdir(parents=True, exist_ok=True)
        Path(val_class_dir).mkdir(parents=True, exist_ok=True)
        Path(test_class_dir).mkdir(parents=True, exist_ok=True)
        
        # Copy files
        print(f"\n{class_name}:")
        print(f"  Total: {total_images} images")
        print(f"  Train: {len(train_images)} images")
        print(f"  Val:   {len(val_images)} images")
        print(f"  Test:  {len(test_images)} images")
        
        # Copy training images
        for img in train_images:
            src = os.path.join(class_source_dir, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copy2(src, dst)
        
        # Copy validation images
        for img in val_images:
            src = os.path.join(class_source_dir, img)
            dst = os.path.join(val_class_dir, img)
            shutil.copy2(src, dst)
        
        # Copy test images
        for img in test_images:
            src = os.path.join(class_source_dir, img)
            dst = os.path.join(test_class_dir, img)
            shutil.copy2(src, dst)
        
        # Update statistics
        total_stats['total'] += total_images
        total_stats['train'] += len(train_images)
        total_stats['val'] += len(val_images)
        total_stats['test'] += len(test_images)
        
        class_stats.append({
            'class': class_name,
            'total': total_images,
            'train': len(train_images),
            'val': len(val_images),
            'test': len(test_images)
        })
    
    # Print summary
    print("\n" + "="*70)
    print("SPLIT SUMMARY")
    print("="*70)
    print(f"\nTotal images: {total_stats['total']}")
    print(f"  Train: {total_stats['train']} ({total_stats['train']/total_stats['total']*100:.1f}%)")
    print(f"  Val:   {total_stats['val']} ({total_stats['val']/total_stats['total']*100:.1f}%)")
    print(f"  Test:  {total_stats['test']} ({total_stats['test']/total_stats['total']*100:.1f}%)")
    
    print("\nPer-class breakdown:")
    print(f"{'Class':<15} {'Total':<10} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-"*70)
    for stats in class_stats:
        print(f"{stats['class']:<15} {stats['total']:<10} {stats['train']:<10} "
              f"{stats['val']:<10} {stats['test']:<10}")
    
    print("\n" + "="*70)
    print("✅ DATASET SPLIT COMPLETE!")
    print("="*70)
    print(f"\n📁 Train data: {train_dir}")
    print(f"📁 Val data:   {val_dir}")
    print(f"📁 Test data:  {test_dir}")


# ============================================================================
# VERIFICATION FUNCTION
# ============================================================================

def verify_split(train_dir, val_dir, test_dir, classes):
    """
    Verify that the split was successful and no data was lost
    """
    print("\n" + "="*70)
    print("VERIFYING SPLIT")
    print("="*70)
    
    all_good = True
    
    for class_name in classes:
        train_count = len([f for f in os.listdir(os.path.join(train_dir, class_name)) 
                          if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))])
        val_count = len([f for f in os.listdir(os.path.join(val_dir, class_name)) 
                        if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))])
        test_count = len([f for f in os.listdir(os.path.join(test_dir, class_name)) 
                         if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))])
        
        total = train_count + val_count + test_count
        
        print(f"{class_name}: {train_count} + {val_count} + {test_count} = {total}")
        
        if train_count == 0 or val_count == 0 or test_count == 0:
            print(f"  ⚠️  Warning: One of the splits is empty!")
            all_good = False
    
    if all_good:
        print("\n✅ All splits verified successfully!")
    else:
        print("\n⚠️  Some issues detected. Please check the warnings above.")
    
    return all_good


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test')
    parser.add_argument('--source', type=str, default=SOURCE_DIR,
                       help='Source directory with original data')
    parser.add_argument('--train-ratio', type=float, default=TRAIN_RATIO,
                       help='Training set ratio (default: 0.70)')
    parser.add_argument('--val-ratio', type=float, default=VAL_RATIO,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=TEST_RATIO,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='Random seed (default: 42)')
    parser.add_argument('--verify', action='store_true',
                       help='Only verify existing split')
    
    args = parser.parse_args()
    
    # Verify ratios sum to 1.0
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"❌ Error: Ratios must sum to 1.0 (currently {total_ratio})")
        return
    
    if args.verify:
        # Only verify
        verify_split(TRAIN_DIR, VAL_DIR, TEST_DIR, YOUR_CLASSES)
    else:
        # Check if source directory exists
        if not os.path.exists(args.source):
            print(f"❌ Error: Source directory not found: {args.source}")
            print("\nPlease update SOURCE_DIR in the script or use --source argument")
            print("Expected structure:")
            print("  dataset/")
            print("    bicycle/")
            print("    boat/")
            print("    bus/")
            print("    ...")
            return
        
        # Check if output directories already exist
        if any(os.path.exists(d) for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]):
            response = input("\n⚠️  Output directories already exist. Overwrite? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted.")
                return
            
            # Remove existing directories
            for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
                if os.path.exists(d):
                    shutil.rmtree(d)
                    print(f"Removed {d}")
        
        # Split dataset
        split_dataset(
            source_dir=args.source,
            train_dir=TRAIN_DIR,
            val_dir=VAL_DIR,
            test_dir=TEST_DIR,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            classes=YOUR_CLASSES,
            seed=args.seed
        )
        
        # Verify split
        verify_split(TRAIN_DIR, VAL_DIR, TEST_DIR, YOUR_CLASSES)


if __name__ == "__main__":
    main()