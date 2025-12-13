import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score
import pandas as pd

############################################
# Paths (based on your provided information)
############################################
TEST_DIR = r"D:\ece253\Vehicle-Type-Detection\dataset\raw_data"
DEHAZED_DIR = r"D:\ece253\Vehicle-Type-Detection\test_dehazed"
os.makedirs(DEHAZED_DIR, exist_ok=True)

############################################
# ---------- Dehazing Algorithm: DCP ----------
############################################
def get_dark_channel(I, window_size=15):
    min_channel = np.min(I, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark = cv2.erode(min_channel, kernel)
    return dark

def estimate_atmospheric_light(I, dark_channel):
    h, w = dark_channel.shape
    num_pixels = h * w
    n_search = int(max(num_pixels * 0.001, 1))
    dark_vec = dark_channel.reshape(num_pixels)
    img_vec = I.reshape(num_pixels, 3)
    indices = dark_vec.argsort()[-n_search:]
    A = np.mean(img_vec[indices], axis=0)
    return A

def estimate_transmission(I, A, omega=0.95, window_size=15):
    norm_I = I / A
    dark = get_dark_channel(norm_I, window_size=window_size)
    transmission = 1 - omega * dark
    return transmission

def recover_image(I, t, A, t0=0.1):
    t = np.clip(t, t0, 1)
    J = (I - A) / t[:, :, None] + A
    J = np.clip(J, 0, 255)
    return J.astype(np.uint8)

def dehaze_dcp(img):
    I = img.astype('float64')
    dark = get_dark_channel(I)
    A = estimate_atmospheric_light(I, dark)
    t = estimate_transmission(I, A)
    J = recover_image(I, t, A)
    return J

############################################
# ---------- Dehazing Algorithm: MSRcr ----------
############################################
def msr(img, sigma_list=[15, 80, 250], alpha=125, beta=46, gain=2.0, gamma=1.0):
    img = img.astype(np.float32) + 1.0
    img_retinex = np.zeros_like(img)

    # Multi-Scale Retinex
    for sigma in sigma_list:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        img_retinex += np.log(img) - np.log(blur + 1.0)
    img_retinex /= len(sigma_list)

    # Color Restoration
    color_restore = beta * (np.log(alpha * img) - np.log(np.sum(img, axis=2, keepdims=True)))
    msrcr = gain * img_retinex * color_restore

    # Normalize
    msrcr = cv2.normalize(msrcr, None, 0, 255, cv2.NORM_MINMAX)
    msrcr = np.clip(msrcr, 0, 255).astype(np.uint8)

    # Gamma correction
    msrcr = msrcr / 255.0
    msrcr = np.power(msrcr, gamma)
    msrcr = (msrcr * 255).astype(np.uint8)

    return msrcr

############################################
# ---------- Apply Dehazing and Save ----------
############################################
def process_dataset(mode="dcp"):
    """mode: 'dcp' or 'msr'"""
    print(f"\n🔧 Running dehazing algorithm: {mode.upper()} ...")
    for cls in os.listdir(TEST_DIR):
        src_dir = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(src_dir):
            continue

        dst_dir = os.path.join(DEHAZED_DIR, cls)
        os.makedirs(dst_dir, exist_ok=True)

        for fname in os.listdir(src_dir):
            path = os.path.join(src_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue

            if mode == "dcp":
                out = dehaze_dcp(img)
            else:
                out = msr(img)

            cv2.imwrite(os.path.join(dst_dir, fname), out)

    print("All dehazed images have been generated!\n")

############################################
# ---------- Model Loading ----------
############################################
# 17 classes used during model training (order must match training)
MODEL_CLASSES = [
    'Ambulance', 'Barge', 'Bicycle', 'Boat', 'Bus', 'Car',
    'Cart', 'Caterpillar', 'Helicopter', 'Limousine',
    'Motorcycle', 'Segway', 'Snowmobile', 'Tank', 'Taxi',
    'Truck', 'Van'
]

# Target 7 classes of interest (folder names should match or be lowercase versions)
TARGET_7 = ['bicycle', 'boat', 'bus', 'car', 'helicopter', 'motorcycle', 'truck']

# Convert MODEL_CLASSES to lowercase for easier comparison
MODEL_CLASSES_LOWER = [c.lower() for c in MODEL_CLASSES]

# Load models (your own paths)
model1 = load_model(r"D:/ece253/Vehicle-Type-Detection/saved_models/mobilenet2.h5")
model2 = load_model(r"D:/ece253/Vehicle-Type-Detection/saved_models/InceptionV3.h5")
models = [model1, model2]

############################################
# ---------- Classification Prediction (17 → 7 Mapping) ----------
############################################
def predict_image(models, img):
    """
    Takes models (with 17-class output) and returns the mapped
    target 7-class label or 'unknown', along with confidence.
    """
    img = cv2.resize(img, (224, 224))
    arr = img.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Ensemble: sum outputs from all models
    preds = sum(m.predict(arr)[0] for m in models)  # shape (17,)
    cls_id = int(np.argmax(preds))
    confidence = float(preds[cls_id] / preds.sum())

    # Original model class name (based on MODEL_CLASSES order)
    model_pred_class = MODEL_CLASSES_LOWER[cls_id]

    # Keep prediction if it belongs to the target 7 classes; otherwise mark as unknown
    if model_pred_class in TARGET_7:
        return model_pred_class, confidence
    else:
        return 'unknown', confidence

############################################
# Add per-class accuracy computation (for the 7 target classes)
############################################
def run_predictions(data_dir):
    y_true = []
    y_pred = []
    records = []

    # Statistics only for the target 7 classes
    class_correct = {c: 0 for c in TARGET_7}
    class_total = {c: 0 for c in TARGET_7}

    for cls in os.listdir(data_dir):
        folder = os.path.join(data_dir, cls)
        if not os.path.isdir(folder):
            continue

        # Ground-truth label (lowercase)
        true_label = cls.lower()

        # Only process folders belonging to the target 7 classes
        if true_label not in TARGET_7:
            continue

        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue

            pred, conf = predict_image(models, img)

            y_true.append(true_label)
            y_pred.append(pred)
            records.append([fname, true_label, pred, conf])

            class_total[true_label] += 1
            if pred == true_label:
                class_correct[true_label] += 1

    # Overall accuracy (only on the target 7 classes; 'unknown' counts as incorrect)
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    df = pd.DataFrame(records, columns=["filename", "true", "pred", "confidence"])

    # Per-class accuracy for the target 7 classes
    class_acc = {
        c: (class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0)
        for c in TARGET_7
    }

    return acc, class_acc, df

############################################
# ---------- Main Pipeline ----------
############################################
print("Starting baseline test (hazy images)...")
baseline_acc, baseline_class_acc, baseline_df = run_predictions(TEST_DIR)
print("baseline accuracy:", baseline_acc)

print("\n baseline per-class accuracy:")
for cls, acc in baseline_class_acc.items():
    print(f"  {cls}: {acc:.4f}")

print("\n Generating dehazed dataset (MSR)...")
# Note: set mode to 'msr' or 'dcp'; do not use invalid modes like 'dsp'
process_dataset(mode="msr")  # or process_dataset(mode="dcp")

print("Starting dehazed test...")
dehaze_acc, dehaze_class_acc, dehaze_df = run_predictions(DEHAZED_DIR)

print("\n dehazed accuracy:", dehaze_acc)
print("\n dehazed per-class accuracy:")
for cls, acc in dehaze_class_acc.items():
    print(f"  {cls}: {acc:.4f}")

baseline_df.to_csv("baseline_results.csv", index=False)
dehaze_df.to_csv("dehazed_results.csv", index=False)

print("\n=====================================")
print("Baseline Accuracy :", baseline_acc)
print("Dehazed Accuracy :", dehaze_acc)
print("Accuracy Improvement :", dehaze_acc - baseline_acc)
print("=====================================\n")
