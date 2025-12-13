import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score
import pandas as pd

############################################
# 路径（根据你提供的信息）
############################################
TEST_DIR = r"D:\ece253\Vehicle-Type-Detection\dataset\train"
DEHAZED_DIR = r"D:\ece253\Vehicle-Type-Detection\test_dehazed"
os.makedirs(DEHAZED_DIR, exist_ok=True)

############################################
# ---------- 去雾算法：DCP ----------
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
# ---------- 去雾算法：MSRcr ----------
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
# ---------- 去雾处理并保存 ----------
############################################
def process_dataset(mode="dcp"):
    """mode: 'dcp' or 'msr'"""
    print(f"\n🔧 正在执行去雾算法: {mode.upper()} ...")
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

    print("✅ 去雾图像已全部生成！\n")

############################################
# ---------- 模型加载 ----------
############################################
# 这是模型训练时的 17 类（顺序必须和模型训练时一致）
MODEL_CLASSES = [
    'Ambulance', 'Barge', 'Bicycle', 'Boat', 'Bus', 'Car',
    'Cart', 'Caterpillar', 'Helicopter', 'Limousine',
    'Motorcycle', 'Segway', 'Snowmobile', 'Tank', 'Taxi',
    'Truck', 'Van'
]

# 你关心的 7 类（目标类，文件夹名应与这些一致或为小写版本）
TARGET_7 = ['bicycle', 'boat', 'bus', 'car', 'helicopter', 'motorcycle', 'truck']

# 把 MODEL_CLASSES 映射为小写，方便比较
MODEL_CLASSES_LOWER = [c.lower() for c in MODEL_CLASSES]

# 加载模型（你自己的路径）
model1 = load_model(r"D:/ece253/Vehicle-Type-Detection/saved_models/mobilenet2.h5")
model2 = load_model(r"D:/ece253/Vehicle-Type-Detection/saved_models/InceptionV3.h5")
models = [model1, model2]

############################################
# ---------- 分类预测（含 17->7 映射）----------
############################################
def predict_image(models, img):
    """
    接收模型（输出应为 17 维），返回映射后的目标 7 类标签或 'unknown'，以及 confidence。
    """
    img = cv2.resize(img, (224, 224))
    arr = img.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    # ensemble: 对所有模型输出求和
    preds = sum(m.predict(arr)[0] for m in models)  # shape (17,)
    cls_id = int(np.argmax(preds))
    confidence = float(preds[cls_id] / preds.sum())

    # 原始模型类别名（按 MODEL_CLASSES 顺序）
    model_pred_class = MODEL_CLASSES_LOWER[cls_id]  # e.g. 'bicycle' or 'ambulance'

    # 如果该模型预测属于你关心的 7 类之一，保留映射；否则标记为 unknown
    if model_pred_class in TARGET_7:
        return model_pred_class, confidence
    else:
        return 'unknown', confidence

############################################
# 加入每类 accuracy 计算（针对 7 类的统计）
############################################
def run_predictions(data_dir):
    y_true = []
    y_pred = []
    records = []

    # 仅针对目标 7 类统计
    class_correct = {c: 0 for c in TARGET_7}
    class_total = {c: 0 for c in TARGET_7}

    for cls in os.listdir(data_dir):
        folder = os.path.join(data_dir, cls)
        if not os.path.isdir(folder):
            continue

        # 真实标签（小写）
        true_label = cls.lower()
        # 只处理我们关注的 7 类文件夹（如果数据集中包含其它类则跳过）
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

    # overall accuracy: 只计算目标 7 类样本（把 'unknown' 视为错误）
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    df = pd.DataFrame(records, columns=["filename", "true", "pred", "confidence"])

    ### 计算每类 accuracy（针对 7 类）
    class_acc = {
        c: (class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0)
        for c in TARGET_7
    }

    return acc, class_acc, df

############################################
# ---------- 主流程 ----------
############################################
print("🎯 开始 baseline 测试（雾化图）...")
baseline_acc, baseline_class_acc, baseline_df = run_predictions(TEST_DIR)
print("📌 baseline accuracy:", baseline_acc)

print("\n📌 baseline per-class accuracy:")
for cls, acc in baseline_class_acc.items():
    print(f"  {cls}: {acc:.4f}")

print("\n🎯 开始生成去雾数据集（MSR）...")
# 修正：把 mode 设为 'msr' 或 'dcp'，不要用不存在的 'dsp'
process_dataset(mode="msr")  # 或 process_dataset(mode="dcp")

print("🎯 开始 dehazed 测试...")
dehaze_acc, dehaze_class_acc, dehaze_df = run_predictions(DEHAZED_DIR)

print("\n📌 dehazed accuracy:", dehaze_acc)
print("\n📌 dehazed per-class accuracy:")
for cls, acc in dehaze_class_acc.items():
    print(f"  {cls}: {acc:.4f}")

baseline_df.to_csv("baseline_results.csv", index=False)
dehaze_df.to_csv("dehazed_results.csv", index=False)

print("\n=====================================")
print("Baseline Accuracy :", baseline_acc)
print("Dehazed Accuracy :", dehaze_acc)
print("Accuracy Improvement :", dehaze_acc - baseline_acc)
print("=====================================\n")
