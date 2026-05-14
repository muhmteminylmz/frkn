# =========================================================
# G-HS + OBL + DİNAMİK PAR/BW ile CNN HİPERPARAMETRE OPTİMİZASYONU
# KAGGLE WASTE CLASSIFICATION (Organic vs Recyclable)
# KLASİK CNN (Baseline) vs G-HS-CNN v2 (Önerilen: SE + Residual + Label Smoothing)
# ✅ GPU OPTIMIZED + tf.data PIPELINE + MIXED PRECISION + SEÇİLEBİLİR VERİ BOYUTU
# =========================================================

import os
import gc
import json
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass, asdict

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# TensorFlow importundan ÖNCE GPU çalışma modu seçimi:
# FRKN_GPU_MODE: nvidia | auto | index | cpu
# FRKN_GPU_INDEX: index modunda kullanılacak GPU indeks değeri (örn. 0)
GPU_MODE = os.environ.get("FRKN_GPU_MODE", "nvidia").strip().lower()
GPU_INDEX = os.environ.get("FRKN_GPU_INDEX", "0").strip()
VALID_GPU_MODES = {"nvidia", "auto", "index", "cpu"}
if GPU_MODE not in VALID_GPU_MODES:
    print(f"⚠️ Geçersiz FRKN_GPU_MODE={GPU_MODE!r}. 'auto' kullanılacak.")
    GPU_MODE = "auto"
if GPU_MODE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# =========================================================
# HIZLI TEST MODU
# QUICK_TEST = True  → küçük veri alt kümesi (hızlı doğrulama)
# QUICK_TEST = False → tüm 22k veri (tam çalıştırma)
# =========================================================
QUICK_TEST = False

# =========================================================
# VERİ BOYUTU (opsiyonel – DATASET_SIZE)
# None  → QUICK_TEST/FULL moduna göre otomatik (2000 veya tüm veri)
# int   → Tam veri setinden seçilecek toplam görüntü sayısı
#          Örnek: DATASET_SIZE = 5000  → 3500 eğitim + 750 doğrulama + 750 test
#          Geçerli aralık: 500 – 22500
# Tüm veri havuzu TRAIN_DIR + TEST_DIR üzerinden birleştirilip 70/15/15 bölünür.
# =========================================================
DATASET_SIZE = None   # Örnek: 5000, 10000, None (otomatik)

# DATASET_SIZE geçerlilik kontrolü
if DATASET_SIZE is not None and not (500 <= DATASET_SIZE <= 22500):
    raise ValueError(f"DATASET_SIZE {DATASET_SIZE} geçersiz. Geçerli aralık: 500 – 22500")

if QUICK_TEST and DATASET_SIZE is not None:
    raise ValueError(
        "QUICK_TEST=True ile DATASET_SIZE birlikte kullanılamaz. "
        "Hızlı test için DATASET_SIZE=None, daha büyük veri koşusu için QUICK_TEST=False kullanın."
    )

# =========================================================
# GPU SETUP
# =========================================================
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n✅ GPU AKTIF: {len(gpus)} cihaz bulundu")
    for gpu in gpus:
        print(f"   → {gpu}")

    try:
        selected_gpu = None
        fallback_reason = None
        if GPU_MODE == "index":
            if GPU_INDEX.isdigit():
                gpu_index = int(GPU_INDEX)
                if 0 <= gpu_index < len(gpus):
                    selected_gpu = gpus[gpu_index]
                else:
                    fallback_reason = f"FRKN_GPU_INDEX={gpu_index} geçersiz (geçerli aralık: 0-{len(gpus)-1})"
                    print(f"⚠️ {fallback_reason}, tüm görünür GPU'lar kullanılacak")
            else:
                fallback_reason = f"FRKN_GPU_INDEX={GPU_INDEX!r} sayısal değil"
                print(f"⚠️ {fallback_reason}, tüm görünür GPU'lar kullanılacak")
        elif GPU_MODE == "nvidia":
            selected_gpu = next((gpu for gpu in gpus if "NVIDIA" in gpu.name.upper()), None)
            if selected_gpu is None:
                fallback_reason = "NVIDIA GPU bulunamadı"
                print(f"⚠️ {fallback_reason}, tüm görünür GPU'lar kullanılacak")
        # auto veya fallback durumunda selected_gpu None kalır ve tüm görünür GPU'lar kullanılır

        if selected_gpu is not None:
            tf.config.set_visible_devices(selected_gpu, 'GPU')
            print(f"🎯 Seçilen GPU modu: {GPU_MODE} → {selected_gpu.name}")
        else:
            if fallback_reason:
                print(f"ℹ️ Fallback modu etkin: tüm görünür GPU'lar kullanılacak")
            else:
                print(f"ℹ️ GPU modu: {GPU_MODE} → tüm görünür GPU'lar kullanılacak")
    except RuntimeError as e:
        print(f"GPU seçim hatası: {e}")

    try:
        for gpu in tf.config.get_visible_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU Hatası: {e}")

    # ⚡ Mixed Precision:
    # NVIDIA GPU'larda öncelik float16 (GTX/RTX uyumu için daha güvenli),
    # gerekirse bfloat16 denenir; DirectML'de float32 kullanılır.
    # DirectML cihazları TF içinde "DML" veya "PluggableDevice" adıyla raporlanır.
    is_directml = any("DML" in gpu.name.upper() or "PLUGGABLE" in gpu.name.upper()
                      for gpu in tf.config.get_visible_devices('GPU'))
    if is_directml:
        print("ℹ️  DirectML cihazı algılandı – Mixed Precision atlandı, float32 kullanılıyor")
    else:
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("⚡ Mixed Precision (float16) AKTİF")
        except Exception as mp_err_fp16:
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
                print("⚡ Mixed Precision (bfloat16) AKTİF (float16 desteklenmedi)")
            except Exception as mp_err_bf16:
                print(
                    "⚠️ Mixed Precision etkinleştirilemedi, float32 kullanılıyor: "
                    f"float16={str(mp_err_fp16)} | bfloat16={str(mp_err_bf16)}"
                )
else:
    print("\n⚠️ GPU bulunamadı, CPU ile devam edilecek")

print()

# =========================================================
# 0. SABİT TOHUM
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(42)


# =========================================================
# 1. HİPERPARAMETRE YAPISI
# =========================================================
@dataclass
class HyperParams:
    filters: int        # İlk conv bloğundaki filtre sayısı (sonraki bloklar 2x artar)
    kernel_size: int    # Konvolüsyon çekirdeği boyutu
    num_blocks: int     # Conv blok sayısı (derinlik)
    dropout: float      # Dropout oranı
    learning_rate: float
    batch_size: int
    dense_units: int    # Fully-connected katman nöron sayısı


# =========================================================
# 2. VERİ YÜKLEME (tf.data Pipeline – HIZLI)
# =========================================================
# tf.data + AUTOTUNE prefetch: ImageDataGenerator'a göre ~2x daha hızlı
TRAIN_DIR = "dataset/train"
TEST_DIR  = "dataset/test"
AUTOTUNE     = tf.data.AUTOTUNE
MAX_FILTERS  = 512  # Her blokta 2x artan filtre sayısı için üst sınır
MIN_SPATIAL_DROPOUT = 0.05
MAX_SPATIAL_DROPOUT = 0.35
RF_N_ESTIMATORS = 100  # Reduced from 300 to improve speed while preserving benchmark quality
CLASS_LABELS = ["Organic", "Recyclable"]

# Dataset cache (batch_size → (train_ds, val_ds, test_ds))
_gen_cache = {}

# Hızlı test modu ayarları
IMG_SIZE = (64, 64) if QUICK_TEST else (128, 128)
ML_IMG_SIZE = (32, 32)
QUICK_TEST_TOTAL = 3000  # QUICK_TEST=True iken toplam örnek sayısı (70/15/15 uygulanır)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
_dataset_file_count_cache = None


def count_image_files(root_dir):
    """Verilen dizin altında desteklenen uzantılardaki görüntü dosyalarını say."""
    total = 0
    for dirpath, _, filenames in os.walk(root_dir):
        total += sum(name.lower().endswith(VALID_IMAGE_EXTENSIONS) for name in filenames)
    return total


def get_total_image_count():
    """TRAIN_DIR + TEST_DIR toplam görüntü sayısını cache'li döndür."""
    global _dataset_file_count_cache
    if _dataset_file_count_cache is None:
        _dataset_file_count_cache = count_image_files(TRAIN_DIR) + count_image_files(TEST_DIR)
    return _dataset_file_count_cache


def compute_split_sizes(total_count):
    """Toplam örnek sayısını sabit 70/15/15 oranına böl."""
    if total_count < 3:
        raise ValueError("70/15/15 bölmesi için en az 3 görüntü gerekli.")
    train_n = int(total_count * TRAIN_RATIO)
    val_n = int(total_count * VAL_RATIO)
    test_n = total_count - train_n - val_n
    return train_n, val_n, test_n


def create_datasets(batch_size, use_subset=QUICK_TEST):
    """tf.data pipeline ile veri yükle (batch_size ve veri boyutuna göre cache'li)"""
    cache_key = (batch_size, DATASET_SIZE if DATASET_SIZE is not None else ("quick" if use_subset else "full"))
    if cache_key in _gen_cache:
        return _gen_cache[cache_key]

    print(f"📦 Dataset pipeline hazırlanıyor (Batch={batch_size}, use_subset={use_subset})...")

    common = dict(
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode='binary',
        seed=42,
    )

    # 70/15/15 zorunlu bölme için TRAIN_DIR + TEST_DIR birleştirilir.
    train_pool_ds = tf.keras.utils.image_dataset_from_directory(TRAIN_DIR, shuffle=True, **common)
    test_pool_ds = tf.keras.utils.image_dataset_from_directory(TEST_DIR, shuffle=True, **common)
    if train_pool_ds.class_names != test_pool_ds.class_names:
        raise ValueError("TRAIN_DIR ve TEST_DIR class isimleri farklı; 70/15/15 birleştirme yapılamadı.")

    all_ds = train_pool_ds.concatenate(test_pool_ds).unbatch()
    total_images = get_total_image_count()

    if DATASET_SIZE is not None:
        target_total = min(DATASET_SIZE, total_images)
    elif use_subset:
        target_total = min(QUICK_TEST_TOTAL, total_images)
    else:
        target_total = total_images

    train_n, val_n, test_n = compute_split_sizes(target_total)
    all_ds = all_ds.shuffle(total_images, seed=42, reshuffle_each_iteration=False).take(target_total)

    train_ds = all_ds.take(train_n).batch(batch_size)
    remain_ds = all_ds.skip(train_n)
    val_ds = remain_ds.take(val_n).batch(batch_size)
    test_ds = remain_ds.skip(val_n).take(test_n).batch(batch_size)

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)
    test_ds  = test_ds.prefetch(AUTOTUNE)

    result = (train_ds, val_ds, test_ds)
    _gen_cache[cache_key] = result
    return result


def dataset_to_numpy(ds):
    """tf.data dataset'i numpy tensörlerine dönüştür."""
    x_parts, y_parts = [], []
    for xb, yb in ds:
        x_parts.append(xb.numpy())
        y_parts.append(flatten_binary_labels(yb.numpy()))
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0).astype(np.int32)
    return x, y


def dataset_to_numpy_small(ds):
    """Sklearn benchmark için görüntüleri küçültüp numpy tensörlerine dönüştür."""
    x_parts, y_parts = [], []
    for xb, yb in ds:
        xb_small = tf.image.resize(xb, ML_IMG_SIZE).numpy()
        x_parts.append(xb_small)
        y_parts.append(flatten_binary_labels(yb.numpy()))
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0).astype(np.int32)
    return x, y


def flatten_binary_labels(y):
    """Binary etiketleri güvenli biçimde 1D vektöre indirger."""
    y = np.asarray(y)
    if y.ndim == 1:
        return y
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0]
    raise ValueError(f"Unexpected label shape: {y.shape}. Expected: (N,) or (N,1)")


def flatten_and_normalize_images(x):
    """Görüntüleri sklearn modelleri için düzleştir ve normalize et."""
    return (x.reshape(x.shape[0], -1) / 255.0).astype(np.float32)


def compute_classification_metrics(y_true, y_prob):
    """Binary sınıflandırma metriklerini hesapla."""
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(np.int32)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def evaluate_keras_model(model, test_ds):
    """Keras modeli için olasılık çıktıları ve metrikleri üret."""
    y_true_parts, y_prob_parts = [], []
    for xb, yb in test_ds:
        probs = model.predict(xb, verbose=0).reshape(-1)
        y_prob_parts.append(probs)
        y_true_parts.append(flatten_binary_labels(yb.numpy()).astype(np.int32))

    y_true = np.concatenate(y_true_parts, axis=0)
    y_prob = np.concatenate(y_prob_parts, axis=0)
    metrics = compute_classification_metrics(y_true, y_prob)
    return y_true, y_prob, metrics


# =========================================================
# 3. G-HS CNN MODELİ (Önerilen – G-HS tarafından optimize edilir)
# =========================================================
def _se_block(x, filters, ratio=8):
    """Squeeze-and-Excitation kanal dikkat bloğu.
    Her conv bloğundan sonra kanal önemini dinamik olarak ağırlıklandırır.
    Parametre sayısı: filters*(filters//ratio) + (filters//ratio)*filters ≈ 2*(filters²/ratio)."""
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    se = tf.keras.layers.Dense(max(filters // ratio, 2), activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    se = tf.keras.layers.Reshape((1, 1, filters))(se)
    return tf.keras.layers.Multiply()([x, se])


def build_cnn_model(hp: HyperParams) -> tf.keras.Model:
    """
    Dinamik derinlikte CNN modeli (v2: SE kanal dikkat + Residual bağlantı).
    G-HS, filters/kernel_size/num_blocks/dropout/lr/batch/dense parametrelerini optimize eder.

    Mimari:
      Rescaling → Augmentation →
      [Conv→BN→ReLU → Conv→BN→SE → Residual Add → ReLU → MaxPool] × num_blocks
      → GlobalAveragePooling → Dense(dense_units) → Dropout → Dense(1, sigmoid)

    Geliştirmeler (v2):
      • SE blok: kanal dikkatini öğrenerek önemli özellikleri güçlendirir
      • Residual bağlantı: gradyan akışını iyileştirir, derin ağlarda eğitimi kolaylaştırır
      • Label smoothing (0.05): aşırı güven önler, genelleşmeyi artırır
    """
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Normalizasyon ve veri artırma (eğitimde aktif, evaluate/predict'te pasif)
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    # Random* augmentasyon katmanları bazı ortamlarda bfloat16 ile hata verebildiğinden
    # augmentasyon girişini float32'de tut.
    x = tf.keras.layers.Lambda(lambda t: tf.cast(t, tf.float32))(x)
    x = tf.keras.layers.RandomFlip("horizontal", dtype='float32')(x)
    x = tf.keras.layers.RandomRotation(0.15, dtype='float32')(x)
    x = tf.keras.layers.RandomZoom(0.15, dtype='float32')(x)
    x = tf.keras.layers.RandomContrast(0.1, dtype='float32')(x)

    # Konvolüsyon blokları: Residual + SE dikkat – her blokta filtre sayısı 2 katına çıkar
    f = hp.filters
    for _ in range(hp.num_blocks):
        shortcut = x

        x = tf.keras.layers.Conv2D(f, (hp.kernel_size, hp.kernel_size),
                                   padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(f, (hp.kernel_size, hp.kernel_size),
                                   padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # SE kanal dikkat: hangi kanalların önemli olduğunu öğren
        x = _se_block(x, f)

        # Residual bağlantı: kanal sayısı değişmişse 1×1 projeksiyon uygula
        if shortcut.shape[-1] != f:
            shortcut = tf.keras.layers.Conv2D(f, 1, padding='same',
                                              use_bias=False)(shortcut)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.MaxPooling2D(2, 2)(x)
        spatial_dropout_rate = float(np.clip(hp.dropout * 0.5, MIN_SPATIAL_DROPOUT, MAX_SPATIAL_DROPOUT))
        x = tf.keras.layers.SpatialDropout2D(spatial_dropout_rate)(x)
        f = min(f * 2, MAX_FILTERS)   # Modül düzeyinde sabit ile sınırla

    # Sınıflandırıcı kafası
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(hp.dense_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(hp.dropout)(x)
    # Mixed precision ile uyumluluk için çıktı katmanı float32
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    return model


# =========================================================
# 4. KLASİK CNN MODELİ (Baseline – sabit hiperparametreler)
# =========================================================
def build_cnn_baseline() -> tf.keras.Model:
    """
    Standart sabit-parametreli CNN baseline.
    G-HS-CNN ile karşılaştırma için referans noktası.
    Mimari: 3 conv blok (32→64→128 filtre), Flatten + Dense.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        tf.keras.layers.Lambda(lambda t: tf.cast(t, tf.float32)),
        tf.keras.layers.RandomFlip("horizontal", dtype='float32'),
        tf.keras.layers.RandomRotation(0.1, dtype='float32'),
        tf.keras.layers.RandomZoom(0.1, dtype='float32'),
        # Blok 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Blok 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Blok 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Sınıflandırıcı
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# =========================================================
# 5. FITNESS FONKSİYONU
# =========================================================
def evaluate_fitness(hp: HyperParams, epochs: int = 3) -> float:
    """
    CNN modeli için fitness hesapla (G-HS optimizasyon döngüsünde kullanılır).
    Düşük val_loss → daha iyi hiperparametre seti.
    """
    try:
        tf.keras.backend.clear_session()

        train_ds, val_ds, _ = create_datasets(hp.batch_size, use_subset=True)
        model = build_cnn_model(hp)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=0,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=2,
                    restore_best_weights=True,
                    verbose=0
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=1,
                    min_lr=1e-6,
                    verbose=0
                ),
            ]
        )

        val_loss = min(history.history["val_loss"])

        del model
        gc.collect()

        return val_loss

    except Exception as e:
        print(f"❌ Fitness hatası: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')


# =========================================================
# 6. PARAMETRE SINIRLARI
# GTX 1650 (4 GB VRAM) için filtre ve batch üst sınırları düşürüldü.
# Collab'da daha geniş arama alanı istenirse üst değerleri yükseltebilirsiniz.
# =========================================================
BOUNDS = {
    "filters":       (16,    64),   # GTX 1650: 128 filtre OOM riskini artırır
    "kernel_size":   (3,     5),    # 2 bilinçli olarak çıkarıldı
    "num_blocks":    (2,     4),
    "dropout":       (0.1,   0.5),
    "learning_rate": (1e-4,  1e-2),
    "batch_size":    (8,     32),   # GTX 1650 4 GB için max 32 (8 düşük-VRAM güvenli seçenek)
    "dense_units":   (64,    256),  # GTX 1650: 512 dense birim yerine 256
}

CHOICES = {
    "filters":     [16, 32, 48, 64],
    "kernel_size": [3, 5],          # 2 bilinçli olarak arama uzayından çıkarıldı
    "num_blocks":  [2, 3, 4],
    "batch_size":  [8, 16, 32],
    "dense_units": [64, 128, 256],  # GTX 1650: 512 çıkarıldı
}

CONTINUOUS_PARAMS = {"dropout", "learning_rate"}
DISCRETE_PARAMS   = {"filters", "kernel_size", "num_blocks", "batch_size", "dense_units"}


def get_nearest_choice(val, choices):
    """En yakın geçerli değeri bul"""
    return min(choices, key=lambda x: abs(x - val))


def random_hyperparams():
    """Rastgele hiperparametreler üret"""
    return HyperParams(
        filters=random.choice(CHOICES["filters"]),
        kernel_size=random.choice(CHOICES["kernel_size"]),
        num_blocks=random.choice(CHOICES["num_blocks"]),
        dropout=round(random.uniform(0.1, 0.5), 4),
        learning_rate=10 ** random.uniform(-4, -2),  # log-scale: 0.0001 – 0.01
        batch_size=random.choice(CHOICES["batch_size"]),
        dense_units=random.choice(CHOICES["dense_units"]),
    )


# =========================================================
# 7. G-HS ALGORİTMASI (OBL + DİNAMİK PAR/BW)
# =========================================================
# Literatürden bilinen iyi başlangıç noktası – ilk harmoni olarak HM'ye eklenir.
# Bu sayede G-HS, az HMS/NI ile bile makul bir yerden başlar.
ELITE_SEED = HyperParams(
    filters=32, kernel_size=3, num_blocks=3,
    dropout=0.3, learning_rate=1e-3,
    batch_size=32, dense_units=128,
)


def ghs_optimize(
    HMS=10,
    NI=30,
    HMCR=0.85,
    PAR_min=0.30,
    PAR_max=0.90,
    BW_min=0.001,
    BW_max=0.1,
    epochs_optimize=3
):
    """G-HS ile CNN hiperparametrelerini optimize et"""

    HM = []
    convergence = []

    print("=" * 70)
    print(">>> HM BAŞLATILIYOR (Harmony Memory Initialize)")
    print("=" * 70)

    # 1. HM başlatması (ilk harmoni: elite seed, kalanlar rastgele)
    for i in range(HMS):
        hp = ELITE_SEED if i == 0 else random_hyperparams()
        fitness = evaluate_fitness(hp, epochs=epochs_optimize)
        HM.append({"hp": hp, "fitness": fitness})
        prefix = "⭐ Elite" if i == 0 else "✓ Init  "
        print(
            f"{prefix} {i+1}/{HMS}: "
            f"filters={hp.filters}, k={hp.kernel_size}, blocks={hp.num_blocks}, "
            f"dropout={hp.dropout:.3f}, lr={hp.learning_rate:.2e}, "
            f"batch={hp.batch_size}, dense={hp.dense_units} "
            f"→ Loss={fitness:.6f}"
        )

    HM = sorted(HM, key=lambda x: x["fitness"])
    best_ever = HM[0]["fitness"]

    print("\n" + "=" * 70)
    print(">>> OPTİMİZASYON BAŞLADI")
    print("=" * 70 + "\n")

    # 2. İterasyon döngüsü
    for t in range(1, NI + 1):

        PAR_t = PAR_min + ((PAR_max - PAR_min) / NI) * t
        BW_t  = BW_max * np.exp(np.log(BW_min / BW_max) * (t / NI))

        new_params = {}

        for param in ["filters", "kernel_size", "num_blocks",
                      "dropout", "learning_rate", "batch_size", "dense_units"]:

            low, high = BOUNDS[param]
            rand1 = random.random()

            # HMCR: Hafızadan seçim
            if rand1 <= HMCR:
                value = getattr(random.choice(HM)["hp"], param)

                # PAR: Dinamik ince ayar
                if random.random() <= PAR_t:
                    r = random.uniform(-1, 1)
                    value = float(max(low, min(high, value + r * BW_t * (high - low))))

                if param in DISCRETE_PARAMS:
                    value = get_nearest_choice(value, CHOICES[param])

            # OBL: Opposition-Based Learning
            else:
                if param in DISCRETE_PARAMS:
                    val_rand = random.choice(CHOICES[param])
                    val_obl  = get_nearest_choice(low + high - val_rand, CHOICES[param])
                else:
                    val_rand = random.uniform(low, high)
                    val_obl  = float(max(low, min(high, low + high - val_rand)))

                base = asdict(HM[0]["hp"])

                def make_hp(override_val, _base=base, _param=param):
                    p = {**_base, _param: override_val}
                    return HyperParams(**{
                        k: (float(v) if k in CONTINUOUS_PARAMS else int(v))
                        for k, v in p.items()
                    })

                print(
                    f"   [OBL - {param}] Rand({val_rand:.4g}) vs Obl({val_obl:.4g})...",
                    end=" ", flush=True
                )
                fit_rand = evaluate_fitness(make_hp(val_rand), epochs=2)
                fit_obl  = evaluate_fitness(make_hp(val_obl),  epochs=2)

                if fit_rand <= fit_obl:
                    value = val_rand
                    # Eşitlik durumunda rastgele değer tercih edilir (basitlik için)
                    print(f"✓ Rand ({fit_rand:.4f})")
                else:
                    value = val_obl
                    print(f"✓ Obl ({fit_obl:.4f})")

            # Parametre tipini sabitle
            new_params[param] = float(value) if param in CONTINUOUS_PARAMS else int(value)

        # Yeni harmoniyi değerlendir
        new_hp      = HyperParams(**new_params)
        new_fitness = evaluate_fitness(new_hp, epochs=epochs_optimize)

        # En kötüyü değiştir
        if new_fitness < HM[-1]["fitness"]:
            HM[-1] = {"hp": new_hp, "fitness": new_fitness}

        HM = sorted(HM, key=lambda x: x["fitness"])
        best_now = HM[0]["fitness"]
        convergence.append(best_now)

        if best_now < best_ever:
            prev_best = best_ever
            best_ever = best_now
            improvement = ((prev_best - best_ever) / prev_best) * 100 if prev_best > 0 else 0
        else:
            improvement = 0
        print(f"Iter {t:02d}/{NI} | New: {new_fitness:.6f} | Best: {best_now:.6f} | +{improvement:.1f}%")

    return HM[0], convergence


# =========================================================
# 8. FINAL EĞİTİM – G-HS CNN (Önerilen Model)
# =========================================================
def final_evaluate_ghs_cnn(best_hp: HyperParams, epochs: int = 35):
    print("\n🔧 FINAL G-HS-CNN MODELİ EĞİTİLİYOR...")
    tf.keras.backend.clear_session()
    _gen_cache.clear()
    gc.collect()

    train_ds, val_ds, test_ds = create_datasets(best_hp.batch_size, use_subset=False)

    def mixup(images, labels, alpha=0.2):
        lam = tf.random.uniform([], 0.0, alpha)
        idx = tf.random.shuffle(tf.range(tf.shape(images)[0]))
        return (lam * images + (1 - lam) * tf.gather(images, idx),
                lam * labels + (1 - lam) * tf.gather(labels, idx))

    train_ds = train_ds.map(lambda x, y: mixup(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds  = test_ds.prefetch(tf.data.AUTOTUNE)

    model = build_cnn_model(best_hp)

    if DATASET_SIZE is not None:
        n_train = int(DATASET_SIZE * TRAIN_RATIO)
    elif QUICK_TEST:
        n_train = int(min(QUICK_TEST_TOTAL, get_total_image_count()) * TRAIN_RATIO)
    else:
        n_train = int(get_total_image_count() * TRAIN_RATIO)
    total_steps  = epochs * max(n_train // best_hp.batch_size, 1)
    cosine_lr    = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=best_hp.learning_rate,
        first_decay_steps=max(total_steps // 3, 200),
        t_mul=1.5, m_mul=0.9, alpha=1e-6,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_lr),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        ]
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    y_true, y_prob, metrics = evaluate_keras_model(model, test_ds)
    metrics["test_loss"] = float(test_loss)
    metrics["test_accuracy_keras_eval"] = float(test_acc)
    return model, history, y_true, y_prob, metrics

# =========================================================
# 9. FINAL EĞİTİM – CNN BASELINE
# =========================================================
def final_evaluate_cnn_baseline(epochs: int = 15):
    """Sabit hiperparametreli standart CNN baseline eğitimi"""
    print("\n📈 BASELINE CNN EĞİTİLİYOR...")
    tf.keras.backend.clear_session()
    _gen_cache.clear()
    gc.collect()
    train_ds, val_ds, test_ds = create_datasets(32, use_subset=False)

    model = build_cnn_baseline()

    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        ]
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    y_true, y_prob, metrics = evaluate_keras_model(model, test_ds)
    metrics["test_loss"] = float(test_loss)
    metrics["test_accuracy_keras_eval"] = float(test_acc)
    return model, history, y_true, y_prob, metrics


def build_resnet50_model() -> tf.keras.Model:
    """ImageNet ağırlıklı ve dondurulmuş ResNet50 ile transfer learning modeli."""
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.layers.Lambda(lambda t: resnet_preprocess_input(tf.cast(t, tf.float32)))(inputs)
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def final_evaluate_resnet50(epochs: int = 12):
    """ResNet50 transfer learning modeli eğit ve değerlendir."""
    print("\n🧠 RESNET50 (Transfer Learning) EĞİTİLİYOR...")
    tf.keras.backend.clear_session()
    _gen_cache.clear()
    gc.collect()
    train_ds, val_ds, test_ds = create_datasets(32, use_subset=False)

    model = build_resnet50_model()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        ],
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    y_true, y_prob, metrics = evaluate_keras_model(model, test_ds)
    metrics["test_loss"] = float(test_loss)
    metrics["test_accuracy_keras_eval"] = float(test_acc)
    return model, history, y_true, y_prob, metrics


def final_evaluate_sklearn_models():
    """SVM ve Random Forest modellerini düzleştirilmiş görüntülerle eğit/değerlendir."""
    print("\n🌲 SVM ve RANDOM FOREST EĞİTİLİYOR...")
    train_ds, val_ds, test_ds = create_datasets(32, use_subset=False)

    x_train, y_train = dataset_to_numpy_small(train_ds)
    x_val, y_val = dataset_to_numpy_small(val_ds)
    x_test, y_test = dataset_to_numpy_small(test_ds)

    x_train = np.concatenate([x_train, x_val], axis=0)
    y_train = np.concatenate([y_train, y_val], axis=0)

    x_train = flatten_and_normalize_images(x_train)
    x_test = flatten_and_normalize_images(x_test)

    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=True, random_state=42, max_iter=2000), # Sınır
    )

    svm.fit(x_train, y_train)
    svm_prob = svm.predict_proba(x_test)[:, 1]
    svm_metrics = compute_classification_metrics(y_test, svm_prob)

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)
    rf_prob = rf.predict_proba(x_test)[:, 1]
    rf_metrics = compute_classification_metrics(y_test, rf_prob)

    return {
        "SVM": {"y_true": y_test, "y_prob": svm_prob, "metrics": svm_metrics},
        "Random Forest": {"y_true": y_test, "y_prob": rf_prob, "metrics": rf_metrics},
    }
def build_mlp_model(input_dim: int) -> tf.keras.Model:
    """
    Düzleştirilmiş piksel girişi üzerinde MLP.
    Keras ile eğitilir → olasılık çıktısı üretir (SVM/RF gibi).
    """
    inp = tf.keras.Input(shape=(input_dim,))
    x   = tf.keras.layers.Dense(512, activation='relu')(inp)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Dropout(0.4)(x)
    x   = tf.keras.layers.Dense(256, activation='relu')(x)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Dense(128, activation='relu')(x)
    x   = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def final_evaluate_mlp(epochs: int = 20):
    """MLP: 32×32 düzleştirilmiş görüntülerle eğit."""
    print("\n🔷 MLP EĞİTİLİYOR...")
    train_ds, val_ds, test_ds = create_datasets(32, use_subset=False)

    x_train, y_train = dataset_to_numpy_small(train_ds)
    x_val,   y_val   = dataset_to_numpy_small(val_ds)
    x_test,  y_test  = dataset_to_numpy_small(test_ds)

    # Val'i de eğitime ekle (küçük val seti, tam veri kullanımı)
    x_tr = np.concatenate([x_train, x_val], axis=0)
    y_tr = np.concatenate([y_train, y_val], axis=0)

    x_tr   = flatten_and_normalize_images(x_tr)
    x_test = flatten_and_normalize_images(x_test)

    tf.keras.backend.clear_session()
    model = build_mlp_model(x_tr.shape[1])

    # Numpy array → tf.data (EarlyStopping için kendi val setini oluştur)
    val_split = int(len(x_tr) * 0.05)
    x_val_mlp, y_val_mlp = x_tr[:val_split], y_tr[:val_split]
    x_tr_mlp,  y_tr_mlp  = x_tr[val_split:],  y_tr[val_split:]

    model.fit(
        x_tr_mlp, y_tr_mlp,
        validation_data=(x_val_mlp, y_val_mlp),
        epochs=epochs, batch_size=128, verbose=0,
        callbacks=[
            EarlyStopping('val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau('val_loss', factor=0.5, patience=3,
                              min_lr=1e-6, verbose=0),
        ]
    )

    y_prob = model.predict(x_test, verbose=0).flatten()
    metrics = compute_classification_metrics(y_test, y_prob)
    print(f"  MLP → F1:{metrics['f1_score']:.4f}  "
          f"Acc:{metrics['accuracy']:.4f}  AUC:{metrics['auc']:.4f}")
    return y_test, y_prob, metrics

# =========================================================
# 10. GRAFİKLER
# =========================================================
def plot_results(model_results):
    """Tek büyük figürde ROC, F1 bar ve model başına confusion matrix çiz."""
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 3)

    ax_roc = fig.add_subplot(gs[0, :2])
    ax_f1 = fig.add_subplot(gs[0, 2])

    for model_name, result in model_results.items():
        y_true = np.asarray(result["y_true"]).astype(np.int32)
        y_prob = np.asarray(result["y_prob"]).reshape(-1)
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = result["metrics"]["auc"]
            auc_txt = f"{auc:.4f}" if np.isfinite(auc) else "N/A"
            ax_roc.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_txt})")
        except ValueError:
            continue

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.7)
    ax_roc.set_title("Tüm Modeller için ROC Eğrileri", fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.grid(True, alpha=0.3)
    ax_roc.legend(loc="lower right")

    names = list(model_results.keys())
    f1_values = [model_results[name]["metrics"]["f1_score"] for name in names]
    bars = ax_f1.bar(names, f1_values, color=sns.color_palette("Set2", n_colors=len(names)))
    ax_f1.set_title("Model Bazlı F1-Score Karşılaştırması", fontweight="bold")
    ax_f1.set_ylabel("F1-Score")
    ax_f1.set_ylim(0, 1.05)
    ax_f1.grid(True, alpha=0.3, axis="y")
    ax_f1.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, f1_values):
        ax_f1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    cm_axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[2, 2]),
    ]
    for ax, (model_name, result) in zip(cm_axes, model_results.items()):
        cm = np.asarray(result["metrics"]["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", cbar=False, ax=ax)
        ax.set_title(f"{model_name}\nConfusion Matrix", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(CLASS_LABELS, rotation=20)
        ax.set_yticklabels(CLASS_LABELS, rotation=0)

    plt.tight_layout()
    plt.savefig("model_comparison_results.png", dpi=150)
    plt.close()
    print("✓ Grafik kaydedildi: model_comparison_results.png")


# =========================================================
# 11. MAIN
# =========================================================
def run():
    set_seed(42)

    print("\n" + "=" * 70)
    print("🎯 G-HS + OBL + DİNAMİK PAR/BW ile CNN HİPERPARAMETRE OPTİMİZASYONU (v2)")
    print(f"Dataset: Waste Classification (Organic vs Recyclable, toplam {get_total_image_count()} görüntü)")
    if DATASET_SIZE is not None:
        train_n, val_n, test_n = compute_split_sizes(DATASET_SIZE)
        print(f"📊 VERİ BOYUTU: {DATASET_SIZE} görüntü kullanılacak "
              f"(Eğitim ≈ {train_n}, Doğrulama ≈ {val_n}, Test ≈ {test_n}), IMG={IMG_SIZE}")
    elif QUICK_TEST:
        train_n, val_n, test_n = compute_split_sizes(min(QUICK_TEST_TOTAL, get_total_image_count()))
        print(f"⚡ HIZLI TEST MODU: IMG={IMG_SIZE}, Eğitim={train_n}, Val={val_n}, Test={test_n} örnek")
    print("=" * 70)

    # G-HS parametreleri: hızlı test ↔ tam çalıştırma
    if QUICK_TEST:
        HMS, NI, epochs_opt, epochs_final, epochs_base, epochs_resnet = 5, 10, 2, 10, 8, 8
    else:
        HMS, NI, epochs_opt, epochs_final, epochs_base, epochs_resnet = 10, 30, 3, 20, 15, 12

    try:
        # G-HS Optimizasyonu
        best_solution, convergence = ghs_optimize(
            HMS=HMS,
            NI=NI,
            HMCR=0.85,
            PAR_min=0.30,
            PAR_max=0.90,
            BW_min=0.001,
            BW_max=0.1,
            epochs_optimize=epochs_opt
        )

        best_hp = best_solution["hp"]

        print("\n" + "=" * 70)
        print("✅ OPTİMİZASYON TAMAMLANDI")
        print("=" * 70)
        print("\n📊 EN İYİ HİPERPARAMETRELER:")
        print(f"   • Filtreler:     {best_hp.filters}")
        print(f"   • Kernel Boyutu: {best_hp.kernel_size}")
        print(f"   • Blok Sayısı:   {best_hp.num_blocks}")
        print(f"   • Dropout:       {best_hp.dropout:.4f}")
        print(f"   • Öğrenme Oranı: {best_hp.learning_rate:.2e}")
        print(f"   • Batch Size:    {best_hp.batch_size}")
        print(f"   • Dense Units:   {best_hp.dense_units}")
        print(f"   • En İyi Val Loss: {best_solution['fitness']:.6f}")
        print("=" * 70)

        model_results = {}

        # Final G-HS-CNN modeli
        _, _, y_true, y_prob, ghs_metrics = final_evaluate_ghs_cnn(best_hp, epochs=epochs_final)
        model_results["G-HS-CNN v2"] = {
            "y_true": y_true,
            "y_prob": y_prob,
            "metrics": ghs_metrics,
        }

        # Baseline CNN
        _, _, y_true, y_prob, baseline_metrics = final_evaluate_cnn_baseline(epochs=epochs_base)
        model_results["Klasik CNN"] = {
            "y_true": y_true,
            "y_prob": y_prob,
            "metrics": baseline_metrics,
        }

        # ResNet50 transfer learning
        _, _, y_true, y_prob, resnet_metrics = final_evaluate_resnet50(epochs=epochs_resnet)
        model_results["ResNet50"] = {
            "y_true": y_true,
            "y_prob": y_prob,
            "metrics": resnet_metrics,
        }

        y_true, y_prob, mlp_metrics = final_evaluate_mlp(epochs=20)
        model_results["MLP"] = {"y_true": y_true, "y_prob": y_prob, "metrics": mlp_metrics}

        # SVM + Random Forest
        model_results.update(final_evaluate_sklearn_models())
        
        # Ortak görseller
        plot_results(model_results)

        print("\n" + "=" * 70)
        print("📈 TEST SONUÇLARI (6 MODEL)")
        print("=" * 70)
        for model_name, result in model_results.items():
            m = result["metrics"]
            auc_txt = f"{m['auc']:.4f}" if np.isfinite(m["auc"]) else "N/A"
            print(f"\n🔹 {model_name}")
            print(f"   Accuracy : {m['accuracy']:.4f}")
            print(f"   Precision: {m['precision']:.4f}")
            print(f"   Recall   : {m['recall']:.4f}")
            print(f"   F1-Score : {m['f1_score']:.4f}")
            print(f"   AUC      : {auc_txt}")
        print("=" * 70)

        # JSON kaydet
        model_metrics_summary = {}
        for model_name, result in model_results.items():
            m = result["metrics"]
            model_metrics_summary[model_name] = {
                "accuracy": float(m["accuracy"]),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "f1_score": float(m["f1_score"]),
                "auc": float(m["auc"]) if np.isfinite(m["auc"]) else None,
                "confusion_matrix": m["confusion_matrix"],
            }
            if "test_loss" in m:
                model_metrics_summary[model_name]["test_loss"] = float(m["test_loss"])

        summary = {
            "optimization_type": "G-HS + OBL + Dynamic PAR/BW",
            "models_compared": ["G-HS-CNN v2", "Klasik CNN", "ResNet50", "MLP", "SVM", "Random Forest"],
            "dataset": (
                f"Waste Classification ({DATASET_SIZE} images, {IMG_SIZE[0]}x{IMG_SIZE[1]})"
                if DATASET_SIZE is not None
                else (
                    (
                        f"Waste Classification (QUICK_TEST 70/15/15, "
                        f"total={min(QUICK_TEST_TOTAL, get_total_image_count())}, {IMG_SIZE[0]}x{IMG_SIZE[1]})"
                    )
                    if QUICK_TEST
                    else f"Waste Classification ({get_total_image_count()} images, 70/15/15, {IMG_SIZE[0]}x{IMG_SIZE[1]})"
                )
            ),
            "best_hyperparameters": {
                "filters":       best_hp.filters,
                "kernel_size":   best_hp.kernel_size,
                "num_blocks":    best_hp.num_blocks,
                "dropout":       float(best_hp.dropout),
                "learning_rate": float(best_hp.learning_rate),
                "batch_size":    best_hp.batch_size,
                "dense_units":   best_hp.dense_units,
            },
            "optimization_val_loss": float(best_solution["fitness"]),
            "metrics": model_metrics_summary,
            "convergence_history": [float(x) for x in convergence]
        }

        with open("summary.json", "w", encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        print("\n✓ Dosyalar kaydedildi:")
        print("  • model_comparison_results.png")
        print("  • summary.json")

    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()


# =========================================================
# 12. ÇALIŞTIR
# =========================================================
if __name__ == "__main__":
    run()
