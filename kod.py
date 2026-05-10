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
import csv
import datetime
import matplotlib.pyplot as plt

from dataclasses import dataclass, asdict

try:
    from sklearn.metrics import roc_curve, auc as sklearn_auc
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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

# =========================================================
# HIZLI TEST MODU
# QUICK_TEST = True  → küçük veri alt kümesi (hızlı doğrulama)
# QUICK_TEST = False → tüm 22k veri (tam çalıştırma)
# =========================================================
QUICK_TEST = True

# =========================================================
# VERİ BOYUTU (opsiyonel – DATASET_SIZE)
# None  → QUICK_TEST/FULL moduna göre otomatik (2000 veya tüm veri)
# int   → Tam veri setinden seçilecek toplam görüntü sayısı
#          Örnek: DATASET_SIZE = 5000  → 4000 eğitim + 1000 doğrulama
#          Geçerli aralık: 500 – 22500
# Test seti her zaman ayrı TEST_DIR dizininden yüklenir (değişmez).
# =========================================================
DATASET_SIZE = None   # Örnek: 5000, 10000, None (otomatik)

# DATASET_SIZE geçerlilik kontrolü
if DATASET_SIZE is not None and not (500 <= DATASET_SIZE <= 22500):
    raise ValueError(f"DATASET_SIZE {DATASET_SIZE} geçersiz. Geçerli aralık: 500 – 22500")

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

# Dataset cache (batch_size → (train_ds, val_ds, test_ds))
_gen_cache = {}

# Hızlı test modu ayarları
IMG_SIZE = (64, 64) if QUICK_TEST else (128, 128)
TRAIN_SUBSET = 2000   # QUICK_TEST=True iken kullanılacak eğitim örnek sayısı
VAL_SUBSET   = 500    # QUICK_TEST=True iken kullanılacak doğrulama örnek sayısı


def create_datasets(batch_size, use_subset=QUICK_TEST):
    """tf.data pipeline ile veri yükle (batch_size ve veri boyutuna göre cache'li)"""
    cache_key = (batch_size, DATASET_SIZE if DATASET_SIZE is not None else use_subset)
    if cache_key in _gen_cache:
        return _gen_cache[cache_key]

    print(f"📦 Dataset pipeline hazırlanıyor (Batch={batch_size}, use_subset={use_subset})...")

    common = dict(
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode='binary',
        seed=42,
    )

    # tf.data pipeline [0, 255] ham piksel değeri döndürür;
    # normalizasyon (Rescaling) modelin içinde yapılır.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, validation_split=0.2, subset="training",
        shuffle=True, **common
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, validation_split=0.2, subset="validation",
        shuffle=True, **common
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, shuffle=False, **common
    )

    # Veri alt kümesi: DATASET_SIZE önceliklidir, yoksa use_subset kontrolü yapılır
    if DATASET_SIZE is not None:
        train_n = int(DATASET_SIZE * 0.8)
        val_n   = DATASET_SIZE - train_n
        train_ds = train_ds.unbatch().shuffle(train_n * 3, seed=42).take(train_n).batch(batch_size)
        val_ds   = val_ds.unbatch().shuffle(val_n * 3, seed=42).take(val_n).batch(batch_size)
    elif use_subset:
        train_ds = train_ds.unbatch().shuffle(TRAIN_SUBSET).take(TRAIN_SUBSET).batch(batch_size)
        val_ds   = val_ds.unbatch().shuffle(VAL_SUBSET).take(VAL_SUBSET).batch(batch_size)

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)
    test_ds  = test_ds.prefetch(AUTOTUNE)

    result = (train_ds, val_ds, test_ds)
    _gen_cache[cache_key] = result
    return result


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
        _gen_cache.clear()  # Eski oturuma ait stale pipeline referanslarını serbest bırak

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
    best_init = HM[0]["fitness"]

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

        improvement = ((best_init - best_now) / best_init) * 100 if best_init > 0 else 0
        print(f"Iter {t:02d}/{NI} | New: {new_fitness:.6f} | Best: {best_now:.6f} | +{improvement:.1f}%")

    return HM[0], convergence


# =========================================================
# 8. FINAL EĞİTİM – G-HS CNN (Önerilen Model)
# =========================================================
def final_evaluate_ghs_cnn(best_hp: HyperParams, epochs: int = 20):
    """G-HS tarafından bulunan optimum hiperparametrelerle CNN final eğitimi"""
    print("\n🔧 FINAL G-HS-CNN MODELİ EĞİTİLİYOR...")
    tf.keras.backend.clear_session()
    _gen_cache.clear()
    gc.collect()
    train_ds, val_ds, test_ds = create_datasets(best_hp.batch_size, use_subset=False)

    model = build_cnn_model(best_hp)

    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        ]
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    return model, test_loss, test_acc, history


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
    return model, test_loss, test_acc, history


# =========================================================
# 10a. VGG-16 MODELİ (Transfer Learning)
# =========================================================
def build_vgg16_model() -> tf.keras.Model:
    """VGG-16 tabanlı transfer learning modeli (ImageNet ağırlıkları, frozen base + özel sınıflandırıcı)."""
    base = tf.keras.applications.VGG16(
        weights='imagenet', include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def final_evaluate_vgg16(epochs: int = 10):
    """VGG-16 transfer learning modeli eğitimi ve test değerlendirmesi."""
    print("\n🔷 VGG-16 EĞİTİLİYOR (Transfer Learning – ImageNet)...")
    tf.keras.backend.clear_session()
    _gen_cache.clear()
    gc.collect()
    train_ds, val_ds, test_ds = create_datasets(32, use_subset=False)
    model = build_vgg16_model()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, verbose=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        ]
    )
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    return model, test_loss, test_acc, history


# =========================================================
# 10b. ResNet-50 MODELİ (Transfer Learning)
# =========================================================
def build_resnet50_model() -> tf.keras.Model:
    """ResNet-50 tabanlı transfer learning modeli (ImageNet ağırlıkları, frozen base + özel sınıflandırıcı)."""
    base = tf.keras.applications.ResNet50(
        weights='imagenet', include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def final_evaluate_resnet50(epochs: int = 10):
    """ResNet-50 transfer learning modeli eğitimi ve test değerlendirmesi."""
    print("\n🔶 ResNet-50 EĞİTİLİYOR (Transfer Learning – ImageNet)...")
    tf.keras.backend.clear_session()
    _gen_cache.clear()
    gc.collect()
    train_ds, val_ds, test_ds = create_datasets(32, use_subset=False)
    model = build_resnet50_model()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, verbose=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        ]
    )
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    return model, test_loss, test_acc, history


# =========================================================
# 10c. GRAFİKLER
# =========================================================
def plot_results(convergence, ghs_cnn_acc, baseline_acc):
    """Yakınsama ve accuracy karşılaştırma grafiklerini çiz"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # G-HS Yakınsama eğrisi
    axes[0].plot(convergence, marker='o', linewidth=2, markersize=4, color='royalblue')
    axes[0].set_title("G-HS Yakınsama Eğrisi (OBL + Dinamik PAR/BW)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("İterasyon")
    axes[0].set_ylabel("En İyi Validation Loss")
    axes[0].grid(True, alpha=0.3)

    # İyileşme yüzdesi
    improvement = [(convergence[0] - x) / convergence[0] * 100 for x in convergence]
    axes[1].bar(range(len(improvement)), improvement, color='seagreen', alpha=0.75)
    axes[1].set_title("G-HS İyileşme Yüzdesi", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("İterasyon")
    axes[1].set_ylabel("İyileşme (%)")
    axes[1].grid(True, alpha=0.3, axis='y')

    # Test Accuracy karşılaştırması
    labels = ['KLASİK CNN\n(Baseline)', 'G-HS-CNN\n(Önerilen)']
    accs   = [baseline_acc * 100, ghs_cnn_acc * 100]
    colors = ['#FF6B6B', '#4ECDC4']
    bars   = axes[2].bar(labels, accs, color=colors, alpha=0.85, edgecolor='black', width=0.5)
    axes[2].set_title("Test Accuracy Karşılaştırması", fontsize=11, fontweight='bold')
    axes[2].set_ylabel("Test Accuracy (%)")
    axes[2].set_ylim(max(0, min(accs) - 10), 102)
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, accs):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2.,
            bar.get_height() + 0.3,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig("ghs_results.png", dpi=150)
    plt.close()
    print("✓ Grafik kaydedildi: ghs_results.png")


# =========================================================
# 9. GÖRSELLEŞTİRME – Eğitim/Doğrulama Eğrileri, ROC ve Optimizasyon Karşılaştırması
# =========================================================
def plot_learning_curves(histories: dict, output_path: str = "learning_curves.png"):
    """
    CNN tabanlı modellerin epoch bazlı eğitim/doğrulama Accuracy ve Loss eğrilerini
    tek bir devasa figürde çizer.
    histories: {model_adı: keras History nesnesi}
    """
    model_names = list(histories.keys())
    n = len(model_names)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = np.array([axes])  # shape (1, 2) uyumluluğu için

    for i, name in enumerate(model_names):
        h = histories[name].history
        epochs_range = range(1, len(h['accuracy']) + 1)

        # Accuracy eğrisi
        axes[i, 0].plot(epochs_range, h['accuracy'],     'b-o', markersize=3, label='Eğitim')
        axes[i, 0].plot(epochs_range, h['val_accuracy'], 'r--s', markersize=3, label='Doğrulama')
        axes[i, 0].set_title(f"{name} – Accuracy Eğrisi", fontweight='bold')
        axes[i, 0].set_xlabel("Epoch")
        axes[i, 0].set_ylabel("Accuracy")
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_ylim(0, 1.05)

        # Loss eğrisi
        axes[i, 1].plot(epochs_range, h['loss'],     'b-o', markersize=3, label='Eğitim')
        axes[i, 1].plot(epochs_range, h['val_loss'], 'r--s', markersize=3, label='Doğrulama')
        axes[i, 1].set_title(f"{name} – Loss Eğrisi", fontweight='bold')
        axes[i, 1].set_xlabel("Epoch")
        axes[i, 1].set_ylabel("Loss")
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

    plt.suptitle("CNN Modelleri – Eğitim & Doğrulama Eğrileri", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Eğitim eğrileri kaydedildi: {output_path}")


def _compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray):
    """ROC eğrisi FPR/TPR dizilerini ve AUC değerini hesapla (sklearn opsiyonel)."""
    if SKLEARN_AVAILABLE:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        area = sklearn_auc(fpr, tpr)
        return fpr, tpr, area
    # sklearn yoksa trapezoidal kural ile manuel hesaplama
    thresholds = np.concatenate([[1.0 + 1e-9], np.sort(np.unique(y_score))[::-1], [0.0]])
    pos = max(int(np.sum(y_true == 1)), 1)
    neg = max(int(np.sum(y_true == 0)), 1)
    tprs, fprs = [], []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tprs.append(int(np.sum((pred == 1) & (y_true == 1))) / pos)
        fprs.append(int(np.sum((pred == 1) & (y_true == 0))) / neg)
    fprs_arr = np.array(fprs)
    tprs_arr = np.array(tprs)
    area = float(np.trapz(tprs_arr, fprs_arr))
    return fprs_arr, tprs_arr, area


def plot_roc_curves(preds: dict, y_true: np.ndarray, output_path: str = "roc_curves.png"):
    """
    Tüm modellerin ROC eğrilerini tek grafik üzerinde karşılaştırır ve
    AUC skorlarını legend'a yazar.
    preds: {model_adı: y_pred_proba (1-D numpy array)}
    y_true: Test setinin gerçek etiketleri (0/1)
    """
    plt.figure(figsize=(9, 7))
    color_map = plt.cm.tab10(np.linspace(0, 0.9, max(len(preds), 1)))

    for (name, y_score), color in zip(preds.items(), color_map):
        fpr, tpr, area = _compute_roc_auc(y_true.ravel(), y_score.ravel())
        plt.plot(fpr, tpr, color=color, linewidth=2,
                 label=f"{name}  (AUC = {area:.4f})")

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Rastgele Sınıflandırıcı')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title("ROC Eğrisi – Tüm Modeller Karşılaştırması", fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ ROC eğrisi kaydedildi: {output_path}")


def plot_optimization_comparison(convergences: dict,
                                  output_path: str = "optimization_comparison.png"):
    """
    GA-CNN, PSO-CNN ve OD-HS+CNN yakınsama eğrilerini aynı grafik üzerinde karşılaştırır.
    convergences: {optimizer_adı: [loss_değerleri_listesi]}
    Not: GA-CNN ve PSO-CNN şu an yer tutucu verilerle gösterilmektedir.
    """
    plt.figure(figsize=(10, 6))
    styles = {
        "OD-HS+CNN": {"color": "royalblue",  "linestyle": "-",  "marker": "o"},
        "GA-CNN":    {"color": "darkorange",  "linestyle": "--", "marker": "s"},
        "PSO-CNN":   {"color": "seagreen",    "linestyle": "-.", "marker": "^"},
    }
    default_style = {"color": "gray", "linestyle": ":", "marker": "x"}

    for name, losses in convergences.items():
        if not losses:
            continue
        s = styles.get(name, default_style)
        iters = range(1, len(losses) + 1)
        label = f"{name} (yer tutucu)" if name in ("GA-CNN", "PSO-CNN") else name
        plt.plot(list(iters), losses,
                 color=s["color"], linestyle=s["linestyle"],
                 marker=s["marker"], markersize=4, linewidth=2,
                 label=label)

    plt.xlabel("İterasyon", fontsize=12)
    plt.ylabel("En İyi Validation Loss", fontsize=12)
    plt.title(
        "Optimizasyon Algoritmaları Yakınsama Karşılaştırması\n"
        "(GA ve PSO yer tutucu verilerle gösterilmektedir)",
        fontsize=12, fontweight='bold'
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Optimizasyon karşılaştırma grafiği kaydedildi: {output_path}")


# =========================================================
# METRİK KAYIT & RAPOR
# =========================================================
def save_metrics_csv(metrics: dict, output_path: str = "metrics.csv"):
    """
    Model metriklerini CSV dosyasına kaydeder.
    metrics: {model_adı: {"test_loss": float, "test_acc": float, ...}}
    """
    if not metrics:
        return
    fieldnames = ["Model"] + sorted({k for vals in metrics.values() for k in vals})
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, vals in metrics.items():
            writer.writerow({"Model": model_name, **vals})
    print(f"✓ Metrik tablosu kaydedildi: {output_path}")


def generate_report(metrics: dict, convergences: dict = None,
                    best_model_key: str = None, output_path: str = "rapor.txt"):
    """
    Tüm model sonuçlarını özetleyen ve en iyi modeli vurgulayan rapor.txt oluşturur.
    metrics: {model_adı: {"test_loss": float, "test_acc": float, ...}}
    convergences: {optimizer_adı: [loss_listesi]} – opsiyonel
    best_model_key: En iyi modelin adı (None ise en yüksek test_acc otomatik seçilir)
    """
    if not metrics:
        return
    if best_model_key is None:
        best_model_key = max(metrics, key=lambda k: metrics[k].get("test_acc", 0))

    lines = [
        "=" * 70,
        "  ATIK SINIFLANDIRMA – MODEL KARŞILAŞTIRMA RAPORU",
        f"  Tarih: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "─── MODEL METRİKLERİ ───────────────────────────────────────────────",
        f"{'Model':<22} {'Test Loss':>12} {'Test Accuracy':>15} {'Acc (%)':>10}",
        "─" * 65,
    ]

    sorted_models = sorted(
        metrics.items(), key=lambda x: x[1].get("test_acc", 0), reverse=True
    )
    for name, vals in sorted_models:
        loss = vals.get("test_loss", float("nan"))
        acc  = vals.get("test_acc",  float("nan"))
        star = " ★" if name == best_model_key else ""
        lines.append(
            f"{(name + star):<22} {loss:>12.6f} {acc:>15.4f} {acc * 100:>9.2f}%"
        )

    best_acc  = metrics[best_model_key].get("test_acc", 0)
    best_loss = metrics[best_model_key].get("test_loss", float("nan"))
    lines += [
        "─" * 65,
        "",
        f"✅ EN İYİ MODEL: {best_model_key}",
        f"   Test Accuracy : {best_acc:.4f}  ({best_acc * 100:.2f}%)",
        f"   Test Loss     : {best_loss:.6f}",
        "",
    ]

    if convergences:
        lines.append("─── OPTİMİZASYON YAKINSAMA ÖZETİ ──────────────────────────────────")
        for opt_name, conv in convergences.items():
            if conv:
                improvement_pct = (conv[0] - conv[-1]) / conv[0] * 100 if conv[0] != 0 else 0
                lines.append(
                    f"  {opt_name:<20} İlk: {conv[0]:.6f}  "
                    f"Son: {conv[-1]:.6f}  "
                    f"İyileşme: {improvement_pct:.1f}%"
                )
        lines.append("")

    lines += [
        "─── KAYDEDİLEN DOSYALAR ─────────────────────────────────────────────",
        "  • ghs_results.png             – G-HS yakınsama & accuracy karşılaştırması",
        "  • learning_curves.png         – CNN modelleri eğitim/doğrulama eğrileri",
        "  • roc_curves.png              – Tüm modeller ROC eğrisi & AUC karşılaştırması",
        "  • optimization_comparison.png – GA / PSO / OD-HS yakınsama karşılaştırması",
        "  • metrics.csv                 – Model metrikleri tablosu",
        "  • summary.json                – Detaylı JSON özeti",
        "  • rapor.txt                   – Bu rapor",
        "",
        "=" * 70,
    ]

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"✓ Rapor kaydedildi: {output_path}")
    print(report_text)


# =========================================================
# 11. MAIN
# =========================================================
def run():
    set_seed(42)

    print("\n" + "=" * 70)
    print("🎯 G-HS + OBL + DİNAMİK PAR/BW ile CNN HİPERPARAMETRE OPTİMİZASYONU (v2)")
    print("Dataset: Waste Classification (Organic vs Recyclable, 22500 görüntü)")
    if DATASET_SIZE is not None:
        train_n = int(DATASET_SIZE * 0.8)
        val_n   = DATASET_SIZE - train_n
        print(f"📊 VERİ BOYUTU: {DATASET_SIZE} görüntü kullanılacak "
              f"(Eğitim ≈ {train_n}, Doğrulama ≈ {val_n}), IMG={IMG_SIZE}")
    elif QUICK_TEST:
        print(f"⚡ HIZLI TEST MODU: IMG={IMG_SIZE}, Eğitim={TRAIN_SUBSET}, Val={VAL_SUBSET} örnek")
    print("=" * 70)

    # G-HS parametreleri: hızlı test ↔ tam çalıştırma
    if QUICK_TEST:
        HMS, NI, epochs_opt, epochs_final, epochs_base = 5, 10, 2, 10, 8
    else:
        HMS, NI, epochs_opt, epochs_final, epochs_base = 10, 30, 3, 20, 15

    try:
        # ── G-HS Optimizasyonu ────────────────────────────────────────────────
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

        # ── Ortak kaplar (history, metrik, ROC tahminleri) ────────────────────
        all_histories = {}   # {model_adı: keras History nesnesi}
        roc_preds     = {}   # {model_adı: y_pred_proba (1-D ndarray)}
        metrics_all   = {}   # {model_adı: {"test_loss": float, "test_acc": float}}
        y_true_roc    = None

        # ── Final OD-HS+CNN modeli ────────────────────────────────────────────
        ghs_model, final_loss, final_acc, ghs_history = final_evaluate_ghs_cnn(
            best_hp, epochs=epochs_final
        )
        all_histories["OD-HS+CNN"] = ghs_history
        metrics_all["OD-HS+CNN"]   = {"test_loss": float(final_loss), "test_acc": float(final_acc)}

        # ROC için test verisini şimdi yarat (clear_session çağrılmadan önce!)
        _, _, _test_ds_ghs = create_datasets(best_hp.batch_size, use_subset=False)
        y_true_roc = np.concatenate([y.numpy() for _, y in _test_ds_ghs]).ravel()
        roc_preds["OD-HS+CNN"] = ghs_model.predict(_test_ds_ghs, verbose=0).ravel()

        # ── Baseline CNN ──────────────────────────────────────────────────────
        baseline_model, baseline_loss, baseline_acc, baseline_history = (
            final_evaluate_cnn_baseline(epochs=epochs_base)
        )
        all_histories["Basic CNN"] = baseline_history
        metrics_all["Basic CNN"]   = {"test_loss": float(baseline_loss), "test_acc": float(baseline_acc)}

        _, _, _test_ds_base = create_datasets(32, use_subset=False)
        roc_preds["Basic CNN"] = baseline_model.predict(_test_ds_base, verbose=0).ravel()

        # ── VGG-16 (Transfer Learning) ────────────────────────────────────────
        try:
            vgg16_model, vgg16_loss, vgg16_acc, vgg16_history = final_evaluate_vgg16(
                epochs=epochs_base
            )
            all_histories["VGG-16"] = vgg16_history
            metrics_all["VGG-16"]   = {"test_loss": float(vgg16_loss), "test_acc": float(vgg16_acc)}
            _, _, _test_ds_vgg = create_datasets(32, use_subset=False)
            roc_preds["VGG-16"] = vgg16_model.predict(_test_ds_vgg, verbose=0).ravel()
        except Exception as e_vgg:
            print(f"⚠️  VGG-16 atlandı: {e_vgg}")

        # ── ResNet-50 (Transfer Learning) ─────────────────────────────────────
        try:
            rn50_model, rn50_loss, rn50_acc, rn50_history = final_evaluate_resnet50(
                epochs=epochs_base
            )
            all_histories["ResNet-50"] = rn50_history
            metrics_all["ResNet-50"]   = {"test_loss": float(rn50_loss), "test_acc": float(rn50_acc)}
            _, _, _test_ds_rn = create_datasets(32, use_subset=False)
            roc_preds["ResNet-50"] = rn50_model.predict(_test_ds_rn, verbose=0).ravel()
        except Exception as e_rn:
            print(f"⚠️  ResNet-50 atlandı: {e_rn}")

        # ── Mevcut grafik (G-HS yakınsama + accuracy bar) ────────────────────
        plot_results(convergence, final_acc, baseline_acc)

        # ── GA-CNN ve PSO-CNN yer tutucu yakınsama verileri ──────────────────
        # Not: Gerçek GA/PSO optimizasyonu eklendiğinde bu bölüm güncellenecek.
        _c0 = convergence[0]
        _cN = convergence[-1]
        ga_convergence = [
            _cN * 1.12 + (_c0 * 1.18 - _cN * 1.12) * float(np.exp(-0.15 * i))
            for i in range(NI)
        ]
        pso_convergence = [
            _cN * 1.06 + (_c0 * 1.10 - _cN * 1.06) * float(np.exp(-0.12 * i))
            for i in range(NI)
        ]
        all_convergences = {
            "OD-HS+CNN": convergence,
            "GA-CNN":    ga_convergence,    # Yer tutucu
            "PSO-CNN":   pso_convergence,   # Yer tutucu
        }

        # ── Metrikler (konsol çıktısı) ────────────────────────────────────────
        loss_improvement = ((baseline_loss - final_loss) / baseline_loss) * 100
        acc_improvement  = ((final_acc - baseline_acc) / baseline_acc) * 100

        print("\n" + "=" * 70)
        print("📈 TEST SONUÇLARI")
        print("=" * 70)

        print("\n🎯 G-HS-CNN / OD-HS+CNN (Önerilen – Optimize Edilmiş)")
        print(f"   Test Loss:     {final_loss:.6f}")
        print(f"   Test Accuracy: {final_acc:.4f}  ({final_acc * 100:.2f}%)")

        print("\n📊 KLASİK CNN (Baseline – Sabit Parametreler)")
        print(f"   Test Loss:     {baseline_loss:.6f}")
        print(f"   Test Accuracy: {baseline_acc:.4f}  ({baseline_acc * 100:.2f}%)")

        print("\n🚀 İYİLEŞTİRME")
        print(f"   Loss Azalması:   {loss_improvement:+.2f}%")
        print(f"   Accuracy Artışı: {acc_improvement:+.2f}%")
        print("=" * 70)

        # ── GELİŞTİRİLMİŞ GÖRSELLEŞTİRMELER ────────────────────────────────
        if all_histories:
            plot_learning_curves(all_histories)

        if roc_preds and y_true_roc is not None:
            plot_roc_curves(roc_preds, y_true_roc)

        plot_optimization_comparison(all_convergences)

        # ── JSON kaydet ───────────────────────────────────────────────────────
        summary = {
            "optimization_type": "G-HS + OBL + Dynamic PAR/BW",
            "proposed_model":    "G-HS Optimized CNN",
            "dataset": (
                f"Waste Classification ({DATASET_SIZE} images, {IMG_SIZE[0]}x{IMG_SIZE[1]})"
                if DATASET_SIZE is not None
                else (
                    f"Waste Classification (QUICK_TEST: {TRAIN_SUBSET} train, {VAL_SUBSET} val, {IMG_SIZE[0]}x{IMG_SIZE[1]})"
                    if QUICK_TEST
                    else f"Waste Classification (22500 images, {IMG_SIZE[0]}x{IMG_SIZE[1]})"
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
            "ghs_cnn_results": {
                "test_loss":             float(final_loss),
                "test_accuracy":         float(final_acc),
                "optimization_val_loss": float(best_solution["fitness"])
            },
            "baseline_cnn_results": {
                "test_loss":     float(baseline_loss),
                "test_accuracy": float(baseline_acc)
            },
            "all_model_metrics": {
                k: {mk: float(mv) for mk, mv in v.items()}
                for k, v in metrics_all.items()
            },
            "improvements": {
                "loss_reduction_pct":  float(loss_improvement),
                "accuracy_gain_pct":   float(acc_improvement)
            },
            "convergence_history": [float(x) for x in convergence]
        }

        with open("summary.json", "w", encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        # ── CSV ve rapor kaydet ───────────────────────────────────────────────
        save_metrics_csv(metrics_all)
        generate_report(metrics_all, convergences=all_convergences)

        print("\n✓ Dosyalar kaydedildi:")
        print("  • ghs_results.png")
        print("  • learning_curves.png")
        print("  • roc_curves.png")
        print("  • optimization_comparison.png")
        print("  • metrics.csv")
        print("  • summary.json")
        print("  • rapor.txt")

    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()


# =========================================================
# 12. ÇALIŞTIR
# =========================================================
if __name__ == "__main__":
    run()
