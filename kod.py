# =========================================================
# G-HS + OBL + DİNAMİK PAR/BW ile EfficientNetB0 TRANSFER LEARNING
# KAGGLE WASTE CLASSIFICATION (Organic vs Recyclable)
# KLASİK CNN vs G-HS-EfficientNetB0
# ✅ GPU OPTIMIZED + tf.data PIPELINE + MIXED PRECISION
# =========================================================

import os
import gc
import json
import random
import warnings
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dataclasses import dataclass, asdict
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU Hatası: {e}")
    # ⚡ Mixed Precision: uyumlu GPU'larda ~2x hız artışı (Compute Capability ≥ 7.0)
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("⚡ Mixed Precision (float16) AKTİF")
    except Exception as mp_err:
        print(f"⚠️ Mixed Precision etkinleştirilemedi, float32 kullanılıyor: {mp_err}")
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
# 1. HİPERPARAMETRE YAPISI (Transfer Learning için güncellendi)
# =========================================================
@dataclass
class HyperParams:
    dropout: float
    learning_rate: float
    batch_size: int
    dense_units: int


# =========================================================
# 2. VERİ YÜKLEME (tf.data Pipeline – HIZLI)
# =========================================================
# EfficientNetB0 için uygun boyut; ImageDataGenerator'a göre ~2x daha hızlı pipeline
IMG_SIZE = (160, 160)
TRAIN_DIR = "dataset/train"
TEST_DIR  = "dataset/test"
AUTOTUNE  = tf.data.AUTOTUNE

# Dataset cache (batch_size → (train_ds, val_ds, test_ds))
_gen_cache = {}


def create_datasets(batch_size):
    """tf.data pipeline ile veri yükle (batch_size başına cache'li)"""
    if batch_size in _gen_cache:
        return _gen_cache[batch_size]

    print(f"📦 Dataset pipeline hazırlanıyor (Batch={batch_size})...")

    common = dict(
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode='binary',
        seed=42,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, validation_split=0.2, subset="training",
        shuffle=True, **common
    ).prefetch(AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, validation_split=0.2, subset="validation",
        shuffle=False, **common
    ).prefetch(AUTOTUNE)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, shuffle=False, **common
    ).prefetch(AUTOTUNE)

    result = (train_ds, val_ds, test_ds)
    _gen_cache[batch_size] = result
    return result


# =========================================================
# 3. EfficientNetB0 TRANSFER LEARNING MODELİ
# =========================================================
# Veri artırma katmanları modelin içinde tanımlandı → GPU'da çalışır,
# model.evaluate/predict sırasında otomatik olarak devre dışı kalır.
_data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
], name="augmentation")


def build_transfer_model(hp: HyperParams, fine_tune_layers: int = 0):
    """
    EfficientNetB0 tabanlı transfer learning modeli.
      fine_tune_layers=0  → tamamen dondurulmuş base, sadece head eğitimi (hızlı)
      fine_tune_layers>0  → son N katman da eğitilir (fine-tune aşaması)

    NOT: EfficientNetB0, [0, 255] ham piksel değeri bekler;
         model içinde normalizasyon otomatik yapılır.
    """
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Veri artırma (model.fit() sırasında aktif, evaluate/predict'te pasif)
    x = _data_augmentation(inputs)

    # EfficientNetB0 feature extraction
    x = base_model(x, training=False)

    # Sınıflandırma kafası
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(hp.dropout)(x)
    x = tf.keras.layers.Dense(hp.dense_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(hp.dropout * 0.5)(x)
    # Mixed precision ile uyumluluk için çıktı katmanı float32
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# =========================================================
# 4. KLASİK CNN MODELİ (Baseline karşılaştırması için)
# =========================================================
def build_cnn_baseline():
    """Standart CNN baseline – 3 conv blok + BatchNorm"""
    model = tf.keras.Sequential([
        # [0,255] → [0,1] normalizasyon
        tf.keras.layers.Rescaling(1.0 / 255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        # Veri artırma (sadece eğitimde aktif)
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        # Conv Blok 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Conv Blok 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Conv Blok 3
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
    Transfer learning modeli için fitness hesapla.
    Düşük val_loss → daha iyi hiperparametre seti.
    """
    try:
        tf.keras.backend.clear_session()

        train_ds, val_ds, _ = create_datasets(hp.batch_size)
        model = build_transfer_model(hp, fine_tune_layers=0)

        es = EarlyStopping(
            monitor="val_loss",
            patience=1,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=0,
            callbacks=[es]
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
# =========================================================
BOUNDS = {
    "dropout":       (0.1,  0.5),
    "learning_rate": (1e-5, 1e-3),
    "batch_size":    (16,   64),
    "dense_units":   (64,   512),
}

CHOICES = {
    "batch_size":  [16, 32, 64],
    "dense_units": [64, 128, 256, 512],
}

CONTINUOUS_PARAMS = {"dropout", "learning_rate"}
DISCRETE_PARAMS   = {"batch_size", "dense_units"}


def get_nearest_choice(val, choices):
    """En yakın geçerli değeri bul"""
    return min(choices, key=lambda x: abs(x - val))


def random_hyperparams():
    """Rastgele hiperparametreler üret"""
    return HyperParams(
        dropout=round(random.uniform(0.1, 0.5), 4),
        learning_rate=10 ** random.uniform(-5, -3),   # log-scale örnekleme
        batch_size=random.choice(CHOICES["batch_size"]),
        dense_units=random.choice(CHOICES["dense_units"]),
    )


# =========================================================
# 7. G-HS ALGORİTMASI (OBL + DİNAMİK PAR/BW)
# =========================================================
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
    """G-HS ile EfficientNetB0 fine-tuning hiperparametrelerini optimize et"""

    HM = []
    convergence = []

    print("=" * 70)
    print(">>> HM BAŞLATILIYOR (Harmony Memory Initialize)")
    print("=" * 70)

    # 1. HM başlatması
    for i in range(HMS):
        hp = random_hyperparams()
        fitness = evaluate_fitness(hp, epochs=epochs_optimize)
        HM.append({"hp": hp, "fitness": fitness})
        print(
            f"✓ Init {i+1}/{HMS}: "
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

        for param in ["dropout", "learning_rate", "batch_size", "dense_units"]:

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
# 8. FINAL EĞİTİM – TRANSFER LEARNING (2 Aşamalı)
# =========================================================
def final_evaluate_transfer(best_hp: HyperParams, epochs_phase1: int = 10, epochs_phase2: int = 10):
    """
    İki aşamalı transfer learning final eğitimi:
      Aşama 1 → Dondurulmuş base, sadece classification head eğitimi.
      Aşama 2 → Son 30 katman açılır, çok düşük LR ile fine-tuning.
    """
    print("\n🔧 FINAL TRANSFER LEARNING MODELİ EĞİTİLİYOR...")
    train_ds, val_ds, test_ds = create_datasets(best_hp.batch_size)

    # --- Aşama 1: Feature extraction (frozen base) ---
    print("   📌 Aşama 1: Frozen EfficientNetB0 → Head eğitimi")
    model = build_transfer_model(best_hp, fine_tune_layers=0)

    model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs_phase1,
        verbose=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        ]
    )

    # --- Aşama 2: Fine-tuning (son 30 katman) ---
    print("\n   🔓 Aşama 2: Son 30 katman açıldı → Fine-tuning (düşük LR)")
    base_model = model.get_layer("efficientnetb0")
    base_model.trainable = True

    # Son 30 katman hariç dondur (~13% of EfficientNetB0's 237 layers).
    # Bu sayı eğitim süresi ile performans arasında iyi bir denge sağlar:
    # çok az katman açmak başarımı düşürür, çok fazla açmak overfitting riskini artırır.
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # BatchNorm katmanlarını her zaman dondur (fine-tune best practice)
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # 10x daha düşük LR: catastrophic forgetting'i önlemek için standart
    # fine-tuning pratiği (Howard & Ruder, 2018 – "discriminative fine-tuning").
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_hp.learning_rate * 0.1),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs_phase2,
        verbose=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-8, verbose=1),
        ]
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    return model, test_loss, test_acc, history


# =========================================================
# 9. FINAL EĞİTİM – CNN BASELINE
# =========================================================
def final_evaluate_cnn(epochs: int = 15):
    """CNN baseline final eğitimi"""
    print("\n📈 BASELINE CNN EĞİTİLİYOR...")
    train_ds, val_ds, test_ds = create_datasets(32)

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
# 10. GRAFİKLER
# =========================================================
def plot_results(convergence, transfer_acc, cnn_acc):
    """Yakınsama ve accuracy karşılaştırma grafiklerini çiz"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # G-HS Yakınsama eğrisi
    axes[0].plot(convergence, marker='o', linewidth=2, markersize=4, color='royalblue')
    axes[0].set_title("G-HS Yakınsama Eğrisi (OBL + EfficientNetB0)", fontsize=11, fontweight='bold')
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
    labels = ['CNN\n(Baseline)', 'G-HS +\nEfficientNetB0']
    accs   = [cnn_acc * 100, transfer_acc * 100]
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
# 11. MAIN
# =========================================================
def run():
    set_seed(42)

    print("\n" + "=" * 70)
    print("🎯 G-HS + OBL + EfficientNetB0 Transfer Learning Optimizasyonu")
    print("Dataset: Waste Classification (Organic vs Recyclable, 22.500 görüntü)")
    print("=" * 70)

    try:
        # G-HS Optimizasyonu
        best_solution, convergence = ghs_optimize(
            HMS=10,
            NI=30,
            HMCR=0.85,
            PAR_min=0.30,
            PAR_max=0.90,
            BW_min=0.001,
            BW_max=0.1,
            epochs_optimize=3
        )

        best_hp = best_solution["hp"]

        print("\n" + "=" * 70)
        print("✅ OPTİMİZASYON TAMAMLANDI")
        print("=" * 70)
        print("\n📊 EN İYİ HİPERPARAMETRELER:")
        print(f"   • Dropout:       {best_hp.dropout:.4f}")
        print(f"   • Öğrenme Oranı: {best_hp.learning_rate:.2e}")
        print(f"   • Batch Size:    {best_hp.batch_size}")
        print(f"   • Dense Units:   {best_hp.dense_units}")
        print(f"   • En İyi Val Loss: {best_solution['fitness']:.6f}")
        print("=" * 70)

        # Final Transfer Learning modeli (2 aşamalı)
        _, final_loss, final_acc, _ = final_evaluate_transfer(
            best_hp, epochs_phase1=10, epochs_phase2=10
        )

        # Baseline CNN
        _, baseline_loss, baseline_acc, _ = final_evaluate_cnn(epochs=15)

        # Grafik
        plot_results(convergence, final_acc, baseline_acc)

        # Metrikler
        loss_improvement = ((baseline_loss - final_loss) / baseline_loss) * 100
        acc_improvement  = ((final_acc - baseline_acc) / baseline_acc) * 100

        print("\n" + "=" * 70)
        print("📈 TEST SONUÇLARI")
        print("=" * 70)

        print("\n🎯 G-HS + EfficientNetB0 (Transfer Learning)")
        print(f"   Test Loss:     {final_loss:.6f}")
        print(f"   Test Accuracy: {final_acc:.4f}  ({final_acc * 100:.2f}%)")

        print("\n📊 KLASİK CNN (Baseline)")
        print(f"   Test Loss:     {baseline_loss:.6f}")
        print(f"   Test Accuracy: {baseline_acc:.4f}  ({baseline_acc * 100:.2f}%)")

        print("\n🚀 İYİLEŞTİRME")
        print(f"   Loss Azalması:   {loss_improvement:+.2f}%")
        print(f"   Accuracy Artışı: {acc_improvement:+.2f}%")
        print("=" * 70)

        # JSON kaydet
        summary = {
            "optimization_type": "G-HS + OBL + Dynamic PAR/BW",
            "proposed_model":    "EfficientNetB0 Transfer Learning (ImageNet)",
            "dataset":           "Waste Classification (22500 images, 160x160)",
            "best_hyperparameters": {
                "dropout":       float(best_hp.dropout),
                "learning_rate": float(best_hp.learning_rate),
                "batch_size":    best_hp.batch_size,
                "dense_units":   best_hp.dense_units,
            },
            "transfer_learning_results": {
                "test_loss":             float(final_loss),
                "test_accuracy":         float(final_acc),
                "optimization_val_loss": float(best_solution["fitness"])
            },
            "baseline_cnn_results": {
                "test_loss":     float(baseline_loss),
                "test_accuracy": float(baseline_acc)
            },
            "improvements": {
                "loss_reduction_pct":  float(loss_improvement),
                "accuracy_gain_pct":   float(acc_improvement)
            },
            "convergence_history": [float(x) for x in convergence]
        }

        with open("summary.json", "w", encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        print("\n✓ Dosyalar kaydedildi:")
        print("  • ghs_results.png")
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