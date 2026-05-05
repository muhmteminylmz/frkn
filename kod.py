# =========================================================
# G-HS + OBL + DİNAMİK PAR/BW ile CNN HİPERPARAMETRE OPTİMİZASYONU
# KAGGLE WASTE CLASSIFICATION (Organic vs Recyclable)
# KLASİK CNN vs ÖNERİLEN G-HS-CNN
# ✅ GPU OPTIMIZED + TURBO MODE + FIXED
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =========================================================
# GPU SETUP (DirectML)
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
    filters: int
    kernel_size: int
    dropout: float
    learning_rate: float
    batch_size: int


# =========================================================
# 2. VERİ YÜKLEME (CACHE İLE)
# =========================================================
IMG_SIZE = (128, 128)
train_dir = "dataset/train"
test_dir = "dataset/test"

# Global cache
_gen_cache = {}


def create_generators(batch_size):
    """Veri yükle (cache'den reuse et)"""
    
    key = batch_size
    if key in _gen_cache:
        return _gen_cache[key]
    
    print(f"📦 Veri yükleniyor (Batch={batch_size})...")
    
    # ✅ seed parametresi ImageDataGenerator'dan KALDIRANDI
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # ✅ seed'i flow_from_directory'ye TAŞINDI
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=42
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=42
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    result = (train_gen, val_gen, test_gen)
    _gen_cache[key] = result
    return result


# =========================================================
# 3. CNN MODEL (BatchNorm ile)
# =========================================================
def build_cnn_model(hp: HyperParams):
    """CNN modelini oluştur"""
    
    model = Sequential([
        Conv2D(
            hp.filters,
            (hp.kernel_size, hp.kernel_size),
            activation='relu',
            input_shape=(128, 128, 3),
            padding='same'
        ),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(
            hp.filters * 2,
            (hp.kernel_size, hp.kernel_size),
            activation='relu',
            padding='same'
        ),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dropout(hp.dropout),

        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=hp.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# =========================================================
# 4. FITNESS (HIZLANDI)
# =========================================================
def evaluate_fitness(hp: HyperParams, epochs=3):
    """Fitness hesapla (hızlı versiyon)"""
    
    try:
        tf.keras.backend.clear_session()

        train_gen, val_gen, _ = create_generators(hp.batch_size)

        model = build_cnn_model(hp)

        # Agresif early stopping
        es = EarlyStopping(
            monitor="val_loss",
            patience=1,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            train_gen,
            validation_data=val_gen,
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
        return float('inf')


# =========================================================
# 5. PARAMETRE SINIRLARI
# =========================================================
BOUNDS = {
    "filters": (16, 128),
    "kernel_size": (2, 5),
    "dropout": (0.1, 0.5),
    "learning_rate": (0.0001, 0.01),
    "batch_size": (16, 64)
}


def get_nearest_choice(val, choices):
    """En yakın geçerli değeri bul"""
    return min(choices, key=lambda x: abs(x - val))


def random_hyperparams():
    """Rastgele hiperparametreler üret"""
    return HyperParams(
        filters=random.choice([16, 32, 64, 128]),
        kernel_size=random.choice([2, 3, 5]),
        dropout=random.uniform(*BOUNDS["dropout"]),
        learning_rate=random.uniform(*BOUNDS["learning_rate"]),
        batch_size=random.choice([16, 32, 64])
    )


# =========================================================
# 6. G-HS ALGORİTMASI (ORIJINAL + OBL)
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
    """G-HS Optimizasyonu"""
    
    HM = []
    convergence = []

    print("=" * 70)
    print(">>> HM BAŞLATILIYOR (Harmony Memory Initialize)")
    print("=" * 70)

    # 1. HM başlatması
    for i in range(HMS):
        hp = random_hyperparams()
        fitness = evaluate_fitness(hp, epochs=epochs_optimize)

        HM.append({
            "hp": hp,
            "fitness": fitness
        })

        print(f"✓ Init {i+1}/{HMS}: Loss={fitness:.6f}")

    HM = sorted(HM, key=lambda x: x["fitness"])
    best_init = HM[0]["fitness"]

    print("\n" + "=" * 70)
    print(">>> OPTİMİZASYON BAŞLADI")
    print("=" * 70 + "\n")

    # 2. İterasyon
    for t in range(1, NI + 1):

        PAR_t = PAR_min + ((PAR_max - PAR_min) / NI) * t
        BW_t = BW_max * np.exp(np.log(BW_min / BW_max) * (t / NI))

        new_params = {}

        for param in ["filters", "kernel_size", "dropout", "learning_rate", "batch_size"]:

            rand1 = random.random()

            # HMCR Şartı Sağlandığında (Hafızadan Seçim)
            if rand1 <= HMCR:
                chosen = random.choice(HM)["hp"]
                value = getattr(chosen, param)
                rand2 = random.random()

                # PAR Şartı Sağlandığında (Dinamik İnce Ayar)
                if rand2 <= PAR_t:
                    low, high = BOUNDS[param]
                    r = random.uniform(-1, 1)
                    value = value + r * BW_t * (high - low)
                    value = max(low, min(high, value))

                # Kategorik değerleri geçerli sınırlara çek
                if param == "filters":
                    value = get_nearest_choice(value, [16, 32, 64, 128])
                elif param == "kernel_size":
                    value = get_nearest_choice(value, [2, 3, 5])
                elif param == "batch_size":
                    value = get_nearest_choice(value, [16, 32, 64])

            # OBL (Opposition-Based Learning)
            else:
                # 1. Random değer üret
                if param == "filters":
                    val_rand = random.choice([16, 32, 64, 128])
                elif param == "kernel_size":
                    val_rand = random.choice([2, 3, 5])
                elif param == "batch_size":
                    val_rand = random.choice([16, 32, 64])
                else:
                    val_rand = random.uniform(*BOUNDS[param])

                # 2. Zıt değeri hesapla
                low, high = BOUNDS[param]
                val_obl_raw = low + high - val_rand

                # 3. Zıt değeri geçerli formata yuvarla
                if param == "filters":
                    val_obl = get_nearest_choice(val_obl_raw, [16, 32, 64, 128])
                elif param == "kernel_size":
                    val_obl = get_nearest_choice(val_obl_raw, [2, 3, 5])
                elif param == "batch_size":
                    val_obl = get_nearest_choice(val_obl_raw, [16, 32, 64])
                else:
                    val_obl = float(max(low, min(high, val_obl_raw)))

                # 4. İkisini test et
                temp_params_rand = asdict(HM[0]["hp"])
                temp_params_rand[param] = val_rand
                hp_rand = HyperParams(**temp_params_rand)

                temp_params_obl = asdict(HM[0]["hp"])
                temp_params_obl[param] = val_obl
                hp_obl = HyperParams(**temp_params_obl)

                print(f"   [OBL Test - {param}] Rand({val_rand}) vs Obl({val_obl})...", end=" ")
                
                fit_rand = evaluate_fitness(hp_rand, epochs=2)
                fit_obl = evaluate_fitness(hp_obl, epochs=2)

                # 5. En iyi seçeni al
                if fit_rand < fit_obl:
                    value = val_rand
                    print(f"✓ Rand ({fit_rand:.4f})")
                else:
                    value = val_obl
                    print(f"✓ Obl ({fit_obl:.4f})")

            # Parametreyi ata
            new_params[param] = float(value) if param in ["dropout", "learning_rate"] else int(value)

        # Yeni harmoniyi değerlendir
        new_hp = HyperParams(**new_params)
        new_fitness = evaluate_fitness(new_hp, epochs=epochs_optimize)

        # En kötüyü bul ve değiştir
        worst_idx = -1
        if new_fitness < HM[worst_idx]["fitness"]:
            HM[worst_idx] = {
                "hp": new_hp,
                "fitness": new_fitness
            }

        HM = sorted(HM, key=lambda x: x["fitness"])

        best_now = HM[0]["fitness"]
        convergence.append(best_now)

        improvement = ((best_init - best_now) / best_init) * 100 if best_init > 0 else 0
        print(f"Iter {t:02d}/{NI} | New Loss: {new_fitness:.6f} | Best: {best_now:.6f} | +{improvement:.1f}%")

    return HM[0], convergence


# =========================================================
# 7. FINAL TEST
# =========================================================
def final_evaluate(best_hp: HyperParams, epochs=15):
    """Optimum parametrelerle final eğitim"""
    
    print("\n🔧 FINAL MODEL EĞİTİLİYOR...")
    
    train_gen, val_gen, test_gen = create_generators(best_hp.batch_size)

    model = build_cnn_model(best_hp)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )

    test_loss, test_acc = model.evaluate(test_gen, verbose=0)

    return model, test_loss, test_acc, history


# =========================================================
# 8. BASELINE CNN
# =========================================================
def baseline_cnn():
    """Standart parametrelerle baseline model"""
    
    print("\n📈 BASELINE CNN EĞİTİLİYOR...")
    
    baseline_hp = HyperParams(
        filters=32,
        kernel_size=3,
        dropout=0.25,
        learning_rate=0.001,
        batch_size=32
    )

    return final_evaluate(baseline_hp, epochs=15)


# =========================================================
# 9. GRAFİK ÇIZME
# =========================================================
def plot_convergence(convergence):
    """Yakınsama grafiğini çiz"""
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(convergence, marker='o', linewidth=2, markersize=4, color='blue')
    plt.title("G-HS Yakınsama Eğrisi (OBL ile)", fontsize=12, fontweight='bold')
    plt.xlabel("İterasyon", fontsize=11)
    plt.ylabel("En İyi Validation Loss", fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    improvement = [(convergence[0] - x) / convergence[0] * 100 for x in convergence]
    plt.bar(range(len(improvement)), improvement, color='green', alpha=0.7)
    plt.title("İyileşme Yüzdesi", fontsize=12, fontweight='bold')
    plt.xlabel("İterasyon", fontsize=11)
    plt.ylabel("İyileşme (%)", fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("ghs_convergence.png", dpi=150)
    plt.close()
    
    print("✓ Grafik kaydedildi: ghs_convergence.png")


# =========================================================
# 10. MAIN
# =========================================================
def run():
    set_seed(42)

    print("\n" + "=" * 70)
    print("🎯 G-HS + OBL + DİNAMİK PAR/BW ile CNN HİPERPARAMETRE OPTİMİZASYONU")
    print("Dataset: Waste Classification (Organic vs Recyclable)")
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
        print(f"   • Filtreler: {best_hp.filters}")
        print(f"   • Kernel Boyutu: {best_hp.kernel_size}")
        print(f"   • Dropout: {best_hp.dropout:.4f}")
        print(f"   • Öğrenme Oranı: {best_hp.learning_rate:.6f}")
        print(f"   • Batch Size: {best_hp.batch_size}")
        print(f"   • En İyi Validation Loss: {best_solution['fitness']:.6f}")
        print("=" * 70)

        # Final model
        final_model, final_loss, final_acc, _ = final_evaluate(best_hp, epochs=15)

        # Baseline model
        _, baseline_loss, baseline_acc, _ = baseline_cnn()

        # Grafik
        plot_convergence(convergence)

        # Sonuçlar
        print("\n" + "=" * 70)
        print("📈 TEST SONUÇLARI")
        print("=" * 70)

        print("\n🎯 ÖNERİLEN G-HS + CNN")
        print(f"   Test Loss:     {final_loss:.6f}")
        print(f"   Test Accuracy: {final_acc:.6f}")

        print("\n📊 KLASİK CNN (Baseline)")
        print(f"   Test Loss:     {baseline_loss:.6f}")
        print(f"   Test Accuracy: {baseline_acc:.6f}")

        # İyileşme hesapla
        loss_improvement = ((baseline_loss - final_loss) / baseline_loss) * 100
        acc_improvement = ((final_acc - baseline_acc) / baseline_acc) * 100

        print("\n🚀 İYİLEŞTİRME")
        print(f"   Loss Azalması:     {loss_improvement:+.2f}%")
        print(f"   Accuracy Artışı:   {acc_improvement:+.2f}%")
        print("=" * 70)

        # JSON'a kaydet
        summary = {
            "optimization_type": "G-HS + OBL + Dynamic PAR/BW",
            "dataset": "Waste Classification",
            "best_hyperparameters": {
                "filters": best_hp.filters,
                "kernel_size": best_hp.kernel_size,
                "dropout": float(best_hp.dropout),
                "learning_rate": float(best_hp.learning_rate),
                "batch_size": best_hp.batch_size
            },
            "ghs_results": {
                "test_loss": float(final_loss),
                "test_accuracy": float(final_acc),
                "validation_loss_at_optimization": float(best_solution["fitness"])
            },
            "baseline_results": {
                "test_loss": float(baseline_loss),
                "test_accuracy": float(baseline_acc)
            },
            "improvements": {
                "loss_reduction_percent": float(loss_improvement),
                "accuracy_gain_percent": float(acc_improvement)
            },
            "convergence_history": [float(x) for x in convergence]
        }

        with open("summary.json", "w", encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        print("\n✓ Dosyalar kaydedildi:")
        print("  • ghs_convergence.png")
        print("  • summary.json")

    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()


# =========================================================
# 11. ÇALIŞTIR
# =========================================================
if __name__ == "__main__":
    run()