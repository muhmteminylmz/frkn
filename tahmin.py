#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  Güneş Enerjisi Üretim Tahmini – G-HS Hibrit Derin Öğrenme (v4)
  ── DirectML Uyumlu Subprocess İzolasyon Mimarisi ─────────────────────────────

  v3 → v4 Düzeltmeleri:
  ─────────────────────
  [1] CudnnRNN Hatası (En Kritik):
      tensorflow-directml, NVIDIA'nın cuDNN kütüphanesini kullanmaz.
      Keras; GRU ve LSTM katmanları için varsayılan olarak implementation=2
      seçer ve bu CudnnRNN op'unu çağırır.  DirectML bu op'u DESTEKLEMEZ.
      Düzeltme: Tüm GRU/LSTM katmanlarına  implementation=1  eklendi.
      implementation=1 → element-wise (pure-TF) ops → DirectML uyumlu.

  [2] RNN Çökmesi (rc=3221226505 = STATUS_STACK_BUFFER_OVERRUN):
      SimpleRNN 3865 saniye sonra Windows heap bozulmasıyla çöktü.
      Neden: Küçük batch (16) ile büyük veri → DirectML D3D12 descriptor
      leak → eventual crash.
      Düzeltme: batch_size=32, gradient clipping (clipnorm=1.0), epochs=60.

  [3] G-HS Son Eğitim Başarısızlığı:
      G-HS final train da CudnnRNN hatasına çarptı (aynı sebep [1]).
      Düzeltme: build_model fonksiyonuna implementation=1 eklendi.

  [4] LSTM/GRU rc=1:
      RNN çökmesi sonrası bozulan process environment nedeniyle başlatılamadı.
      Subprocess izolasyonu sayesinde bu zaten çözülmeli, ama [1] düzeltmesi
      temel sorunu ortadan kaldırır.

  MİMARİ:
    Ana Süreç ──┬── worker_train_v4.py [CNN]      → cnn.weights.h5
                ├── worker_train_v4.py [RNN]      → rnn.weights.h5
                ├── worker_train_v4.py [LSTM]     → lstm.weights.h5
                ├── worker_train_v4.py [GRU]      → gru.weights.h5
                └── worker_ghs_v4.py  [G-HS+GRU] → ghs_cnn_gru.weights.h5
================================================================================
"""

import os, sys, json, time, subprocess, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ==============================================================================
# 1) ORTAK VERİ HAZIRLAMA
# ==============================================================================
GENERATION_FILE = "Plant_1_Generation_Data.csv"
WEATHER_FILE    = "Plant_1_Weather_Sensor_Data.csv"
DATA_DIR        = "ghs_data"
WEIGHTS_DIR     = "ghs_weights"
os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

WINDOW_SIZE = 96    # 24 saat × 4 (15 dk ölçüm)
TARGET_COL  = "DC_POWER"
HORIZON     = 1
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
SEED        = 42
np.random.seed(SEED)

print("=" * 72)
print("  VERİ HAZIRLAMA")
print("=" * 72)

gen_df     = pd.read_csv(GENERATION_FILE)
weather_df = pd.read_csv(WEATHER_FILE)
gen_df["DATE_TIME"]     = pd.to_datetime(gen_df["DATE_TIME"],     dayfirst=True)
weather_df["DATE_TIME"] = pd.to_datetime(weather_df["DATE_TIME"], dayfirst=True)

# Aynı tesisteki çoklu inverter/sensör satırlarını birleştir
gen_agg = gen_df.groupby("DATE_TIME", as_index=False).agg(
    DC_POWER    =("DC_POWER",    "sum"),
    AC_POWER    =("AC_POWER",    "sum"),
    DAILY_YIELD =("DAILY_YIELD", "sum"),
)
weather_agg = weather_df.groupby("DATE_TIME", as_index=False).agg(
    AMBIENT_TEMPERATURE =("AMBIENT_TEMPERATURE", "mean"),
    MODULE_TEMPERATURE  =("MODULE_TEMPERATURE",  "mean"),
    IRRADIATION         =("IRRADIATION",         "mean"),
)
df = pd.merge(gen_agg, weather_agg, on="DATE_TIME", how="inner")
df = df.sort_values("DATE_TIME").reset_index(drop=True)

# Döngüsel zaman kodlaması
df["hour"]        = df["DATE_TIME"].dt.hour + df["DATE_TIME"].dt.minute / 60.0
df["day_of_year"] = df["DATE_TIME"].dt.dayofyear
df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
df["doy_sin"]     = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
df["doy_cos"]     = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

# Sahte sıfır temizliği: ışınım varken DC_POWER=0 → sensör hatası
df[TARGET_COL] = df[TARGET_COL].where(df[TARGET_COL] >= 0, np.nan)
if "IRRADIATION" in df.columns:
    sahte_sifir = (df["IRRADIATION"] > 0.01) & (df[TARGET_COL] == 0)
    df.loc[sahte_sifir, TARGET_COL] = np.nan
    print(f"  Sahte sıfır → NaN: {sahte_sifir.sum()} satır")

# Yalnızca NaN'leri interpolasyon ile doldur (gerçek gece sıfırlarına dokunma)
df[TARGET_COL] = df[TARGET_COL].interpolate(method="linear", limit_direction="both")
df = df.ffill().bfill().dropna().reset_index(drop=True)

FEATURE_COLS = [c for c in [
    "DC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
    "IRRADIATION", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
] if c in df.columns]

print(f"  Özellikler    : {FEATURE_COLS}")

data        = df[FEATURE_COLS].values.astype(np.float32)
scaler      = MinMaxScaler((0, 1))
data_scaled = scaler.fit_transform(data)
target_idx  = FEATURE_COLS.index(TARGET_COL)
n_features  = data_scaled.shape[1]

# Kayan pencere → 3D tensör
def create_windows(data, ws, ti, h=1):
    X, y = [], []
    for i in range(len(data) - ws - h + 1):
        X.append(data[i : i + ws])
        y.append(data[i + ws + h - 1, ti])
    return np.array(X, np.float32), np.array(y, np.float32)

X, y    = create_windows(data_scaled, WINDOW_SIZE, target_idx, HORIZON)
n       = len(X)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * VAL_RATIO)

X_train, y_train = X[:n_train],                  y[:n_train]
X_val,   y_val   = X[n_train : n_train+n_val],   y[n_train : n_train+n_val]
X_test,  y_test  = X[n_train+n_val:],             y[n_train+n_val:]

# Alt süreçlere ilet
for name, arr in [
    ("X_train", X_train), ("y_train", y_train),
    ("X_val",   X_val),   ("y_val",   y_val),
    ("X_test",  X_test),  ("y_test",  y_test),
    ("scaler_min",   scaler.data_min_),
    ("scaler_scale", scaler.scale_),
]:
    np.save(f"{DATA_DIR}/{name}.npy", arr)

meta = {"window_size": WINDOW_SIZE, "n_features": n_features,
        "target_idx": target_idx, "feature_cols": FEATURE_COLS}
with open(f"{DATA_DIR}/meta.json", "w") as f:
    json.dump(meta, f)

print(f"  Veri kaydedildi → {DATA_DIR}/")
print(f"  Train:{X_train.shape}  Val:{X_val.shape}  Test:{X_test.shape}")

# ==============================================================================
# 2) WORKER DOSYALARI
# ==============================================================================

# ─── Worker 1: Standart Modeller ─────────────────────────────────────────────
WORKER_STANDARD = r'''"""
worker_train_v4.py  –  Standart model eğitimi (DirectML uyumlu)

Komut: python worker_train_v4.py <model_adi>
  model_adi: cnn | rnn | lstm | gru

DirectML Uyumluluk Notu:
  GRU ve LSTM katmanlarında implementation=1 kullanılmaktadır.
  Bu parametre Keras'ın CudnnRNN yolu yerine pure-TF element-wise
  ops yolunu seçmesini zorlar.  DirectML CudnnRNN'i desteklemez!
"""
import os, sys, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]    = "3"
os.environ["TF_DIRECTML_DEBUG_LAYER"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, SimpleRNN,
    Conv1D, MaxPooling1D, Flatten,
    Dropout, BatchNormalization, Input,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# GPU: dinamik bellek büyümesi (GTX1650 için zorunlu)
for gpu in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(gpu, True)
    except: pass
tf.random.set_seed(42); np.random.seed(42)

DATA_DIR    = "ghs_data"
WEIGHTS_DIR = "ghs_weights"
MODEL_NAME  = sys.argv[1]
WINDOW_SIZE = 96
# DirectML'de küçük batch DirectX descriptor leak'e yol açar.
# 32 → hem bellek güvenli hem de hızlı.
BATCH_SIZE  = 32
EPOCHS      = 60

X_train = np.load(f"{DATA_DIR}/X_train.npy")
y_train = np.load(f"{DATA_DIR}/y_train.npy")
X_val   = np.load(f"{DATA_DIR}/X_val.npy")
y_val   = np.load(f"{DATA_DIR}/y_val.npy")
n_feat  = X_train.shape[2]

# tf.data pipeline: önbellek + önceden getirme
def make_ds(X, y, bs, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle: ds = ds.shuffle(8192, seed=42)
    return ds.batch(bs).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(X_train, y_train, BATCH_SIZE, shuffle=True)
val_ds   = make_ds(X_val,   y_val,   BATCH_SIZE)

def build(name, ws, nf):
    m = Sequential(name=name)
    m.add(Input(shape=(ws, nf)))

    if name == "cnn":
        # 1D-CNN: iki konvolüsyon bloğu + düzleştirme
        m.add(Conv1D(32, 3, activation="relu", padding="same"))
        m.add(BatchNormalization()); m.add(MaxPooling1D(2))
        m.add(Conv1D(16, 3, activation="relu", padding="same"))
        m.add(BatchNormalization()); m.add(MaxPooling1D(2))
        m.add(Flatten())
        m.add(Dense(32, activation="relu")); m.add(Dropout(0.2))

    elif name == "rnn":
        # SimpleRNN: kaybolan gradyan problemi nedeniyle zayıf baz model
        # clipnorm=1.0 → gradyan patlamasını engeller (RNN'nin ana problemi)
        m.add(SimpleRNN(32, return_sequences=True)); m.add(Dropout(0.2))
        m.add(SimpleRNN(16, return_sequences=False)); m.add(Dropout(0.2))
        m.add(Dense(16, activation="relu"))

    elif name == "lstm":
        # LSTM: implementation=1 → DirectML'de CudnnRNN'yi ATLAR
        # implementation=2 (varsayılan) → CudnnRNN çağırır → HATA!
        m.add(LSTM(32, return_sequences=True,  implementation=1, recurrent_dropout=0.2))
        m.add(LSTM(16, return_sequences=False, implementation=1, recurrent_dropout=0.2))
        m.add(Dense(16, activation="relu"))

    elif name == "gru":
        # GRU: implementation=1 → DirectML uyumlu
        m.add(GRU(32, return_sequences=True,  implementation=1, reset_after=False, recurrent_dropout=0.2))
        m.add(GRU(16, return_sequences=False, implementation=1, reset_after=False, recurrent_dropout=0.2))
        m.add(Dense(16, activation="relu"))

    m.add(Dense(1))
    # clipnorm=1.0: gradyan normunu 1'de kırpar → RNN/LSTM/GRU için kritik
    m.compile(Adam(1e-3, clipnorm=1.0), "mse")
    return m

model = build(MODEL_NAME, WINDOW_SIZE, n_feat)
cbs = [
    EarlyStopping(monitor="val_loss", patience=10,
                  restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=5, min_lr=1e-6, verbose=0),
]
model.fit(train_ds, validation_data=val_ds,
          epochs=EPOCHS, callbacks=cbs, verbose=0)

out = f"{WEIGHTS_DIR}/{MODEL_NAME}.weights.h5"
model.save_weights(out)
print(f"SAVED:{out}")
'''

# ─── Worker 2: G-HS Optimizasyon + CNN-GRU ───────────────────────────────────
WORKER_GHS = r'''"""
worker_ghs_v4.py  –  G-HS Optimizasyon + CNN-GRU Eğitimi (DirectML uyumlu)

Düzeltme: Tüm GRU katmanlarına implementation=1 eklendi.
          Bu CudnnRNN hatasını ortadan kaldırır.
"""
import os, sys, json, warnings, time, gc
os.environ["TF_CPP_MIN_LOG_LEVEL"]    = "3"
os.environ["TF_DIRECTML_DEBUG_LAYER"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GRU, Conv1D, MaxPooling1D,
    Dropout, Input, BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

for gpu in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(gpu, True)
    except: pass
tf.random.set_seed(42); np.random.seed(42)

DATA_DIR    = "ghs_data"
WEIGHTS_DIR = "ghs_weights"
WINDOW_SIZE = 96

X_train = np.load(f"{DATA_DIR}/X_train.npy")
y_train = np.load(f"{DATA_DIR}/y_train.npy")
X_val   = np.load(f"{DATA_DIR}/X_val.npy")
y_val   = np.load(f"{DATA_DIR}/y_val.npy")
n_feat  = X_train.shape[2]

def make_ds(X, y, bs, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle: ds = ds.shuffle(8192, seed=42)
    return ds.batch(bs).prefetch(tf.data.AUTOTUNE)

def build_model(ws, nf, cnn_filters=32, kernel_size=3,
                gru_units=32, dropout_rate=0.2, lr=1e-3):
    """
    Hibrit CNN-GRU modeli.
    ÖNEMLİ: GRU katmanlarında implementation=1 kullanılır.
    implementation=2 (varsayılan) → CudnnRNN → DirectML'de çalışmaz!
    implementation=1 → element-wise pure-TF ops → DirectML uyumlu ✓
    """
    inp = Input(shape=(ws, nf))

    # CNN bloğu: yerel zaman örüntülerini öğrenir
    x = Conv1D(cnn_filters, kernel_size, activation="relu",
               padding="same")(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(dropout_rate)(x)
    x = Conv1D(max(cnn_filters // 2, 8), kernel_size,
               activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # GRU bloğu: uzun vadeli bağımlılıkları öğrenir
    # implementation=1  →  DirectML uyumlu  ✓
    x = GRU(gru_units, return_sequences=True,
            implementation=1, reset_after=False, recurrent_dropout=dropout_rate)(x)
    x = GRU(max(gru_units // 2, 8), return_sequences=False,
            implementation=1, reset_after=False, recurrent_dropout=dropout_rate)(x)

    x   = Dense(16, activation="relu")(x)
    out = Dense(1)(x)

    m = Model(inp, out, name="G_HS_CNN_GRU")
    # clipnorm=1.0: gradyan patlamasını önler
    m.compile(Adam(lr, clipnorm=1.0), "mse")
    return m

def evaluate_params(p, epochs=12):
    """
    Verilen hiper-parametre sözlüğüyle kısa eğitim yapar.
    K.clear_session() + gc.collect(): DirectML bellek temizliği.
    """
    try:
        K.clear_session(); gc.collect()
        bs    = max(int(p["batch_size"]), 16)
        ds_tr = make_ds(X_train, y_train, bs, shuffle=True)
        ds_va = make_ds(X_val,   y_val,   bs)
        m = build_model(
            WINDOW_SIZE, n_feat,
            cnn_filters  = int(p["cnn_filters"]),
            kernel_size  = int(p["kernel_size"]),
            gru_units    = int(p["gru_units"]),
            dropout_rate = float(p["dropout"]),
            lr           = float(p["lr"]),
        )
        es   = EarlyStopping(monitor="val_loss", patience=4,
                             restore_best_weights=True, verbose=0)
        hist = m.fit(ds_tr, validation_data=ds_va,
                     epochs=epochs, callbacks=[es], verbose=0)
        score = float(min(hist.history["val_loss"]))
        del m; K.clear_session(); gc.collect()
        return score
    except Exception as e:
        print(f"EVAL_ERROR:{e}", flush=True)
        K.clear_session(); gc.collect()
        return 999.0

# ── Hiper-Parametre Arama Uzayı ───────────────────────────────────────────────
# (alt_sınır, üst_sınır, tam_sayı_mı?)
SPACE = {
    "lr"          : (1e-4, 8e-3,  False),
    "cnn_filters" : (16,   64,    True),
    "kernel_size" : (2,    5,     True),
    "gru_units"   : (16,   64,    True),
    "dropout"     : (0.05, 0.40,  False),
    "batch_size"  : (16,   64,    True),
}
KEYS = list(SPACE.keys())

def rand_h():
    """Arama uzayından düzgün rassal harmoni vektörü."""
    h = []
    for k in KEYS:
        lb, ub, is_int = SPACE[k]
        v = np.random.uniform(lb, ub)
        h.append(int(round(v)) if is_int else v)
    return h

def clip_h(h):
    """Harmoniyi sınırlar içinde tut."""
    out = []
    for i, k in enumerate(KEYS):
        lb, ub, is_int = SPACE[k]
        v = float(np.clip(float(h[i]), lb, ub))
        out.append(int(round(v)) if is_int else v)
    return out

def h2p(h):
    """Harmoni vektörünü sözlüğe çevir."""
    return {k: h[i] for i, k in enumerate(KEYS)}

def centroid(HM):
    """Harmoni belleğinin ağırlık merkezini hesapla."""
    return [np.mean([float(HM[j][i]) for j in range(len(HM))])
            for i in range(len(KEYS))]

# ── G-HS Algoritma Parametreleri ─────────────────────────────────────────────
# HMS=6 → Küçük bellek, DirectML'de az model eşzamanlı tutulur
# NI=15 → 15 iterasyon × HMS=6 = 90 değerlendirme (makul süre)
HMS     = 6       # Harmoni Belleği Boyutu
NI      = 15      # İterasyon Sayısı
HMCR    = 0.90    # Bellek Göz Önüne Alma Oranı
PAR_MIN = 0.30    # Başlangıç Perde Ayarlama Oranı (keşif)
PAR_MAX = 0.92    # Bitiş PAR (istismar)
BW_MIN  = 5e-5    # Minimum Bant Genişliği
BW_MAX  = 0.12    # Maksimum Bant Genişliği
HS_EVAL_EPOCHS = 12   # Kısa değerlendirme epoch'u

print("GHS_START", flush=True)
t0 = time.time()

# Harmoni belleğini başlat
HM     = [rand_h() for _ in range(HMS)]
scores = [evaluate_params(h2p(h), HS_EVAL_EPOCHS) for h in HM]
bi     = int(np.argmin(scores))
best_score = scores[bi]
best_h     = list(HM[bi])
cen        = centroid(HM)
print(f"GHS_INIT:{best_score:.6f}", flush=True)

# Ana G-HS döngüsü
for t in range(1, NI + 1):
    # Dinamik PAR: iterasyonla doğrusal artar (keşif → istismar)
    PAR = PAR_MIN + (PAR_MAX - PAR_MIN) * t / NI

    # Dinamik BW: üstel azalma (kaba → ince arama)
    BW  = BW_MAX * (BW_MIN / BW_MAX) ** (t / NI)

    new_h = []
    for i, k in enumerate(KEYS):
        lb, ub, is_int = SPACE[k]
        r1 = np.random.rand()

        if r1 <= HMCR:
            # Bellekten seç + PAR olasılığıyla perde ayarlama
            note = float(HM[np.random.randint(HMS)][i])
            if np.random.rand() <= PAR:
                note += BW * (ub - lb) * np.random.uniform(-1, 1)
        else:
            # HMCR başarısız → OBL (Zıtlık Tabanlı Öğrenme)
            # Rassal değer üret, arama uzayı merkezine göre zıttını hesapla.
            # Her ikisini ağırlık merkezine göre karşılaştır;
            # merkeze daha yakın olan aday seçilir.
            x    = np.random.uniform(lb, ub)
            x_op = lb + ub - x         # OBL formülü: x_min + x_max - x
            c    = float(cen[i])
            # Merkeze (cen) daha yakın olanı seç → arama uzayında daha verimli konum
            note = x if abs(x - c) <= abs(x_op - c) else x_op

        note = float(np.clip(note, lb, ub))
        new_h.append(int(round(note)) if is_int else note)

    ns = evaluate_params(h2p(new_h), HS_EVAL_EPOCHS)

    # En kötü harmoniyi yeni ile değiştir (eğer daha iyiyse)
    wi = int(np.argmax(scores))
    if ns < scores[wi]:
        HM[wi]     = new_h
        scores[wi] = ns
        if ns < best_score:
            best_score = ns
            best_h     = list(new_h)
        cen = centroid(HM)

    elapsed = time.time() - t0
    print(f"GHS_ITER:{t}/{NI}:{ns:.6f}:{best_score:.6f}:{elapsed:.0f}",
          flush=True)

best_params = h2p(best_h)
print(f"GHS_BEST:{json.dumps(best_params)}", flush=True)

# ── Son (Tam) Eğitim – Optimal Parametrelerle ─────────────────────────────────
print("GHS_FINAL_TRAIN", flush=True)
K.clear_session(); gc.collect()

bs_final = max(int(best_params["batch_size"]), 16)
ds_tr    = make_ds(X_train, y_train, bs_final, shuffle=True)
ds_va    = make_ds(X_val,   y_val,   bs_final)

m_final = build_model(
    WINDOW_SIZE, n_feat,
    cnn_filters  = int(best_params["cnn_filters"]),
    kernel_size  = int(best_params["kernel_size"]),
    gru_units    = int(best_params["gru_units"]),
    dropout_rate = float(best_params["dropout"]),
    lr           = float(best_params["lr"]),
)
cbs = [
    EarlyStopping(monitor="val_loss", patience=15,
                  restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=6, min_lr=1e-6, verbose=0),
]
m_final.fit(ds_tr, validation_data=ds_va,
            epochs=120, callbacks=cbs, verbose=0)

out = f"{WEIGHTS_DIR}/ghs_cnn_gru.weights.h5"
m_final.save_weights(out)
print(f"SAVED:{out}", flush=True)
'''

# ─── Worker 3: Tahmin Üretici ─────────────────────────────────────────────────
WORKER_PREDICT = r'''"""
worker_predict_v4.py  –  Tahmin üretici (DirectML uyumlu)

Tüm GRU/LSTM katmanlarında implementation=1 kullanılır.
"""
import os, sys, json, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]    = "3"
os.environ["TF_DIRECTML_DEBUG_LAYER"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, SimpleRNN,
    Conv1D, MaxPooling1D, Flatten,
    Dropout, BatchNormalization, Input,
)
from tensorflow.keras.optimizers import Adam

for gpu in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(gpu, True)
    except: pass

DATA_DIR    = "ghs_data"
WEIGHTS_DIR = "ghs_weights"
MODEL_NAME  = sys.argv[1]     # cnn | rnn | lstm | gru | ghs
WINDOW_SIZE = 96

X_test = np.load(f"{DATA_DIR}/X_test.npy")
n_feat = X_test.shape[2]
bs     = 64

test_ds = (tf.data.Dataset
           .from_tensor_slices((X_test, np.zeros(len(X_test), np.float32)))
           .batch(bs).prefetch(tf.data.AUTOTUNE))

def build_standard(name, ws, nf):
    m = Sequential(name=name)
    m.add(Input(shape=(ws, nf)))
    if name == "cnn":
        m.add(Conv1D(32, 3, activation="relu", padding="same"))
        m.add(BatchNormalization()); m.add(MaxPooling1D(2))
        m.add(Conv1D(16, 3, activation="relu", padding="same"))
        m.add(BatchNormalization()); m.add(MaxPooling1D(2))
        m.add(Flatten())
        m.add(Dense(32, activation="relu")); m.add(Dropout(0.2))
    elif name == "rnn":
        m.add(SimpleRNN(32, return_sequences=True)); m.add(Dropout(0.2))
        m.add(SimpleRNN(16, return_sequences=False)); m.add(Dropout(0.2))
        m.add(Dense(16, activation="relu"))
    elif name == "lstm":
        # implementation=1 → DirectML uyumlu
        m.add(LSTM(32, return_sequences=True,  implementation=1, recurrent_dropout=0.2))
        m.add(LSTM(16, return_sequences=False, implementation=1, recurrent_dropout=0.2))
        m.add(Dense(16, activation="relu"))
    elif name == "gru":
        # implementation=1 → DirectML uyumlu
        m.add(GRU(32, return_sequences=True,  implementation=1, reset_after=False, recurrent_dropout=0.2))
        m.add(GRU(16, return_sequences=False, implementation=1, reset_after=False, recurrent_dropout=0.2))
        m.add(Dense(16, activation="relu"))
    m.add(Dense(1))
    m.compile(Adam(1e-3), "mse")
    return m

def build_ghs(ws, nf, p):
    inp = Input(shape=(ws, nf))
    x = Conv1D(int(p["cnn_filters"]), int(p["kernel_size"]),
               activation="relu", padding="same")(inp)
    x = BatchNormalization()(x); x = MaxPooling1D(2)(x)
    x = Dropout(float(p["dropout"]))(x)
    x = Conv1D(max(int(p["cnn_filters"]) // 2, 8), int(p["kernel_size"]),
               activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Dropout(float(p["dropout"]))(x)
    # implementation=1 → DirectML uyumlu
    x = GRU(int(p["gru_units"]), return_sequences=True, implementation=1,
            reset_after=False, recurrent_dropout=float(p["dropout"]))(x)
    x = GRU(max(int(p["gru_units"]) // 2, 8), return_sequences=False,
            implementation=1, reset_after=False, recurrent_dropout=float(p["dropout"]))(x)
    x = Dense(16, activation="relu")(x)
    out = Dense(1)(x)
    m = Model(inp, out)
    m.compile(Adam(1e-3), "mse")
    return m

if MODEL_NAME == "ghs":
    pf = f"{WEIGHTS_DIR}/ghs_best_params.json"
    p  = json.load(open(pf)) if os.path.exists(pf) else {}
    model = build_ghs(WINDOW_SIZE, n_feat, p) if p else build_ghs(WINDOW_SIZE, n_feat, {
        "cnn_filters": 32, "kernel_size": 3, "gru_units": 32, "dropout": 0.2})
    model.load_weights(f"{WEIGHTS_DIR}/ghs_cnn_gru.weights.h5")
else:
    model = build_standard(MODEL_NAME, WINDOW_SIZE, n_feat)
    model.load_weights(f"{WEIGHTS_DIR}/{MODEL_NAME}.weights.h5")

preds = model.predict(test_ds, verbose=0).flatten()
np.save(f"{DATA_DIR}/pred_{MODEL_NAME}.npy", preds)
print(f"PRED_SAVED:{MODEL_NAME}:{len(preds)}", flush=True)
'''

# Worker dosyalarını yaz
workers = {
    "worker_train_v4.py"  : WORKER_STANDARD,
    "worker_ghs_v4.py"    : WORKER_GHS,
    "worker_predict_v4.py": WORKER_PREDICT,
}
for fname, content in workers.items():
    with open(fname, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✓ Worker yazıldı: {fname}")

# ==============================================================================
# 3) SUBPROCESS ÇALIŞTIRICI
# ==============================================================================

def run_subprocess(cmd, label, timeout=10800):
    """
    Alt süreci çalıştırır, stdout'u gerçek zamanlı iletir.

    - Her alt süreç bağımsız D3D12 bağlamı açar.
    - Kapatıldığında DirectML belleği OS tarafından tamamen serbest bırakılır.
    - stderr=DEVNULL: DirectML'nin oluşturduğu gürültülü uyarıları bastırır.
    - timeout=10800: 3 saat (G-HS için güvenli üst sınır).
    """
    print(f"\n{'─'*70}")
    print(f"  ▶  {label}")
    print(f"{'─'*70}")
    t0     = time.time()
    py     = sys.executable
    lines  = []

    proc = subprocess.Popen(
        [py] + cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True, bufsize=1,
    )
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line: print(f"    {line}"); lines.append(line)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"  [ZAMAN AŞIMI] {label} – {timeout}s")

    elapsed = time.time() - t0
    status  = "✓" if proc.returncode == 0 else "✗"
    print(f"  {status}  {label}  [{elapsed:.0f}s]")
    return lines, proc.returncode

# ==============================================================================
# 4) EĞİTİM AŞAMASI
# ==============================================================================
print("\n" + "=" * 72)
print("  EĞİTİM AŞAMASI  (Her model ayrı subprocess – DirectML izolasyonu)")
print("=" * 72)

ghs_history  = []
best_params  = {}

# Standart modeller
for mn, label in [
    ("cnn",  "[1/5] 1D-CNN"),
    ("rnn",  "[2/5] Vanilla RNN"),
    ("lstm", "[3/5] LSTM"),
    ("gru",  "[4/5] GRU"),
]:
    lines, rc = run_subprocess(["worker_train_v4.py", mn], label)
    if rc != 0:
        print(f"  [UYARI] {mn} hata kodu {rc} – devam ediliyor.")

# G-HS
lines, rc = run_subprocess(
    ["worker_ghs_v4.py", "optimize"],
    "[5/5] G-HS Optimizasyon + CNN-GRU",
)
for line in lines:
    if line.startswith("GHS_ITER:"):
        parts = line.split(":")
        try: ghs_history.append(float(parts[3]))
        except: pass
    elif line.startswith("GHS_BEST:"):
        try:
            best_params = json.loads(line.split("GHS_BEST:")[1])
            with open(f"{WEIGHTS_DIR}/ghs_best_params.json", "w") as f:
                json.dump(best_params, f, indent=2)
        except: pass

# ==============================================================================
# 5) TAHMİN AŞAMASI
# ==============================================================================
print("\n" + "=" * 72)
print("  TAHMİN AŞAMASI")
print("=" * 72)

MODEL_MAP = {
    "1D-CNN"     : "cnn",
    "RNN"        : "rnn",
    "LSTM"       : "lstm",
    "GRU"        : "gru",
    "G-HS CNN-GRU": "ghs",
}

for display_name, warg in MODEL_MAP.items():
    wf = (f"{WEIGHTS_DIR}/ghs_cnn_gru.weights.h5"
          if warg == "ghs" else f"{WEIGHTS_DIR}/{warg}.weights.h5")
    if not os.path.exists(wf):
        print(f"  [ATLA] {display_name}: ağırlık dosyası yok ({wf})")
        continue
    run_subprocess(["worker_predict_v4.py", warg], f"Tahmin: {display_name}")

# ==============================================================================
# 6) METRİK HESAPLAMA
# ==============================================================================
print("\n" + "=" * 72)
print("  METRİK HESAPLAMA")
print("=" * 72)

scaler_min   = np.load(f"{DATA_DIR}/scaler_min.npy")
scaler_scale = np.load(f"{DATA_DIR}/scaler_scale.npy")

def inv_target(pred_s):
    """Yalnızca hedef sütunu ters ölçekler (sklearn gerektirmez)."""
    return pred_s.flatten() / scaler_scale[target_idx] + scaler_min[target_idx]

y_test_real = inv_target(np.load(f"{DATA_DIR}/y_test.npy"))

def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mask = y_true > 1.0
    mape = float(np.mean(np.abs(
        (y_true[mask]-y_pred[mask]) / y_true[mask])) * 100
    ) if mask.sum() > 0 else np.nan
    r2   = float(r2_score(y_true, y_pred))
    return rmse, mae, mape, r2

predictions = {}
rows        = []

for display_name, warg in MODEL_MAP.items():
    pf = f"{DATA_DIR}/pred_{warg}.npy"
    if not os.path.exists(pf):
        print(f"  [ATLA] {display_name}: tahmin dosyası yok.")
        continue
    pred_real = inv_target(np.load(pf))
    predictions[display_name] = pred_real
    rmse, mae, mape, r2 = metrics(y_test_real, pred_real)
    rows.append({
        "Model"    : display_name,
        "RMSE (W)" : round(rmse, 2),
        "MAE  (W)" : round(mae,  2),
        "MAPE (%)" : round(mape, 2) if not np.isnan(mape) else "N/A",
        "R²"       : round(r2,   4),
    })
    marker = "  ← Önerilen" if display_name == "G-HS CNN-GRU" else ""
    print(f"  {display_name:<16} RMSE={rmse:>9.2f}  MAE={mae:>9.2f}  "
          f"MAPE={mape:>7.2f}%  R²={r2:.4f}{marker}")

metrics_df = (pd.DataFrame(rows).set_index("Model")
              .sort_values("RMSE (W)"))
print(f"\n{'─'*70}")
print("   MODEL PERFORMANS TABLOSU  –  TEST SETİ")
print(f"{'─'*70}")
print(metrics_df.to_string())
print(f"{'─'*70}")

# ==============================================================================
# 7) GÖRSELLEŞTİRME  (Tez Formatı – Times New Roman – DPI=300)
# ==============================================================================
print("\n" + "=" * 72)
print("  GÖRSELLEŞTİRME")
print("=" * 72)

matplotlib.rcParams.update({
    "font.family"    : "serif",
    "font.serif"     : ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize" : 14, "axes.labelsize" : 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "figure.facecolor": "white",
    "savefig.dpi"    : 300,
})
TNR = {"fontname": "Times New Roman"}

# Son 3 günlük pencere
N_PLOT = min(288, len(y_test_real))
t_ax   = np.arange(N_PLOT) * 0.25      # saat cinsinden eksen
real_p = y_test_real[-N_PLOT:]
ghs_p  = predictions.get("G-HS CNN-GRU", np.zeros(N_PLOT))[-N_PLOT:]
lstm_p = predictions.get("LSTM",         np.zeros(N_PLOT))[-N_PLOT:]

# ─── Grafik 1: Tahmin Karşılaştırması ─────────────────────────────────────────
fig, axes = plt.subplots(
    2, 1, figsize=(16, 11), sharex=True,
    gridspec_kw={"height_ratios": [3, 1.5]},
)
fig.suptitle(
    "Güneş Enerjisi Üretim Tahmini – Son 3 Günlük Test Seti Karşılaştırması",
    fontsize=15, fontweight="bold", **TNR, y=0.98,
)

ax1 = axes[0]
ax1.plot(t_ax, real_p,  color="black",   lw=2.8, zorder=5,
         label="Gerçek Güneş Enerjisi Üretimi")
ax1.plot(t_ax, ghs_p,   color="#E63946", lw=1.9, ls="--", zorder=4,
         label="G-HS CNN-GRU  (Önerilen Model)")
ax1.plot(t_ax, lstm_p,  color="#2196F3", lw=1.6, ls=":",  zorder=3,
         label="Standart LSTM")

ax1.set_ylabel("DC Güç Üretimi (W)", **TNR)
ax1.legend(loc="upper left", framealpha=0.93, edgecolor="gray")
ax1.grid(True, ls="--", lw=0.6, alpha=0.45)
ax1.set_xlim(0, t_ax[-1]); ax1.set_ylim(bottom=0)

# Gece bölgelerini gri arka planla göster (her 24 saatte bir)
for d in range(4):
    night_start = d * 24
    night_end   = night_start + 6   # 00:00–06:00
    if night_start < t_ax[-1]:
        ax1.axvspan(night_start, min(night_end, t_ax[-1]),
                    alpha=0.07, color="navy")

# Gün sınır çizgileri
for d, h in enumerate([24, 48], 2):
    if h < t_ax[-1]:
        ax1.axvline(h, color="gray", ls="-.", lw=0.9, alpha=0.65)
        ax1.text(h + 0.2, ax1.get_ylim()[1]*0.93, f"Gün {d}",
                 fontsize=10, color="gray", **TNR)

# G-HS R² kutusu
if "G-HS CNN-GRU" in metrics_df.index:
    r2_ghs  = metrics_df.loc["G-HS CNN-GRU", "R²"]
    r2_lstm = metrics_df.loc["LSTM",          "R²"] if "LSTM" in metrics_df.index else "–"
    ax1.text(
        0.985, 0.965,
        f"G-HS R² = {r2_ghs:.4f}\nLSTM  R² = {r2_lstm}",
        transform=ax1.transAxes, fontsize=11,
        ha="right", va="top", **TNR,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFF3CD",
                  edgecolor="#E63946", lw=1.3, alpha=0.95),
    )

# ─── Panel 2: Mutlak Hata ─────────────────────────────────────────────────────
ax2 = axes[1]
ax2.fill_between(t_ax, np.abs(real_p - lstm_p), alpha=0.40,
                 color="#2196F3", label="|LSTM Hatası| (W)")
ax2.fill_between(t_ax, np.abs(real_p - ghs_p),  alpha=0.65,
                 color="#E63946", label="|G-HS Hatası| (W)")
ax2.set_xlabel("Zaman (Saat)", **TNR)
ax2.set_ylabel("Mutlak Hata (W)", **TNR)
ax2.legend(loc="upper left", framealpha=0.90, fontsize=10)
ax2.grid(True, ls="--", lw=0.6, alpha=0.45)
ax2.set_ylim(bottom=0)
ax2.set_xticks(np.arange(0, t_ax[-1] + 1, 6))
ax2.set_xticklabels(
    [f"{int(h%24):02d}:00" for h in np.arange(0, t_ax[-1] + 1, 6)],
    **TNR,
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out1 = "solar_forecast_comparison.png"
plt.savefig(out1, dpi=300, bbox_inches="tight")
print(f"  ✓  '{out1}'  kaydedildi (DPI=300)")
plt.show()

# ─── Grafik 2: G-HS Yakınsama ─────────────────────────────────────────────────
if ghs_history:
    fig2, ax = plt.subplots(figsize=(10, 5))
    iters = np.arange(1, len(ghs_history) + 1)
    ax.plot(iters, ghs_history, marker="o", ms=6, lw=2.2,
            color="#E63946", markerfacecolor="white",
            markeredgecolor="#E63946", markeredgewidth=2,
            label="En İyi Doğrulama Kaybı (G-HS)")
    ax.fill_between(iters, ghs_history, alpha=0.12, color="#E63946")
    ax.set_xlabel("İterasyon Numarası", **TNR)
    ax.set_ylabel("Doğrulama Kaybı (MSE)", **TNR)
    ax.set_title(
        "G-HS Algoritması Yakınsama Eğrisi\n"
        "(OBL + Dinamik PAR/BW Modülleri ile)",
        **TNR, fontweight="bold",
    )
    ax.legend(fontsize=10); ax.grid(True, ls="--", alpha=0.50)
    ax.set_xticks(iters)
    plt.tight_layout()
    out2 = "ghs_convergence.png"
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    print(f"  ✓  '{out2}'  kaydedildi (DPI=300)")
    plt.show()

# ─── Grafik 3: Metrik Çubuk Grafikleri ───────────────────────────────────────
if len(rows) >= 2:
    METRICS_BAR = ["RMSE (W)", "MAE  (W)", "R²"]
    model_names = metrics_df.index.tolist()
    short_names = [
        n.replace(" CNN-GRU", "\nCNN-GRU").replace("G-HS", "G-HS") for n in model_names
    ]
    colors = ["#457B9D", "#F4A261", "#2A9D8F", "#E9C46A", "#E63946"]

    fig3, axes3 = plt.subplots(1, 3, figsize=(17, 5.5))
    fig3.suptitle(
        "Model Performans Karşılaştırması – Test Seti",
        **TNR, fontsize=15, fontweight="bold",
    )
    for ax, met in zip(axes3, METRICS_BAR):
        vals = []
        for mn in model_names:
            try:
                v = float(metrics_df.loc[mn, met])
            except (ValueError, KeyError):
                v = 0.0
            vals.append(v)

        bars = ax.bar(
            range(len(model_names)), vals,
            color=colors[:len(model_names)],
            edgecolor="white", linewidth=0.9, alpha=0.92,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.015,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=9.5, **TNR,
            )
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(short_names, fontsize=9, **TNR)
        ax.set_title(met, **TNR, fontsize=13)
        ax.set_ylabel(met, **TNR, fontsize=11)
        ax.grid(True, axis="y", ls=":", alpha=0.50, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(
            bottom=min(vals) * 0.92 if met == "R²" else 0,
            top=max(vals) * 1.13,
        )

    plt.tight_layout(pad=2.0)
    out3 = "model_metrics_comparison.png"
    plt.savefig(out3, dpi=300, bbox_inches="tight")
    print(f"  ✓  '{out3}'  kaydedildi (DPI=300)")
    plt.show()

# ==============================================================================
# ÖZET RAPOR
# ==============================================================================
print("\n" + "=" * 72)
print("  ✅  TÜM İŞLEMLER TAMAMLANDI")
print("=" * 72)

if best_params:
    print("\n  G-HS Optimal Hiper-Parametreler:")
    for k, v in best_params.items():
        fmt = f"{v:.6f}" if isinstance(v, float) else str(v)
        print(f"    {k:<16}: {fmt}")

print(f"\n  Nihai Metrik Tablosu:\n{metrics_df.to_string()}")
print("\n" + "=" * 72)
