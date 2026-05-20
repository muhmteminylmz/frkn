#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  Güneş Enerjisi Üretim Tahmini  –  G-HS-DCN
  [Güncelleme] 15 Dakikalık Yüksek Çözünürlüklü Veri ve Optimize DCN Mimarisi
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

# --- KLASÖR YAPISI ---
INPUT_DIR   = "dataset"
BASE_OUT    = "sonuclar"
DATA_DIR    = os.path.join(BASE_OUT, "processed_data")
WEIGHTS_DIR = os.path.join(BASE_OUT, "weights")
LOG_DIR     = os.path.join(BASE_OUT, "logs")

for d in [DATA_DIR, WEIGHTS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# --- HİPERPARAMETRELER ---
HMS     = 30
NI      = 60      # Optimizasyon iterasyon sayısı
EP_HS   = 35      # Arama sırasındaki hızlı epoch

WINDOW_SIZE = 96  # 15 dakikalık veride 1 tam gün (24 * 4)
TARGET_COL  = "DC_POWER"
HORIZON     = 1
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
SEED        = 42
np.random.seed(SEED)

# ==============================================================================
# BÖLÜM 1 – VERİ HAZIRLAMA (15 Dakikalık Hassasiyet)
# ==============================================================================
print("=" * 72)
print("  BÖLÜM 1 │ YÜKSEK ÇÖZÜNÜRLÜKLÜ VERİ HAZIRLAMA (15 Dk)")
print("=" * 72)

PLANTS = {
    1: (os.path.join(INPUT_DIR, "Plant_1_Generation_Data.csv"), 
        os.path.join(INPUT_DIR, "Plant_1_Weather_Sensor_Data.csv")),
    2: (os.path.join(INPUT_DIR, "Plant_2_Generation_Data.csv"), 
        os.path.join(INPUT_DIR, "Plant_2_Weather_Sensor_Data.csv")),
}

FEATURE_COLS = [
    "DC_POWER", "AC_POWER", "DAILY_YIELD",
    "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION",
    "irradiation_temp", "module_ambient_gap", "is_daylight",
    "dc_roll_mean_4", "dc_roll_mean_16", "irr_roll_mean_4",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "PLANT_ID",
]

def parse_dt(series):
    parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")
    if parsed.isna().any():
        parsed = parsed.where(parsed.notna(), pd.to_datetime(series, errors="coerce"))
    return parsed

def load_plant(gen_f, wth_f):
    gen = pd.read_csv(gen_f)
    wth = pd.read_csv(wth_f)
    
    gen["DATE_TIME"] = parse_dt(gen["DATE_TIME"])
    wth["DATE_TIME"] = parse_dt(wth["DATE_TIME"])
    
    # Inverterleri topluyoruz ki tüm santralin enerjisi tek bir sütunda çıksın (Sum)
    gen_agg = gen.groupby("DATE_TIME", as_index=False)[["DC_POWER", "AC_POWER", "DAILY_YIELD"]].sum()
    # Hava durumu sensörü tek olduğu için ortalama alıyoruz (Mean)
    wth_agg = wth.groupby("DATE_TIME", as_index=False)[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]].mean()
    
    df = pd.merge(gen_agg, wth_agg, on="DATE_TIME", how="inner")
    df = df.sort_values("DATE_TIME").reset_index(drop=True)
    
    # Zaman özellikleri
    df["hour"]        = df["DATE_TIME"].dt.hour + df["DATE_TIME"].dt.minute / 60.0
    df["day_of_year"] = df["DATE_TIME"].dt.dayofyear
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"]     = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"]     = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["irradiation_temp"]  = df["IRRADIATION"] * df["MODULE_TEMPERATURE"]
    df["module_ambient_gap"] = df["MODULE_TEMPERATURE"] - df["AMBIENT_TEMPERATURE"]
    df["is_daylight"]       = (df["IRRADIATION"] > 0).astype(np.float32)
    df["dc_roll_mean_4"]    = df["DC_POWER"].rolling(4, min_periods=1).mean().shift(1)
    df["dc_roll_mean_16"]   = df["DC_POWER"].rolling(16, min_periods=1).mean().shift(1)
    df["irr_roll_mean_4"]   = df["IRRADIATION"].rolling(4, min_periods=1).mean().shift(1)
    
    # Hatalı/Fizik dışı sensör verilerini temizleme
    df[TARGET_COL] = df[TARGET_COL].where(df[TARGET_COL] >= 0, np.nan)
    if "IRRADIATION" in df.columns:
        sahte = (df["IRRADIATION"] > 0.01) & (df[TARGET_COL] == 0)
        df.loc[sahte, TARGET_COL] = np.nan
        
    df = df.interpolate(method="linear", limit_direction="both")
    return df.ffill().bfill().dropna().reset_index(drop=True)

def prepare_plant(gen_f, wth_f, plant_id):
    df = load_plant(gen_f, wth_f)
    df["PLANT_ID"] = float(plant_id)
    fcols   = [c for c in FEATURE_COLS if c in df.columns]
    data    = df[fcols].values.astype(np.float32)
    sc      = MinMaxScaler((0, 1))
    data_sc = sc.fit_transform(data)
    ti      = fcols.index(TARGET_COL)
    
    X_list, y_list = [], []
    for i in range(len(data_sc) - WINDOW_SIZE - HORIZON + 1):
        X_list.append(data_sc[i : i + WINDOW_SIZE])
        y_list.append(data_sc[i + WINDOW_SIZE + HORIZON - 1, ti])
        
    X_all = np.array(X_list, np.float32)
    y_all = np.array(y_list, np.float32)
    n    = len(X_all)
    n_tr = int(n * TRAIN_RATIO)
    n_va = int(n * VAL_RATIO)
    return {
        "X_train"   : X_all[:n_tr], "y_train": y_all[:n_tr],
        "X_val"     : X_all[n_tr:n_tr+n_va], "y_val": y_all[n_tr:n_tr+n_va],
        "X_test"    : X_all[n_tr+n_va:], "y_test": y_all[n_tr+n_va:],
        "sc_min"    : float(sc.data_min_[ti]),
        "sc_scale"  : float(sc.scale_[ti]),
        "n_features": data_sc.shape[1],
    }

available = {pid: paths for pid, paths in PLANTS.items() 
             if os.path.exists(paths[0]) and os.path.exists(paths[1])}
if not available:
    sys.exit("[HATA] 'dataset' klasöründe CSV dosyaları bulunamadı!")

plant_data = {}
for pid, (gf, wf) in available.items():
    plant_data[pid] = prepare_plant(gf, wf, pid - 1)
    p = plant_data[pid]
    print(f"  Plant {pid}: Train={p['X_train'].shape[0]}  Val={p['X_val'].shape[0]}  Test={p['X_test'].shape[0]}")

def concat_splits(key):
    return np.concatenate([plant_data[pid][key] for pid in sorted(plant_data)])

X_train = concat_splits("X_train"); y_train = concat_splits("y_train")
X_val   = concat_splits("X_val");   y_val   = concat_splits("y_val")
X_test  = concat_splits("X_test");  y_test  = concat_splits("y_test")
n_features = X_train.shape[2]

y_val_real_parts = []
y_test_real_parts = []
for pid in sorted(plant_data):
    p    = plant_data[pid]
    val_part = p["y_val"].flatten() / p["sc_scale"] + p["sc_min"]
    part = p["y_test"].flatten() / p["sc_scale"] + p["sc_min"]
    y_val_real_parts.append(val_part)
    y_test_real_parts.append(part)
y_val_real = np.concatenate(y_val_real_parts)
y_test_real = np.concatenate(y_test_real_parts)

print(f"\n  Birleşik Veri: Train={X_train.shape[0]}  Val={X_val.shape[0]}  Test={X_test.shape[0]}  Özellik={n_features}")

for name, arr in [("X_train",X_train), ("y_train",y_train), ("X_val",X_val), ("y_val",y_val), 
                  ("X_test",X_test), ("y_test",y_test), ("y_val_real", y_val_real),
                  ("y_test_real", y_test_real)]:
    np.save(os.path.join(DATA_DIR, f"{name}.npy"), arr)

inv_info = {
    str(pid): {
        "sc_min": plant_data[pid]["sc_min"],
        "sc_scale": plant_data[pid]["sc_scale"],
        "val_size": int(plant_data[pid]["X_val"].shape[0]),
        "test_size": int(plant_data[pid]["X_test"].shape[0]),
    }
    for pid in plant_data
}
with open(os.path.join(DATA_DIR, "inv_info.json"), "w") as f:
    json.dump(inv_info, f, indent=2)
with open(os.path.join(DATA_DIR, "feature_index.json"), "w") as f:
    json.dump({c: i for i, c in enumerate(FEATURE_COLS)}, f, indent=2)

# ==============================================================================
# BÖLÜM 2 – WORKER DOSYALARI (TÜMÜ GPU DESTEKLİ)
# ==============================================================================
WORKER_STD_CNN = f'''
import os, warnings; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; warnings.filterwarnings("ignore")
import numpy as np, tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass
tf.random.set_seed(42); np.random.seed(42)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

D="{DATA_DIR}"; W="{WEIGHTS_DIR}"; WS={WINDOW_SIZE}; BS=32
Xtr=np.load(f"{{D}}/X_train.npy"); ytr=np.load(f"{{D}}/y_train.npy")
Xv =np.load(f"{{D}}/X_val.npy");   yv =np.load(f"{{D}}/y_val.npy")
nf=Xtr.shape[2]

m=Sequential([Input(shape=(WS,nf)), Conv1D(64,3,activation="relu",padding="same"),BatchNormalization(),MaxPooling1D(2),
              Conv1D(32,3,activation="relu",padding="same"),BatchNormalization(),MaxPooling1D(2),
              Flatten(),Dense(32,activation="relu"),Dropout(0.2),Dense(1)], name="STD_CNN")
m.compile(Adam(1e-3),"mse")
m.fit(Xtr,ytr,batch_size=BS,validation_data=(Xv,yv),epochs=100,
      callbacks=[EarlyStopping("val_loss",patience=10,restore_best_weights=True)],verbose=0)
m.save_weights(f"{{W}}/cnn.weights.h5")
print("SAVED:cnn",flush=True)
'''

WORKER_STD_RNN = f'''
import os, sys, warnings; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; warnings.filterwarnings("ignore")
import numpy as np, tensorflow as tf

# --- CPU kısıtlaması kaldırıldı, GPU bellek büyümesi açıldı ---
for g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

tf.random.set_seed(42); np.random.seed(42)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, SimpleRNN, LSTM, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

D="{DATA_DIR}"; W="{WEIGHTS_DIR}"; MN=sys.argv[1]; WS={WINDOW_SIZE}; BS=32
Xtr=np.load(f"{{D}}/X_train.npy"); ytr=np.load(f"{{D}}/y_train.npy")
Xv =np.load(f"{{D}}/X_val.npy");   yv =np.load(f"{{D}}/y_val.npy")
nf=Xtr.shape[2]

l_map={{"rnn":[SimpleRNN(32,return_sequences=False)],"lstm":[LSTM(32,return_sequences=False)],"gru":[GRU(32,return_sequences=False)]}}
m=Sequential([Input(shape=(WS,nf))]+l_map[MN]+[Dropout(0.2),Dense(1)],name=MN.upper())
m.compile(Adam(1e-3),"mse")
m.fit(Xtr,ytr,batch_size=BS,validation_data=(Xv,yv),epochs=100,
      callbacks=[EarlyStopping("val_loss",patience=10,restore_best_weights=True)],verbose=0)
m.save_weights(f"{{W}}/{{MN}}.weights.h5")
print(f"SAVED:{{MN}}",flush=True)
'''

WORKER_GHS_DCN = f'''
import os, sys, json, warnings, time, gc; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"; warnings.filterwarnings("ignore")
import numpy as np, tensorflow as tf

# --- GPU bellek yönetimi aktif ---
for g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Add, Lambda, Multiply
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, SpatialDropout1D, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

D="{DATA_DIR}"; W="{WEIGHTS_DIR}"; WS={WINDOW_SIZE}
Xtr=np.load(f"{{D}}/X_train.npy"); ytr=np.load(f"{{D}}/y_train.npy"); Xv=np.load(f"{{D}}/X_val.npy"); yv=np.load(f"{{D}}/y_val.npy")
yv_real=np.load(f"{{D}}/y_val_real.npy")
inv_info=json.load(open(f"{{D}}/inv_info.json"))
feat_idx=json.load(open(f"{{D}}/feature_index.json"))
nf=Xtr.shape[2]
target_idx=int(feat_idx["DC_POWER"]); irr_idx=int(feat_idx["IRRADIATION"]); day_idx=int(feat_idx["is_daylight"])

def inv_split(p_sc, split):
    segs=[]; idx=0
    for pid in sorted(int(k) for k in inv_info):
        inf=inv_info[str(pid)]; n=int(inf[f"{{split}}_size"])
        segs.append(p_sc[idx:idx+n] / float(inf["sc_scale"]) + float(inf["sc_min"])); idx += n
    return np.concatenate(segs)

def apply_postprocess(p_sc, X_sc, alpha=1.0, gate=0.0):
    out = alpha * p_sc + (1.0 - alpha) * X_sc[:, -1, target_idx]
    out = np.where((X_sc[:, -1, day_idx] < 0.5) | (X_sc[:, -1, irr_idx] <= gate), 0.0, out)
    return np.clip(out, 0.0, None)

def tune_postprocess(pred_sc, X_sc):
    best_rmse, best_cfg = 1e18, {{"alpha": 1.0, "gate": 0.0}}
    for alpha in np.linspace(0.6, 1.0, 5):
        for gate in np.linspace(0.0, 0.04, 9):
            cand = inv_split(apply_postprocess(pred_sc, X_sc, float(alpha), float(gate)), "val")
            rmse = float(np.sqrt(np.mean((yv_real - cand) ** 2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_cfg = {{"alpha": float(alpha), "gate": float(gate)}}
    return best_rmse, best_cfg

def se_block(x, filters):
    s = GlobalAveragePooling1D()(x)
    s = Dense(max(filters // 8, 8), activation="relu")(s)
    s = Dense(filters, activation="sigmoid")(s)
    s = Lambda(lambda t: tf.expand_dims(t, 1))(s)
    return Multiply()([x, s])

def build_dcn(filters=96, kernel_size=3, n_layers=6, dropout=0.15, lr=7e-4):
    inp = Input(shape=(WS, nf))
    x = LayerNormalization()(Conv1D(filters, 1, padding="causal")(inp))
    for i in range(n_layers):
        res = x
        dil = 2**i
        x = Conv1D(filters, kernel_size, padding="causal", dilation_rate=dil, activation="swish")(x)
        x = LayerNormalization()(x)
        x = SpatialDropout1D(dropout)(x)
        x = Conv1D(filters, kernel_size, padding="causal", dilation_rate=dil, activation="swish")(x)
        x = LayerNormalization()(x)
        x = se_block(x, filters)
        x = Add()([x, res])
    x_last = Lambda(lambda t: t[:, -1, :])(x)
    x = Concatenate()([x_last, GlobalAveragePooling1D()(x), GlobalMaxPooling1D()(x)])
    x = Dense(max(filters, 64), activation="swish")(x)
    x = Dropout(dropout)(x)
    x = Dense(max(filters // 2, 32), activation="swish")(x)
    x = Dropout(dropout / 2)(x)
    m = Model(inp, Dense(1)(x), name="G_HS_DCN")
    m.compile(Adam(learning_rate=lr), tf.keras.losses.Huber())
    return m

def eval_m(p):
    K.clear_session(); gc.collect()
    m = build_dcn(int(p["f"]), int(p["k"]), int(p["l"]), float(p["d"]), float(p["lr"]))
    h = m.fit(Xtr, ytr, batch_size=int(p["b"]), validation_data=(Xv, yv),
              epochs={EP_HS},
              callbacks=[EarlyStopping("val_loss", patience=6, restore_best_weights=True),
                         ReduceLROnPlateau("val_loss", patience=3, factor=0.5, min_lr=1e-5)],
              verbose=0)
    pred_v = m.predict(Xv, batch_size=256, verbose=0).flatten()
    score, _ = tune_postprocess(pred_v, Xv)
    return score

SPACE = {{"lr":(3e-4,2e-3,False), "f":(64,128,True), "k":(2,4,True), "l":(4,6,True), "d":(0.08,0.22,False), "b":(16,64,True)}}
KEYS = list(SPACE.keys())
def rh(): return [int(round(np.random.uniform(lb, ub))) if ii else float(np.random.uniform(lb, ub)) for lb, ub, ii in SPACE.values()]
def h2p(h): return {{k: h[i] for i, k in enumerate(KEYS)}}

HMS=30; NI={NI}; HM = [rh() for _ in range(HMS)]
scores = [eval_m(h2p(h)) for h in HM]
best_s = min(scores); best_h = HM[np.argmin(scores)]

for t in range(1, NI + 1):
    nh = []
    for i, k in enumerate(KEYS):
        lb, ub, ii = SPACE[k]
        if np.random.rand() <= 0.85: note = HM[np.random.randint(HMS)][i] + 0.05 * (ub - lb) * np.random.uniform(-1, 1)
        else: note = np.random.uniform(lb, ub)
        note = float(np.clip(note, lb, ub)); nh.append(int(round(note)) if ii else note)
    ns = eval_m(h2p(nh)); wi = int(np.argmax(scores))
    if ns < scores[wi]:
        HM[wi] = nh; scores[wi] = ns
        if ns < best_s: best_s = ns; best_h = list(nh)
    print(f"GHS_ITER:{{t}}/{{NI}}:{{ns:.6f}}:{{best_s:.6f}}", flush=True)

bp = h2p(best_h); print(f"GHS_BEST:{{json.dumps(bp)}}", flush=True)

m = build_dcn(int(bp["f"]), int(bp["k"]), int(bp["l"]), float(bp["d"]), float(bp["lr"]))
m.fit(Xtr, ytr, batch_size=int(bp["b"]), validation_data=(Xv, yv),
      epochs=200,
      callbacks=[EarlyStopping("val_loss", patience=15, restore_best_weights=True),
                 ReduceLROnPlateau("val_loss", patience=5, factor=0.5, min_lr=1e-5)],
      verbose=0)
post_rmse, post_cfg = tune_postprocess(m.predict(Xv, batch_size=256, verbose=0).flatten(), Xv)
with open(f"{{W}}/dcn_postprocess.json", "w") as f: json.dump(post_cfg, f)
print(f"POSTPROC_BEST:{{json.dumps({{'rmse': post_rmse, **post_cfg}})}}", flush=True)
m.save_weights(f"{{W}}/dcn.weights.h5"); print("SAVED:dcn", flush=True)
'''

WORKER_PREDICT = f'''
import sys, json, os, numpy as np, tensorflow as tf

# --- Tahmin aşamasında da GPU kullanımı aktif ---
for g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten, SimpleRNN, LSTM, GRU, Add, Lambda, Multiply
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, SpatialDropout1D, LayerNormalization
MN = sys.argv[1]
D="{DATA_DIR}"; W="{WEIGHTS_DIR}"; WS={WINDOW_SIZE}; Xt=np.load(f"{{D}}/X_test.npy"); nf=Xt.shape[2]
feat_idx=json.load(open(f"{{D}}/feature_index.json"))
target_idx=int(feat_idx["DC_POWER"]); irr_idx=int(feat_idx["IRRADIATION"]); day_idx=int(feat_idx["is_daylight"])

def b_cnn(): m=Sequential([Input(shape=(WS,nf)),Conv1D(64,3,activation="relu",padding="same"),BatchNormalization(),MaxPooling1D(2),Conv1D(32,3,activation="relu",padding="same"),BatchNormalization(),MaxPooling1D(2),Flatten(),Dense(32,activation="relu"),Dropout(0.2),Dense(1)]); m.compile("adam","mse"); return m
def b_rnn(n): m=Sequential([Input(shape=(WS,nf)), {{"rnn":SimpleRNN,"lstm":LSTM,"gru":GRU}}[n](32,return_sequences=False), Dropout(0.2), Dense(1)]); m.compile("adam","mse"); return m
def se_block(x, filters):
    s=GlobalAveragePooling1D()(x)
    s=Dense(max(filters//8,8),activation="relu")(s)
    s=Dense(filters,activation="sigmoid")(s)
    s=Lambda(lambda t: tf.expand_dims(t,1))(s)
    return Multiply()([x,s])
def b_dcn(p):
    filters, kernel, layers = int(p["f"]), int(p["k"]), int(p["l"])
    dropout = float(p["d"])
    inp = Input(shape=(WS, nf)); x = LayerNormalization()(Conv1D(filters, 1, padding="causal")(inp))
    for i in range(layers):
        res = x; dil = 2**i
        x = Conv1D(filters, kernel, padding="causal", dilation_rate=dil, activation="swish")(x)
        x = LayerNormalization()(x)
        x = SpatialDropout1D(dropout)(x)
        x = Conv1D(filters, kernel, padding="causal", dilation_rate=dil, activation="swish")(x)
        x = LayerNormalization()(x)
        x = se_block(x, filters)
        x = Add()([x, res])
    z = Concatenate()([Lambda(lambda t: t[:, -1, :])(x), GlobalAveragePooling1D()(x), GlobalMaxPooling1D()(x)])
    z = Dense(max(filters, 64), activation="swish")(z)
    z = Dropout(dropout)(z)
    z = Dense(max(filters // 2, 32), activation="swish")(z)
    z = Dropout(dropout / 2)(z)
    m = Model(inp, Dense(1)(z)); m.compile(Adam(learning_rate=float(p["lr"])), tf.keras.losses.Huber()); return m

def apply_postprocess(p_sc, X_sc, cfg):
    alpha = float(cfg.get("alpha", 1.0)); gate = float(cfg.get("gate", 0.0))
    out = alpha * p_sc + (1.0 - alpha) * X_sc[:, -1, target_idx]
    out = np.where((X_sc[:, -1, day_idx] < 0.5) | (X_sc[:, -1, irr_idx] <= gate), 0.0, out)
    return np.clip(out, 0.0, None)

if MN=="cnn": m=b_cnn()
elif MN in ["rnn","lstm","gru"]: m=b_rnn(MN)
elif MN=="dcn": m=b_dcn(json.load(open(f"{{W}}/dcn_best_params.json")))

m.load_weights(f"{{W}}/{{MN}}.weights.h5")
pred = m.predict(Xt, batch_size=32, verbose=0).flatten()
if MN=="dcn" and os.path.exists(f"{{W}}/dcn_postprocess.json"):
    pred = apply_postprocess(pred, Xt, json.load(open(f"{{W}}/dcn_postprocess.json")))
np.save(f"{{D}}/pred_{{MN}}.npy", pred)
print(f"PRED_SAVED:{{MN}}",flush=True)
'''

for fname, content in [("wk_std_cnn.py", WORKER_STD_CNN), ("wk_std_rnn.py", WORKER_STD_RNN), 
                       ("wk_ghs_dcn.py", WORKER_GHS_DCN), ("wk_predict.py", WORKER_PREDICT)]:
    with open(fname, "w", encoding="utf-8") as f: f.write(content)

# ==============================================================================
# BÖLÜM 3 – EĞİTİM VE TAHMİN
# ==============================================================================
def run(cmd, label):
    logf = os.path.join(LOG_DIR, label.replace("/","_").replace(" ","_") + ".log")
    print(f"  ▶ {label} çalışıyor...")
    with open(logf, "w") as lf:
        proc = subprocess.Popen([sys.executable] + cmd, stdout=subprocess.PIPE, stderr=lf, text=True)
        lines = []
        for line in proc.stdout: lines.append(line.strip())
        proc.wait()
    print(f"    ✓ Tamamlandı." if proc.returncode == 0 else f"    ✗ Hata ({logf})")
    return lines

print("\n" + "=" * 72 + "\n  BÖLÜM 3 │ MODELLERİN EĞİTİLMESİ\n" + "=" * 72)
run(["wk_std_cnn.py"], "CNN")
for m in ["rnn", "lstm", "gru"]: run(["wk_std_rnn.py", m], m.upper())

lines = run(["wk_ghs_dcn.py"], "G-HS-DCN (Önerilen Model)")
for line in lines:
    if line.startswith("GHS_BEST:"):
        with open(os.path.join(WEIGHTS_DIR, "dcn_best_params.json"), "w") as f:
            json.dump(json.loads(line.split("GHS_BEST:")[1]), f)

print("\n" + "=" * 72 + "\n  BÖLÜM 4 │ TAHMİNLERİN OLUŞTURULMASI\n" + "=" * 72)
for m in ["cnn", "rnn", "lstm", "gru", "dcn"]: run(["wk_predict.py", m], f"Predict_{m}")

# ==============================================================================
# BÖLÜM 5 – METRİK HESAPLAMA VE GÖRSELLEŞTİRME
# ==============================================================================
print("\n" + "=" * 72 + "\n  BÖLÜM 5 │ SONUÇLAR VE METRİKLER\n" + "=" * 72)
with open(os.path.join(DATA_DIR, "inv_info.json")) as f: inv_info = json.load(f)

def inv_pred(p_sc):
    segs = []; idx = 0
    for pid in sorted(int(k) for k in inv_info):
        inf = inv_info[str(pid)]; n = inf["test_size"]
        segs.append(p_sc[idx:idx+n] / inf["sc_scale"] + inf["sc_min"]); idx += n
    return np.concatenate(segs)

rows = []
for m, name in zip(["cnn","rnn","lstm","gru","dcn"], ["CNN","RNN","LSTM","GRU","G-HS-DCN"]):
    pf = os.path.join(DATA_DIR, f"pred_{m}.npy")
    if not os.path.exists(pf): continue
    pr = inv_pred(np.load(pf))
    # Eksi değerleri 0'a yuvarlıyoruz (Güneş paneli eksi elektrik üretemez)
    pr = np.clip(pr, 0, None) 
    rmse = np.sqrt(mean_squared_error(y_test_real, pr))
    r2 = r2_score(y_test_real, pr)
    
    mk = "  ◄ Önerilen" if name == "G-HS-DCN" else ""
    rows.append({"Model": name, "RMSE": round(rmse,2), "R²": round(r2,4), "Onerilen": mk})

# Tabloyu oluştur ve ekrana bas
mdf = pd.DataFrame(rows).set_index("Model").sort_values("RMSE")
for idx in mdf.index:
    print(f"{idx:<12} RMSE: {mdf.loc[idx, 'RMSE']:>8.2f}  |  R²: {mdf.loc[idx, 'R²']:.4f} {mdf.loc[idx, 'Onerilen']}")

# Görselleştirme (Son 3 Gün = 96 * 3 = 288 Veri Noktası)
N = min(288, len(y_test_real))
plt.figure(figsize=(14, 5))
plt.title("Gerçek vs Tahmin (Son 3 Gün - 15 Dk Çözünürlük)")
plt.plot(y_test_real[-N:], label="Gerçek Değerler", color="black", linewidth=2)
if os.path.exists(os.path.join(DATA_DIR, "pred_dcn.npy")):
    dcn_pred = np.clip(inv_pred(np.load(os.path.join(DATA_DIR, "pred_dcn.npy"))), 0, None)
    plt.plot(dcn_pred[-N:], label="G-HS-DCN", linestyle="--", color="red", alpha=0.9)
if os.path.exists(os.path.join(DATA_DIR, "pred_lstm.npy")):
    lstm_pred = np.clip(inv_pred(np.load(os.path.join(DATA_DIR, "pred_lstm.npy"))), 0, None)
    plt.plot(lstm_pred[-N:], label="LSTM", linestyle=":", color="blue", alpha=0.7)
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(BASE_OUT, "karsilastirma.png"), dpi=300, bbox_inches="tight")
print(f"\n  ✓ Grafik kaydedildi: {os.path.join(BASE_OUT, 'karsilastirma.png')}")
print("=" * 72)
