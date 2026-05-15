# =========================================================
# Solar Power Generation Time Series Forecasting
# Standart Modeller + G-HS (OBL + Dinamik PAR/BW) Optimize GRU
# =========================================================

import os
import gc
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ---------------------------------------------------------
# 1) Global ayarlar
# ---------------------------------------------------------
@dataclass
class Config:
    # Veri dosyaları (repo kök dizini veya verilen dizin)
    generation_csv: str = "Plant_1_Generation_Data.csv"
    weather_csv: str = "Plant_1_Weather_Sensor_Data.csv"

    # Hedef değişken: DC_POWER (varsayılan) veya TOTAL_YIELD
    target_col: str = "DC_POWER"

    # Zaman serisi pencere ayarları
    window_size: int = 96        # 96 adım ~= 24 saat (15 dk çözünürlükte)
    forecast_horizon: int = 1    # 1 adım ileri tahmin

    # Eğitim ayarları
    seed: int = 42
    epochs: int = 30
    batch_size: int = 32

    # Donanım dostu (GTX1650) G-HS ayarları
    hms: int = 6
    ni: int = 12
    hmcr: float = 0.85
    par_min: float = 0.30
    par_max: float = 0.90
    bw_min: float = 0.001
    bw_max: float = 0.10
    optimize_epochs: int = 12

    # Çıktılar
    result_plot_path: str = "solar_forecast_comparison.png"
    prediction_csv_path: str = "solar_predictions.csv"


@dataclass
class ProposedHP:
    units: int
    dropout: float
    learning_rate: float
    batch_size: int
    conv_filters: int


# ---------------------------------------------------------
# 2) Yardımcı fonksiyonlar
# ---------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_runtime() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"✅ GPU bulundu: {len(gpus)}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("⚡ Mixed precision aktif: mixed_float16")
        except Exception:
            print("ℹ️ Mixed precision etkinleştirilemedi, float32 ile devam")
    else:
        print("⚠️ GPU bulunamadı, CPU ile devam")


def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    return pd.read_csv(path)


def normalize_to_exact_split(df: pd.DataFrame) -> pd.DataFrame:
    """70/15/15 oranını tam sağlamak için satır sayısını 20'nin katına indirir."""
    n = len(df)
    normalized_n = n - (n % 20)
    if normalized_n < 20:
        raise ValueError(f"70/15/15 için yeterli kayıt yok: n={n}")
    if normalized_n != n:
        print(f"ℹ️ Tam 70/15/15 için son {n - normalized_n} satır çıkarıldı")
    return df.iloc[:normalized_n].copy()


def exact_chronological_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_n = int(n * 0.70)
    val_n = int(n * 0.15)
    test_n = int(n * 0.15)
    if train_n + val_n + test_n != n:
        raise ValueError("70/15/15 bölmesi tam sağlanamadı")
    train_df = df.iloc[:train_n].copy()
    val_df = df.iloc[train_n:train_n + val_n].copy()
    test_df = df.iloc[train_n + val_n:].copy()
    return train_df, val_df, test_df


def infer_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    protected = {"DATE_TIME", "SOURCE_KEY", "PLANT_ID", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD", "DC_POWER"}
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # Hedef kolon kalsın, diğer target benzeri kolonlar leakage olmasın
    leakage_cols = {"AC_POWER", "DAILY_YIELD", "TOTAL_YIELD", "DC_POWER"} - {target_col}

    features = [c for c in numeric_cols if c not in protected]

    # Hedefi de gecikmeli/past bilgi olarak özelliklere eklemek seride yararlıdır.
    if target_col in numeric_cols:
        features = [target_col] + features

    features = [c for c in features if c not in leakage_cols]

    if not features:
        raise ValueError("Model için kullanılabilir özellik kolonu bulunamadı")

    return features


def create_windows(
    feature_array: np.ndarray,
    target_array: np.ndarray,
    timestamps: np.ndarray,
    window_size: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, ts = [], [], []
    n = len(feature_array)
    end_limit = n - window_size - horizon + 1
    for i in range(end_limit):
        start = i
        end = i + window_size
        target_idx = end + horizon - 1
        x.append(feature_array[start:end])
        y.append(target_array[target_idx])
        ts.append(timestamps[target_idx])

    if not x:
        return np.empty((0, window_size, feature_array.shape[1])), np.empty((0, 1)), np.empty((0,))

    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32).reshape(-1, 1), np.asarray(ts)


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE(%)": mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


# ---------------------------------------------------------
# 3) Veri hazırlık
# ---------------------------------------------------------
def load_and_prepare_dataframe(cfg: Config) -> pd.DataFrame:
    print("\n📥 Veri dosyaları okunuyor...")
    gen_df = safe_read_csv(cfg.generation_csv)
    wea_df = safe_read_csv(cfg.weather_csv)

    required_gen = {"DATE_TIME", cfg.target_col}
    required_wea = {"DATE_TIME"}

    if not required_gen.issubset(gen_df.columns):
        raise ValueError(f"Generation dosyasında eksik kolon(lar): {required_gen - set(gen_df.columns)}")
    if not required_wea.issubset(wea_df.columns):
        raise ValueError(f"Weather dosyasında eksik kolon(lar): {required_wea - set(wea_df.columns)}")

    # DATE_TIME dönüşümü ve temizleme
    gen_df["DATE_TIME"] = pd.to_datetime(gen_df["DATE_TIME"], errors="coerce")
    wea_df["DATE_TIME"] = pd.to_datetime(wea_df["DATE_TIME"], errors="coerce")

    gen_df = gen_df.dropna(subset=["DATE_TIME"]).sort_values("DATE_TIME")
    wea_df = wea_df.dropna(subset=["DATE_TIME"]).sort_values("DATE_TIME")

    # Aynı timestamp'teki tekrarlı kayıtları ortalama alarak birleştir
    numeric_gen = [c for c in gen_df.columns if pd.api.types.is_numeric_dtype(gen_df[c])]
    numeric_wea = [c for c in wea_df.columns if pd.api.types.is_numeric_dtype(wea_df[c])]

    gen_agg = gen_df.groupby("DATE_TIME", as_index=False)[numeric_gen].mean()
    wea_agg = wea_df.groupby("DATE_TIME", as_index=False)[numeric_wea].mean()

    # Zaman damgasına göre birleştirme
    df = pd.merge(gen_agg, wea_agg, on="DATE_TIME", how="inner", suffixes=("", "_W"))
    df = df.drop_duplicates(subset=["DATE_TIME"]).sort_values("DATE_TIME").reset_index(drop=True)

    if df.empty:
        raise ValueError("Birleştirme sonrası veri boş kaldı")

    # Gece saatleri ve NaN temizliği
    if cfg.target_col not in df.columns:
        raise ValueError(f"Hedef kolon birleştirme sonrası bulunamadı: {cfg.target_col}")

    # Gece tespiti: önce IRRADIATION varsa onu kullan, yoksa saat bazlı kural
    if "IRRADIATION" in df.columns:
        night_mask = df["IRRADIATION"].fillna(0) <= 0
    else:
        hour = df["DATE_TIME"].dt.hour
        night_mask = (hour < 6) | (hour >= 20)

    # Gece 0 değerlerini NaN yapıp zaman enterpolasyonu ile doldur
    zero_night = (df[cfg.target_col] == 0) & night_mask
    df.loc[zero_night, cfg.target_col] = np.nan

    # Sayısal kolonları enterpolasyon + ffill/bfill ile tamamla
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # Kritik kontroller
    if df[cfg.target_col].isna().any():
        raise ValueError(f"Hedef kolonda NaN kaldı: {cfg.target_col}")

    print(f"✅ Hazırlanan toplam kayıt: {len(df)}")
    return df


def prepare_datasets(cfg: Config):
    df = load_and_prepare_dataframe(cfg)
    df = normalize_to_exact_split(df)

    feature_cols = infer_feature_columns(df, cfg.target_col)
    print(f"📌 Özellik kolonları ({len(feature_cols)}): {feature_cols}")

    train_df, val_df, test_df = exact_chronological_split(df)
    print(f"📊 Bölme: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Ölçekleyiciler sadece train üzerinde fit edilir
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    x_train_raw = train_df[feature_cols].values
    y_train_raw = train_df[[cfg.target_col]].values

    x_scaler.fit(x_train_raw)
    y_scaler.fit(y_train_raw)

    def transform_split(split_df: pd.DataFrame):
        x = x_scaler.transform(split_df[feature_cols].values)
        y = y_scaler.transform(split_df[[cfg.target_col]].values)
        ts = split_df["DATE_TIME"].values
        return x, y, ts

    x_train, y_train, ts_train = transform_split(train_df)
    x_val, y_val, ts_val = transform_split(val_df)
    x_test, y_test, ts_test = transform_split(test_df)

    x_train_w, y_train_w, ts_train_w = create_windows(x_train, y_train, ts_train, cfg.window_size, cfg.forecast_horizon)
    x_val_w, y_val_w, ts_val_w = create_windows(x_val, y_val, ts_val, cfg.window_size, cfg.forecast_horizon)
    x_test_w, y_test_w, ts_test_w = create_windows(x_test, y_test, ts_test, cfg.window_size, cfg.forecast_horizon)

    if min(len(x_train_w), len(x_val_w), len(x_test_w)) == 0:
        raise ValueError("Window sonrası train/val/test alt kümelerinden en az biri boş kaldı")

    print("📐 Window tensör boyutları:")
    print(f"   Train: X={x_train_w.shape}, y={y_train_w.shape}")
    print(f"   Val  : X={x_val_w.shape}, y={y_val_w.shape}")
    print(f"   Test : X={x_test_w.shape}, y={y_test_w.shape}")

    return {
        "feature_cols": feature_cols,
        "x_train": x_train_w,
        "y_train": y_train_w,
        "x_val": x_val_w,
        "y_val": y_val_w,
        "x_test": x_test_w,
        "y_test": y_test_w,
        "ts_test": ts_test_w,
        "y_scaler": y_scaler,
    }


# ---------------------------------------------------------
# 4) Model tanımları
# ---------------------------------------------------------
def build_cnn_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, dtype="float32"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def build_rnn_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.SimpleRNN(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, dtype="float32"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, dtype="float32"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def build_gru_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, dtype="float32"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def build_proposed_hybrid_model(input_shape: Tuple[int, int], hp: ProposedHP) -> tf.keras.Model:
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(hp.conv_filters, kernel_size=3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.GRU(hp.units, return_sequences=False)(x)
    x = layers.Dropout(hp.dropout)(x)
    x = layers.Dense(max(hp.units // 2, 16), activation="relu")(x)
    out = layers.Dense(1, dtype="float32")(x)

    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.learning_rate), loss="mse")
    return model


def fit_model(model: tf.keras.Model, x_train, y_train, x_val, y_val, epochs: int, batch_size: int):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=0),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    return history


# ---------------------------------------------------------
# 5) G-HS + OBL + Dinamik PAR/BW
# ---------------------------------------------------------
HP_BOUNDS = {
    "units": (32, 128),
    "dropout": (0.1, 0.5),
    "learning_rate": (1e-4, 5e-3),
    "batch_size": (16, 64),
    "conv_filters": (16, 96),
}

HP_CHOICES = {
    "units": [32, 48, 64, 96, 128],
    "batch_size": [16, 32, 64],
    "conv_filters": [16, 32, 48, 64, 96],
}

CONTINUOUS_PARAMS = {"dropout", "learning_rate"}
DISCRETE_PARAMS = {"units", "batch_size", "conv_filters"}


def nearest_choice(value: float, choices: List[int]) -> int:
    return int(min(choices, key=lambda c: abs(c - value)))


def random_hp() -> ProposedHP:
    return ProposedHP(
        units=random.choice(HP_CHOICES["units"]),
        dropout=round(random.uniform(*HP_BOUNDS["dropout"]), 4),
        learning_rate=10 ** random.uniform(np.log10(HP_BOUNDS["learning_rate"][0]), np.log10(HP_BOUNDS["learning_rate"][1])),
        batch_size=random.choice(HP_CHOICES["batch_size"]),
        conv_filters=random.choice(HP_CHOICES["conv_filters"]),
    )


def evaluate_hp(hp: ProposedHP, data: Dict[str, np.ndarray], epochs: int) -> float:
    tf.keras.backend.clear_session()
    model = build_proposed_hybrid_model(data["x_train"].shape[1:], hp)
    fit_model(model, data["x_train"], data["y_train"], data["x_val"], data["y_val"], epochs=epochs, batch_size=hp.batch_size)
    val_loss = float(model.evaluate(data["x_val"], data["y_val"], verbose=0))
    del model
    gc.collect()
    return val_loss


def ghs_optimize(data: Dict[str, np.ndarray], cfg: Config) -> Tuple[ProposedHP, List[float], float]:
    print("\n�� G-HS optimizasyonu başlıyor (OBL + Dinamik PAR/BW)")

    # Harmony Memory başlat
    HM = []
    for i in range(cfg.hms):
        hp = random_hp()
        fit = evaluate_hp(hp, data, epochs=max(4, cfg.optimize_epochs // 2))
        HM.append({"hp": hp, "fitness": fit})
        print(f"  Init {i + 1}/{cfg.hms} -> val_loss={fit:.6f}, hp={hp}")

    HM = sorted(HM, key=lambda x: x["fitness"])
    convergence = []

    for t in range(1, cfg.ni + 1):
        par_t = cfg.par_min + ((cfg.par_max - cfg.par_min) / cfg.ni) * t
        bw_t = cfg.bw_max * np.exp(np.log(cfg.bw_min / cfg.bw_max) * (t / cfg.ni))

        best_base = HM[0]["hp"]
        new_params = {}

        for param in ["units", "dropout", "learning_rate", "batch_size", "conv_filters"]:
            low, high = HP_BOUNDS[param]
            r = random.random()

            # HMCR: hafızadan seçim
            if r <= cfg.hmcr:
                value = getattr(random.choice(HM)["hp"], param)

                # Dinamik PAR/BW ile pitch adjustment
                if random.random() <= par_t:
                    step = random.uniform(-1, 1)
                    value = float(np.clip(value + step * bw_t * (high - low), low, high))

                if param in DISCRETE_PARAMS:
                    value = nearest_choice(value, HP_CHOICES[param])

            # OBL: HMCR başarısızsa random + opposite karşılaştır
            else:
                if param in DISCRETE_PARAMS:
                    rand_val = random.choice(HP_CHOICES[param])
                    opp_raw = low + high - rand_val
                    opp_val = nearest_choice(opp_raw, HP_CHOICES[param])
                else:
                    rand_val = random.uniform(low, high)
                    opp_val = float(np.clip(low + high - rand_val, low, high))

                def make_candidate(override_val):
                    candidate = ProposedHP(
                        units=best_base.units,
                        dropout=best_base.dropout,
                        learning_rate=best_base.learning_rate,
                        batch_size=best_base.batch_size,
                        conv_filters=best_base.conv_filters,
                    )
                    setattr(candidate, param, float(override_val) if param in CONTINUOUS_PARAMS else int(override_val))
                    return candidate

                fit_rand = evaluate_hp(make_candidate(rand_val), data, epochs=4)
                fit_opp = evaluate_hp(make_candidate(opp_val), data, epochs=4)
                value = rand_val if fit_rand <= fit_opp else opp_val

            new_params[param] = float(value) if param in CONTINUOUS_PARAMS else int(value)

        new_hp = ProposedHP(**new_params)
        new_fit = evaluate_hp(new_hp, data, epochs=cfg.optimize_epochs)

        if new_fit < HM[-1]["fitness"]:
            HM[-1] = {"hp": new_hp, "fitness": new_fit}

        HM = sorted(HM, key=lambda x: x["fitness"])
        convergence.append(HM[0]["fitness"])
        print(f"  Iter {t:02d}/{cfg.ni} -> best_val_loss={HM[0]['fitness']:.6f}")

    return HM[0]["hp"], convergence, HM[0]["fitness"]


# ---------------------------------------------------------
# 6) Eğitim/değerlendirme akışı
# ---------------------------------------------------------
def train_and_predict(
    model_name: str,
    model: tf.keras.Model,
    data: Dict[str, np.ndarray],
    y_scaler: MinMaxScaler,
    epochs: int,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    print(f"\n🧠 Eğitim: {model_name}")
    fit_model(model, data["x_train"], data["y_train"], data["x_val"], data["y_val"], epochs=epochs, batch_size=batch_size)

    pred_scaled = model.predict(data["x_test"], verbose=0)
    y_true = y_scaler.inverse_transform(data["y_test"]).reshape(-1)
    y_pred = y_scaler.inverse_transform(pred_scaled).reshape(-1)

    metrics = compute_regression_metrics(y_true, y_pred)
    print(f"   RMSE={metrics['RMSE']:.4f} | MAE={metrics['MAE']:.4f} | MAPE={metrics['MAPE(%)']:.2f}% | R2={metrics['R2']:.4f}")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "metrics": metrics,
    }


def plot_last_3_days(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred_ghs: np.ndarray,
    y_pred_lstm: np.ndarray,
    path: str,
) -> None:
    # Times New Roman tez formatı
    plt.rcParams["font.family"] = "Times New Roman"

    ts = pd.to_datetime(timestamps)
    if len(ts) == 0:
        raise ValueError("Grafik için timestamp bulunamadı")

    end_ts = ts[-1]
    start_ts = end_ts - pd.Timedelta(days=3)
    mask = ts >= start_ts

    plt.figure(figsize=(14, 6))
    plt.plot(ts[mask], y_true[mask], color="black", linewidth=2.8, label="Gerçek Güneş Enerjisi Üretimi")
    plt.plot(ts[mask], y_pred_ghs[mask], color="tab:red", linewidth=2.0, label="G-HS Model Tahmini")
    plt.plot(ts[mask], y_pred_lstm[mask], color="tab:blue", linewidth=2.0, label="Standart LSTM Tahmini")

    plt.title("Test Seti Son 3 Gün: Gerçek vs Tahmin", fontsize=14)
    plt.xlabel("DATE_TIME", fontsize=12)
    plt.ylabel("Power", fontsize=12)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"🖼️ Grafik kaydedildi: {path}")


def run(cfg: Config) -> None:
    set_seed(cfg.seed)
    configure_runtime()

    print("\n" + "=" * 70)
    print("Solar Power Forecasting - Standart Modeller + G-HS Hibrit Model")
    print("=" * 70)

    data = prepare_datasets(cfg)

    # Standart modeller
    input_shape = data["x_train"].shape[1:]
    results = {}

    standard_models = {
        "1D-CNN": build_cnn_model(input_shape),
        "RNN": build_rnn_model(input_shape),
        "LSTM": build_lstm_model(input_shape),
        "GRU": build_gru_model(input_shape),
    }

    for name, model in standard_models.items():
        results[name] = train_and_predict(name, model, data, data["y_scaler"], cfg.epochs, cfg.batch_size)
        tf.keras.backend.clear_session()
        gc.collect()

    # Önerilen model: G-HS optimize hibrit (CNN + GRU)
    best_hp, convergence, best_val_loss = ghs_optimize(data, cfg)
    print(f"\n✅ G-HS en iyi hiperparametre: {best_hp}")
    print(f"✅ G-HS en iyi val_loss: {best_val_loss:.6f}")

    proposed_model = build_proposed_hybrid_model(input_shape, best_hp)
    results["G-HS-Hybrid"] = train_and_predict(
        "G-HS-Hybrid",
        proposed_model,
        data,
        data["y_scaler"],
        epochs=max(cfg.epochs, cfg.optimize_epochs + 8),
        batch_size=best_hp.batch_size,
    )

    # Metrik tablosu
    metrics_rows = []
    for model_name, result in results.items():
        row = {"Model": model_name}
        row.update(result["metrics"])
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)

    # "En iyi" işaretleme (RMSE/MAE/MAPE küçük, R2 büyük)
    metrics_df["ScoreRank"] = (
        metrics_df["RMSE"].rank(method="min")
        + metrics_df["MAE"].rank(method="min")
        + metrics_df["MAPE(%)"].rank(method="min")
        + (metrics_df["R2"].rank(ascending=False, method="min"))
    )
    best_model_name = metrics_df.sort_values("ScoreRank").iloc[0]["Model"]
    metrics_df["Best"] = np.where(metrics_df["Model"] == best_model_name, "⭐", "")
    metrics_df = metrics_df[["Best", "Model", "RMSE", "MAE", "MAPE(%)", "R2", "ScoreRank"]].sort_values("ScoreRank")

    print("\n📊 Test Metrikleri (Tüm Modeller)")
    print(metrics_df.to_string(index=False))

    # Tahmin CSV çıktısı
    pred_df = pd.DataFrame({
        "DATE_TIME": pd.to_datetime(data["ts_test"]),
        "ACTUAL": results["G-HS-Hybrid"]["y_true"],
        "PRED_GHS": results["G-HS-Hybrid"]["y_pred"],
        "PRED_LSTM": results["LSTM"]["y_pred"],
    })
    pred_df.to_csv(cfg.prediction_csv_path, index=False)
    print(f"💾 Tahmin CSV kaydedildi: {cfg.prediction_csv_path}")

    # Son 3 gün görselleştirme
    plot_last_3_days(
        timestamps=data["ts_test"],
        y_true=results["G-HS-Hybrid"]["y_true"],
        y_pred_ghs=results["G-HS-Hybrid"]["y_pred"],
        y_pred_lstm=results["LSTM"]["y_pred"],
        path=cfg.result_plot_path,
    )

    # Optimizasyon yakınsama bilgisi
    print(f"📉 G-HS yakınsama adım sayısı: {len(convergence)}")


if __name__ == "__main__":
    try:
        config = Config()
        run(config)
    except Exception as exc:
        print(f"\n❌ Çalıştırma hatası: {exc}")
        import traceback
        traceback.print_exc()
