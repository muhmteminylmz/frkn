# frkn

Bu proje, **ikili atık sınıflandırma** (Organic vs Recyclable) problemi için bir CNN modeli kurar ve modelin hiperparametrelerini **G-HS (Global-best Harmony Search) + OBL (Opposition-Based Learning) + dinamik PAR/BW** ile optimize eder. Kod tek giriş noktası olan `kod.py` dosyasında toplanmıştır.

---

## 1) Projenin amacı

Proje iki yaklaşımı kıyaslar:

1. **Baseline CNN** (sabit hiperparametreler)
2. **G-HS optimize edilmiş CNN** (dinamik hiperparametre araması)

Çıktı olarak:
- Test metrikleri (loss/accuracy)
- İyileşme yüzdeleri
- Yakınsama grafikleri (`ghs_results.png`)
- Özet rapor (`summary.json`)

---

## 2) Yüksek seviye mimari

```mermaid
flowchart TD
    A[run()] --> B[Ortam Hazırlığı\nSeed + GPU + Mixed Precision]
    B --> C[G-HS Optimizasyonu\nghs_optimize]
    C --> D[Fitness Hesabı Döngüsü\nevaluate_fitness]
    D --> E[Dataset Pipeline\ncreate_datasets]
    D --> F[Model İnşası\nbuild_cnn_model]
    C --> G[En İyi Hiperparametreler]
    G --> H[Final Eğitim: G-HS-CNN\nfinal_evaluate_ghs_cnn]
    G --> I[Final Eğitim: Baseline\nfinal_evaluate_cnn_baseline]
    H --> J[Test Sonuçları]
    I --> J
    J --> K[Grafikler\nplot_results]
    J --> L[JSON Özet\nsummary.json]
```

---

## 3) Katmanlı yapı ve sorumluluklar

### A. Ortam ve donanım katmanı
- **GPU seçim stratejisi**: `FRKN_GPU_MODE` (`nvidia`, `auto`, `index`, `cpu`) ve `FRKN_GPU_INDEX`.
- TensorFlow GPU görünürlüğü ayarlanır, memory growth açılır.
- Uygun ortamda mixed precision (`mixed_float16` / `mixed_bfloat16`) etkinleştirilir.

### B. Deney konfigürasyon katmanı
- `QUICK_TEST`: hızlı/mini deney akışı.
- `DATASET_SIZE`: eğitim/doğrulama veri hacmini sınırlama.
- Boyut, batch, filtre üst sınırları gibi sabitler (`MAX_FILTERS`, `BOUNDS`, `CHOICES`).

### C. Veri katmanı (`create_datasets`)
- `tf.keras.utils.image_dataset_from_directory` ile train/val/test yüklenir.
- `train` için `%80`, `val` için `%20` ayrım yapılır.
- Alt küme seçimi (`QUICK_TEST` veya `DATASET_SIZE`) uygulanır.
- `prefetch(AUTOTUNE)` ile pipeline hızlandırılır.
- Batch boyutuna göre cache (`_gen_cache`) kullanılır.

### D. Modelleme katmanı

#### 1) Önerilen model (`build_cnn_model`)
Blok yapısı:
- Rescaling + veri artırma
- Her blokta:
  - Conv → BN → ReLU
  - Conv → BN
  - **SE (Squeeze-and-Excitation)** kanal dikkati
  - **Residual bağlantı** (gerekirse 1x1 projeksiyon)
  - ReLU → MaxPool → SpatialDropout2D
- Sınıflandırıcı başlık:
  - GAP → Dense → Dropout → Sigmoid

Ek özellikler:
- `BinaryCrossentropy(label_smoothing=0.05)`
- Adam optimizer (öğrenme oranı hiperparametredir)

#### 2) Baseline model (`build_cnn_baseline`)
- 3 sabit conv blok (32→64→128)
- Flatten + Dense ile klasik referans mimari

### E. Optimizasyon katmanı (`ghs_optimize`)
G-HS süreci:
1. Harmony Memory (HM) başlangıcı (elite seed + rastgele harmoniler)
2. Her iterasyonda yeni aday üretimi:
   - **HMCR** ile hafızadan seçim
   - **PAR/BW** ile ince ayar (dinamik artan/azalan)
   - **OBL** ile karşıt örnek değerlendirmesi
3. Yeni adayın fitness hesabı (`evaluate_fitness`)
4. Daha iyi ise HM’de en kötü harmoninin yerine geçmesi
5. Yakınsama geçmişinin tutulması

### F. Değerlendirme/raporlama katmanı
- En iyi hiperparametre ile final G-HS-CNN eğitimi
- Baseline eğitimi
- Test metriklerinin karşılaştırılması
- Grafik ve JSON rapor üretimi

---

## 4) Detaylı veri ve kontrol akışı

### 4.1 Fitness değerlendirme akışı
```text
evaluate_fitness(hp)
  -> clear_session + cache temizliği
  -> create_datasets(hp.batch_size, use_subset=True)
  -> build_cnn_model(hp)
  -> fit(..., EarlyStopping + ReduceLROnPlateau)
  -> min(val_loss) döndür
```

### 4.2 Ana çalışma akışı
```text
run()
  -> G-HS optimizasyonu (best_hp)
  -> final_evaluate_ghs_cnn(best_hp)
  -> final_evaluate_cnn_baseline()
  -> plot_results(...)
  -> summary.json yaz
```

---

## 5) Fonksiyon haritası

- **Ortam/altyapı**: `set_seed`, GPU kurulum bloğu
- **Veri**: `create_datasets`
- **Model blokları**: `_se_block`, `build_cnn_model`, `build_cnn_baseline`
- **Optimizasyon**: `evaluate_fitness`, `random_hyperparams`, `ghs_optimize`
- **Final değerlendirme**: `final_evaluate_ghs_cnn`, `final_evaluate_cnn_baseline`
- **Çıktılar**: `plot_results`, `run`

---

## 6) Yerelde GPU yapılandırması

`kod.py` GPU seçimini ortam değişkenleriyle kontrol eder:

- `FRKN_GPU_MODE=nvidia` (varsayılan): NVIDIA GPU varsa onu seçer
- `FRKN_GPU_MODE=auto`: görünür GPU'larda otomatik devam eder
- `FRKN_GPU_MODE=index` + `FRKN_GPU_INDEX=0`: belirli indeksteki GPU'yu seçer
- `FRKN_GPU_MODE=cpu`: CPU ile çalıştırır

Örnek:

```bash
FRKN_GPU_MODE=nvidia python kod.py
```

---

## 7) Üretilen çıktı dosyaları

- `ghs_results.png`  
  - Yakınsama eğrisi
  - İyileşme yüzdesi
  - Baseline vs G-HS-CNN accuracy karşılaştırması

- `summary.json`  
  - En iyi hiperparametreler
  - Test metrikleri
  - İyileşme yüzdeleri
  - Yakınsama geçmişi

---

## 8) Klasör ve dosya görünümü

```text
frkn/
├── kod.py      # Tüm eğitim/optimizasyon ve raporlama akışı
├── README.md   # Bu dokümantasyon
└── .gitignore
```

