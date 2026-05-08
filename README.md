# frkn

## Yerelde çalıştırma (GTX 1650 öncelikli)

`kod.py` artık GPU seçimini ortam değişkenleriyle kontrol eder:

- `FRKN_GPU_MODE=nvidia` (varsayılan): NVIDIA GPU varsa onu seçer (GTX 1650 için önerilen)
- `FRKN_GPU_MODE=auto`: görünür GPU'larda otomatik devam eder
- `FRKN_GPU_MODE=index` + `FRKN_GPU_INDEX=0`: belirli indeksteki GPU'yu seçer
- `FRKN_GPU_MODE=cpu`: CPU ile çalıştırır

Örnek:

```bash
FRKN_GPU_MODE=nvidia python kod.py
```
