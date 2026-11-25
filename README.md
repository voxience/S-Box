# S-Box ML Constructor

Konstruksi S-box dengan pendekatan *machine learning* (KNN, Random Forest, Ridge Regression),
dikembangkan dari jurnal:

> Alamsyah, Setiawan, A., Putra, A.T., Budiman, K., Muslim, M.A., Salahudin, S.N., Prasetiyo, B. (2025).
> *AES S-box modification uses affine matrices exploration for increased S-box strength.*
> Nonlinear Dynamics, 113, 3869–3890. https://doi.org/10.1007/s11071-024-10414-3

Aplikasi punya **dua mode**:

- **Mode 1 (ML-affine, metode paper):** `B(X) = K·X⁻¹ + C` di GF(2⁸); ML memprediksi SAC/BIC-SAC dari matriks affine.
- **Mode 2 (ML-permutasi, baru):** S-box dibangun dari **permutasi acak** dan dioptimasi dengan hill-climbing
  yang dipandu *surrogate model* ML. Tidak memakai invers GF(2⁸), matriks affine, atau S-box AES.

## Isi aplikasi

| Menu | Isi |
|---|---|
| 1. Ringkasan & Reproduksi Paper | Konstruksi AES S-box & S-box44, nilai NL/SAC/BIC-NL/BIC-SAC/LAP/DAP dicocokkan dengan tabel paper |
| 2. Eksplorasi Affine Sirkulan | Metode paper: 256 matriks sirkulan → 128 valid → K44 terbaik (S = 0,00155) |
| 3. Konstruksi S-box dengan ML | Mode 1: dataset → latih KNN/RF/Ridge → saring ribuan kandidat → uji penuh top-k → bandingkan |
| 4. Uji S-box Sendiri | Tempel/unggah 256 nilai, langsung diuji (termasuk NL komponen & DU) |
| 5. Mode 2 — Konstruksi Permutasi Berpandu ML | Hasil eksperimen offline, demo live, S-box terbaik, tabel perbandingan akhir |

## Struktur file

```
app.py              # antarmuka Streamlit (menu 1–4, Tentang) + memanggil mode2_ui
mode2_ui.py         # halaman menu 5 (Mode 2)
sbox_core.py        # GF(2^8), invers, affine, uji NL/SAC/BIC/LAP/DAP + evaluasi exact tervektorisasi
ml_engine.py        # Mode 1: dataset, fitur, model ML, pencarian matriks affine
ml_permutation.py   # Mode 2: cost WHS, fitur murah, surrogate, hill-climbing, eksperimen pembanding
run_experiment.py   # eksperimen offline Mode 2 (paralel) -> results/*.json
results/            # mode2_results.json, best_sbox_mode2.json (dibaca aplikasi)
requirements.txt
```

## Menjalankan di komputer sendiri

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Upload ke Streamlit Community Cloud

1. Buat repository baru di GitHub (boleh public), lalu upload semua file di atas **termasuk folder `results/`**.
   ```bash
   git add app.py mode2_ui.py sbox_core.py ml_engine.py ml_permutation.py run_experiment.py \
           requirements.txt README.md results/
   git commit -m "S-box ML constructor + Mode 2"
   git push
   ```
2. Buka https://share.streamlit.io → login dengan akun GitHub.
3. Klik **Create app** → pilih **Deploy a public app from GitHub**.
4. Isi: Repository = `<username>/<nama-repo>`, Branch = `main`, Main file path = `app.py`.
5. Klik **Deploy**. Tunggu 1–3 menit sampai dependency terpasang.

## Mode 1 — ringkasan metode & hasil

- Konstruksi: `B(X) = (K · X⁻¹ + C) mod 2`, polinomial `x⁸+x⁴+x³+x+1`, `C = 0x63`.
- NL, BIC-NL, LAP, DAP **tidak berubah** oleh matriks affine invertible (selalu 112, 112, 0,0625, 0,015625),
  jadi kualitas kandidat dibedakan oleh SAC & BIC-SAC → nilai S (Eq. 9).
- ML dipakai sebagai *surrogate model* untuk memprediksi SAC & BIC-SAC dari matriks affine, sehingga ruang
  pencarian bisa diperluas dari 128 matriks sirkulan ke semua matriks 8×8 invertible (~5,3×10¹⁸).
- Fitur: 64 bit mentah (gagal, R² ≈ 0) vs struktur baris (one-hot nilai baris + XOR pasangan baris).
  Dengan fitur struktur baris, Ridge Regression mencapai R² ≈ 1; KNN & Random Forest jauh lebih lemah.

| S-box | NL | SAC | BIC-NL | BIC-SAC | LAP | DAP | S | SV |
|---|---|---|---|---|---|---|---|---|
| AES | 112 | 0,50488 | 112 | 0,50460 | 0,0625 | 0,015625 | 0,00474 | 16,0095 |
| S-box44 (paper) | 112 | 0,50073 | 112 | 0,50237 | 0,0625 | 0,015625 | 0,00155 | 16,0031 |
| S-box ML (Ridge, contoh) | 112 | 0,50000 | 112 | 0,50000 | 0,0625 | 0,015625 | 0,00000 | 16,0000 |

Catatan: SAC/BIC-SAC adalah rata-rata matriks; rata-rata 0,5 tidak berarti tiap entri 0,5 (lihat heatmap di aplikasi).

## Mode 2 — Konstruksi permutasi berpandu ML (surrogate-assisted)

### Mengapa perlu Mode 2
Pada Mode 1 semua matriks affine invertible menghasilkan NL = 112, BIC-NL = 112, LAP = 0,0625, DAP = 0,015625,
karena nilai itu milik fungsi invers GF(2⁸) dan tidak diubah oleh transformasi affine. Jadi nilai tinggi itu
**berasal dari matematika AES, bukan dari ML**. Mode 2 membangun S-box dari **permutasi acak** dan
memperbaikinya dengan operator **swap dua nilai output** (selalu bijektif). Tidak ada invers GF(2⁸), matriks
affine, atau S-box AES yang dipakai, sehingga NL/DU/LAP **tidak dijamin** dan kontribusi ML dapat diukur.

### Metode
- **Evaluasi exact ("mahal")** — `sbox_core.evaluate_full`: spektrum Walsh 255 fungsi komponen (= LAT penuh,
  via perkalian matriks Hadamard), DDT penuh, SAC (mean/std/max dev), BIC-NL, BIC-SAC, fixed point.
  NL dilaporkan dua versi: **NL komponen** (min atas 255 fungsi b·S(x)) dan **NL koordinat** (min 8 bit, definisi paper).
  **1 evaluasi exact = 1 unit anggaran**, sama untuk semua metode.
- **Cost yang dioptimasi**: `Σ_{b≠0} Σ_a | |W_b(a)| − X |^r`. Default **X = 0, r = 8**, dipilih dari studi
  parameter (8.000 evaluasi, 2 seed): r = 3/4 → hill-climbing hanya mencapai NL 100; r = 6–8 → NL 102 (tanpa ML)
  dan 104 (dengan ML). X dan r bisa diatur di UI dan `run_experiment.py`.
- **Skor akhir (leksikografis)**: NL komponen ↑ → DU ↓ → LAP ↓ → |SAC−0,5| + |BIC-SAC−0,5| ↓.
- **Fitur murah per kandidat swap** (inkremental, ribuan kandidat sekaligus):
  1. 10 fitur dari **8 fungsi koordinat + matriks SAC 8×8**: Δcost koordinat, Δmax|W|, Δ#|W| ≥ ambang, Δ Σ max|W_i|,
     #koordinat berubah, Δ Σ|SAC−0,5|, Δmax|SAC−0,5|, jarak Hamming posisi swap;
  2. 4 fitur **entri puncak spektrum** S-box *saat ini* (Δ#|W| ≥ puncak, Δ#|W| ≥ puncak−4, Δmax|W|, Δcost parsial).
     Spektrum itu sudah tersedia dari evaluasi exact solusi sekarang, jadi tidak menambah evaluasi exact.
     Karena satu swap mengubah tiap W_b(a) sebesar 0 atau ±4, dua fitur pertama bersifat exact.
  Fitur dinormalisasi z-score di dalam kolam kandidat tiap iterasi (mengatasi drift skala selama optimasi).
- **Loop surrogate (C)**: warm-up swap acak → tiap iterasi bangkitkan 1.000 kandidat swap → surrogate
  (KNN k = 15 / Random Forest / Ridge) memprediksi Δcost → **hanya top-10 dievaluasi exact** → terima yang terbaik
  jika cost turun → surrogate dilatih ulang online dari data exact (active learning).
- **Pembanding (anggaran sama)**: (A) random search, (B) hill-climbing tanpa ML (1 swap acak = 1 evaluasi exact),
  (C) hill-climbing + ML, serta ablasi **(D)** penyaringan dengan *satu* fitur murah (Δcost parsial puncak) tanpa ML
  dan **(C′)** KNN dengan fitur 8 koordinat + SAC saja.

### Menjalankan eksperimen offline

```bash
python run_experiment.py --budget 2000 --seeds 0 1 --jobs 4            # cek cepat (±1 menit)
python run_experiment.py --budget 6000 --seeds 0 1 2 3 4 --jobs 8      # setting hasil di bawah (±11 menit)
python run_experiment.py --help                                         # semua opsi (X, r, k, top-m, model, ...)
```

Hasil disimpan ke `results/mode2_results.json` (ringkasan, uji statistik, kurva konvergensi, detail run) dan
`results/best_sbox_mode2.json` (S-box terbaik + metrik). Aplikasi otomatis menampilkan file ini di menu 5.
Demo live di aplikasi memakai anggaran kecil (default 1.000 evaluasi, 1 seed; ±15 detik di laptop, 1 thread).

### Hasil aktual

Dijalankan 2026-11-25 12:06:06: **anggaran 6.000 evaluasi exact per run, 5 seed (0–4)**,
1.000 kandidat/iterasi, top-m = 10, X = 0, r = 8, k = 15. Nilai = S-box terbaik tiap run, rata-rata ± std atas seed.
"Capai NL ≥ …" = persentase seed yang mencapai level itu (rata-rata evaluasi exact yang dibutuhkan).

| Metode | NL komponen | maks | DU | LAP | SAC | BIC-SAC | cost akhir | capai NL ≥ 102 | capai NL ≥ 104 |
|---|---|---|---|---|---|---|---|---|---|
| A. Random search | 98,0 ± 0,0 | 98 | 10,0 ± 0,0 | 0,1172 | 0,5007 | 0,5004 | 2,51e+16 | 0% | 0% |
| B. Hill-climbing tanpa ML | 102,0 ± 0,0 | 102 | 8,0 ± 0,0 | 0,1016 | 0,5021 | 0,5012 | 1,71e+16 | 100% (≈1.044) | 0% |
| C'. Hill-climbing + ML (KNN, fitur koordinat saja) | 102,0 ± 0,0 | 102 | 8,4 ± 0,8 | 0,1016 | 0,4987 | 0,5012 | 1,69e+16 | 100% (≈1.342) | 0% |
| C. Hill-climbing + ML (KNN) | 104,0 ± 0,0 | 104 | 8,0 ± 0,0 | 0,0938 | 0,5022 | 0,5023 | 1,44e+16 | 100% (≈320) | 100% (≈2.186) |
| C. Hill-climbing + ML (Random Forest) | 104,0 ± 0,0 | 104 | 8,0 ± 0,0 | 0,0938 | 0,5018 | 0,4997 | 1,48e+16 | 100% (≈328) | 100% (≈1.968) |
| C. Hill-climbing + ML (Ridge Regression) | 104,0 ± 0,0 | 104 | 8,0 ± 0,0 | 0,0938 | 0,4980 | 0,5033 | 1,46e+16 | 100% (≈331) | 100% (≈2.301) |
| D. Penyaringan heuristik (tanpa ML) | 104,0 ± 0,0 | 104 | 8,0 ± 0,0 | 0,0938 | 0,5030 | 0,4990 | 1,48e+16 | 100% (≈242) | 100% (≈2.452) |

**Uji Mann–Whitney U satu sisi vs (B)** (5 vs 5 seed):

| Pembanding | Metrik | p-value |
|---|---|---|
| C'. Hill-climbing + ML (KNN, fitur koordinat saja) vs B | NL_comp | 1,0000 |
| C'. Hill-climbing + ML (KNN, fitur koordinat saja) vs B | cost | 0,2103 |
| C. Hill-climbing + ML (KNN) vs B | NL_comp | 0,0020 |
| C. Hill-climbing + ML (KNN) vs B | cost | 0,0040 |
| C. Hill-climbing + ML (Random Forest) vs B | NL_comp | 0,0020 |
| C. Hill-climbing + ML (Random Forest) vs B | cost | 0,0040 |
| C. Hill-climbing + ML (Ridge Regression) vs B | NL_comp | 0,0020 |
| C. Hill-climbing + ML (Ridge Regression) vs B | cost | 0,0040 |
| D. Penyaringan heuristik (tanpa ML) vs B | NL_comp | 0,0020 |
| D. Penyaringan heuristik (tanpa ML) vs B | cost | 0,0040 |

**Akurasi surrogate** (kandidat acak dievaluasi exact khusus untuk diagnostik, tidak dihitung anggaran):

| Metode | Spearman prediksi vs Δcost aktual | Spearman heuristik 1 fitur | Precision@top-10 | % kandidat yang memperbaiki: acak → top-m |
|---|---|---|---|---|
| C'. Hill-climbing + ML (KNN, fitur koordinat saja) | 0,02 | 0,06 | 0,03 (acak 0,03) | 1,13% → 0,7% |
| C. Hill-climbing + ML (KNN) | 0,85 | 0,96 | 0,61 (acak 0,03) | 0,04% → 1,3% |
| C. Hill-climbing + ML (Random Forest) | 0,76 | 0,96 | 0,64 (acak 0,03) | 0,02% → 0,7% |
| C. Hill-climbing + ML (Ridge Regression) | 0,96 | 0,97 | 0,77 (acak 0,03) | 0,04% → 1,3% |

**Waktu** (laptop i7-1360P, 8 proses paralel): evaluasi exact ≈ 5,0 ms/S-box, fitur murah ≈ 0,16 ms/kandidat (±32× lebih murah).

**S-box Mode 2 terbaik** (C. Hill-climbing + ML (Ridge Regression), seed 3): NL komponen = 104, NL koordinat = 104,
BIC-NL = 104, DU = 8 (DAP = 0.03125), LAP = 0.09375, SAC = 0.50000,
BIC-SAC = 0.50028, fixed point = 3, bijektif ✅.

### Perbandingan akhir

| S-box | NL komponen | NL koordinat | BIC-NL | DU | LAP | Nilai dijamin konstruksi? |
|---|---|---|---|---|---|---|
| AES | 112 | 112 | 112 | 4 | 0,0625 | Ya (invers GF(2⁸)) |
| S-box44 (paper) | 112 | 112 | 112 | 4 | 0,0625 | Ya (invers GF(2⁸)) |
| S-box Mode 1 (ML-affine) | 112 | 112 | 112 | 4 | 0,0625 | Ya (invers GF(2⁸)) |
| **S-box Mode 2 (ML-permutasi)** | **104** | **104** | **104** | **8** | **0,0938** | Tidak — hasil pencarian |
| Random search (terbaik) | 98 | 100 | 98 | 10 | 0,1172 | Tidak |
| Hill-climbing tanpa ML (terbaik) | 102 | 104 | 102 | 8 | 0,1016 | Tidak |

### Interpretasi (jujur)
1. **ML-guided (C) konsisten lebih baik daripada hill-climbing tanpa ML (B) pada anggaran evaluasi exact yang sama**:
   NL komponen 104 vs 102 di kelima seed (p = 0,002), cost akhir lebih rendah (p = 0,004), dan NL ≥ 102 dicapai
   ±3× lebih cepat (≈ 320 vs ≈ 1.040 evaluasi). (B) macet di optimum lokal: hanya ±0,04% swap acak yang masih
   memperbaiki cost, sedangkan top-m pilihan surrogate memperbaiki ±30× lebih sering.
2. **Surrogate benar-benar memprediksi**: Spearman 0,76–0,96 dan precision@top-10 0,61–0,77 (acak: 0,03).
3. **Pilihan fitur menentukan**: dengan fitur 8 koordinat saja (C′) Spearman ≈ 0,02 — setara acak — dan hasilnya
   sama dengan (B). Satu swap mengubah 128 fungsi komponen; 8 koordinat hanya melihat ±4 di antaranya.
4. **Ablasi (D)**: penyaringan dengan satu fitur buatan tangan (Δcost parsial entri puncak) tanpa ML mencapai hasil
   yang sama (NL 104). Jadi keuntungan utama berasal dari *penyaringan kandidat dengan fitur murah yang informatif*;
   ML (terutama Ridge, Spearman ≈ heuristik) mempelajari bobot kombinasi fitur tersebut secara otomatis dan online
   tanpa diberi tahu fitur mana yang penting. KNN dan RF sedikit di bawah heuristik dalam akurasi peringkat.
5. **NL 104 < 112**: angka 112 milik AES adalah hasil konstruksi aljabar. Literatur heuristik untuk permutasi 8×8
   umumnya melaporkan NL 104–106; anggaran lebih besar (`--budget 20000`) bisa dicoba untuk mengejar 106.
6. Catatan biaya: untuk cost WHS saja sebenarnya ada pembaruan inkremental exact; anggaran di sini dihitung dalam
   evaluasi *penuh* (termasuk DDT, LAT, SAC, BIC) karena semua metrik itu dipakai dalam skor leksikografis.

