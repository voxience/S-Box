# S-Box ML Constructor

Konstruksi S-box dengan pendekatan *machine learning* (KNN, Random Forest, Ridge Regression),
dikembangkan dari jurnal:

> Alamsyah, Setiawan, A., Putra, A.T., Budiman, K., Muslim, M.A., Salahudin, S.N., Prasetiyo, B. (2025).
> *AES S-box modification uses affine matrices exploration for increased S-box strength.*
> Nonlinear Dynamics, 113, 3869–3890. https://doi.org/10.1007/s11071-024-10414-3

## Isi aplikasi

| Menu | Isi |
|---|---|
| 1. Ringkasan & Reproduksi Paper | Konstruksi AES S-box & S-box44, nilai NL/SAC/BIC-NL/BIC-SAC/LAP/DAP dicocokkan dengan tabel paper |
| 2. Eksplorasi Affine Sirkulan | Metode paper: 256 matriks sirkulan → 128 valid → K44 terbaik (S = 0,00155) |
| 3. Konstruksi S-box dengan ML | Dataset → latih KNN/RF/Ridge → saring ribuan kandidat → uji penuh top-k → bandingkan |
| 4. Uji S-box Sendiri | Tempel/unggah 256 nilai, langsung diuji |

## Struktur file

```
app.py           # antarmuka Streamlit
sbox_core.py     # GF(2^8), invers, transformasi affine, uji NL/SAC/BIC/LAP/DAP
ml_engine.py     # dataset, fitur, model ML, pencarian berpandu ML
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

1. Buat repository baru di GitHub (boleh public), lalu upload keempat file di atas ke root repo.
   ```bash
   git init
   git add app.py sbox_core.py ml_engine.py requirements.txt README.md
   git commit -m "S-box ML constructor"
   git branch -M main
   git remote add origin https://github.com/<username>/<nama-repo>.git
   git push -u origin main
   ```
2. Buka https://share.streamlit.io → login dengan akun GitHub.
3. Klik **Create app** → pilih **Deploy a public app from GitHub**.
4. Isi: Repository = `<username>/<nama-repo>`, Branch = `main`, Main file path = `app.py`.
5. Klik **Deploy**. Tunggu 1–3 menit sampai dependency terpasang. Link aplikasi `https://<nama>.streamlit.app` bisa dikumpulkan.

## Ringkasan metode & hasil

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
