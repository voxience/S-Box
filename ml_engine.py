"""
ml_engine.py
Pendekatan machine learning untuk konstruksi S-box (pengembangan dari paper Alamsyah et al. 2025).

Ide:
  Paper mengeksplorasi matriks affine SIRKULAN (256 kandidat -> 128 valid).
  Di sini ruang pencarian diperluas ke SEMUA matriks affine 8x8 biner yang invertible
  (~5.3 x 10^18 matriks) sehingga tidak mungkin dievaluasi satu per satu.
  Model ML (surrogate) dilatih untuk memprediksi SAC dan BIC-SAC dari matriks affine,
  lalu dipakai untuk menyaring (screening) puluhan ribu kandidat dengan cepat.
  Hanya kandidat terbaik hasil prediksi yang dikonstruksi & diuji penuh.

Catatan penting (dibuktikan juga di aplikasi):
  - NL, BIC-NL, LAP, DAP TIDAK berubah oleh transformasi affine yang invertible
    (selalu 112, 112, 0.0625, 0.015625). Yang membedakan kandidat hanyalah SAC & BIC-SAC.
  - SAC hanya bergantung pada nilai tiap BARIS matriks affine,
    BIC-SAC hanya bergantung pada XOR tiap PASANGAN baris.
    Maka fitur "struktur baris" jauh lebih informatif daripada 64 bit mentah.
"""
from __future__ import annotations

import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

import sbox_core as sc

ROW_W = (1 << np.arange(7, -1, -1))   # bit paling kiri = MSB
PAIRS = [(j, k) for j in range(8) for k in range(j + 1, 8)]

MODEL_NAMES = ["KNN", "Random Forest", "Ridge Regression"]
FEATURE_SETS = {
    "Struktur baris (one-hot baris + XOR pasangan baris, 512 fitur)": "struct",
    "Bit mentah matriks (64 fitur)": "raw",
}


# ----------------------------------------------------------------------------
# Generator kandidat
# ----------------------------------------------------------------------------
def random_invertible_matrices(n: int, rng: np.random.Generator) -> np.ndarray:
    out = []
    while len(out) < n:
        batch = rng.integers(0, 2, (max(64, 4 * (n - len(out))), 8, 8), dtype=np.uint8)
        for K in batch:
            if sc.is_invertible(K):
                out.append(K)
                if len(out) == n:
                    break
    return np.array(out, dtype=np.uint8)


def label_matrices(Ks: np.ndarray, c: int = sc.C_AES) -> np.ndarray:
    """Konstruksi S-box + hitung [SAC, BIC-SAC, S] untuk tiap matriks."""
    Y = np.zeros((len(Ks), 3))
    for i, K in enumerate(Ks):
        e = sc.evaluate(sc.construct_sbox(K, c), full=False)
        Y[i] = (e["SAC"], e["BIC-SAC"], e["S"])
    return Y


# ----------------------------------------------------------------------------
# Fitur
# ----------------------------------------------------------------------------
def featurize(Ks: np.ndarray, kind: str = "struct") -> np.ndarray:
    n = len(Ks)
    if kind == "raw":
        return Ks.reshape(n, -1).astype(np.float32)
    rv = (Ks * ROW_W).sum(-1)                       # nilai integer tiap baris (n, 8)
    F = np.zeros((n, 512), dtype=np.float32)
    ar = np.arange(n)
    for j in range(8):
        np.add.at(F, (ar, rv[:, j]), 1)
    for j, k in PAIRS:
        np.add.at(F, (ar, 256 + (rv[:, j] ^ rv[:, k])), 1)
    return F


def s_from_pred(P: np.ndarray) -> np.ndarray:
    return (np.abs(P[:, 0] - 0.5) + np.abs(P[:, 1] - 0.5)) / 2


def make_model(name: str, k: int = 7):
    if name == "KNN":
        return KNeighborsRegressor(n_neighbors=k, weights="distance")
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=100, max_features="sqrt", n_jobs=-1, random_state=0)
    if name == "Ridge Regression":
        return Ridge(alpha=1e-3)
    raise ValueError(name)


# ----------------------------------------------------------------------------
# Training & evaluasi model
# ----------------------------------------------------------------------------
def train_and_compare(Ks, Y, feature_kind, model_names, k=7, test_size=0.25, seed=0):
    F = featurize(Ks, feature_kind)
    idx = np.arange(len(Ks))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
    results, models = [], {}
    for name in model_names:
        m = make_model(name, k)
        t = time.time()
        m.fit(F[tr], Y[tr, :2])
        train_t = time.time() - t
        P = m.predict(F[te])
        Sp = s_from_pred(P)
        # korelasi peringkat (Spearman sederhana)
        rank_corr = np.corrcoef(np.argsort(np.argsort(Sp)), np.argsort(np.argsort(Y[te, 2])))[0, 1]
        # precision@10%: berapa banyak top-10% prediksi yang benar-benar top-10%
        q = max(1, len(te) // 10)
        top_pred = set(np.argsort(Sp)[:q])
        top_true = set(np.argsort(Y[te, 2])[:q])
        results.append({
            "Model": name,
            "R² SAC": r2_score(Y[te, 0], P[:, 0]),
            "R² BIC-SAC": r2_score(Y[te, 1], P[:, 1]),
            "R² nilai S": r2_score(Y[te, 2], Sp),
            "MAE nilai S": mean_absolute_error(Y[te, 2], Sp),
            "Korelasi peringkat": rank_corr,
            "Precision@10%": len(top_pred & top_true) / q,
            "Waktu latih (s)": train_t,
        })
        models[name] = m
    return results, models, (tr, te), F


# ----------------------------------------------------------------------------
# Pencarian S-box berpandu ML
# ----------------------------------------------------------------------------
def ml_guided_search(model, feature_kind, n_screen, top_k, rng, c=sc.C_AES):
    """Saring n_screen kandidat dengan model, evaluasi asli hanya top_k terbaik."""
    t0 = time.time()
    cand = random_invertible_matrices(n_screen, rng)
    t_gen = time.time() - t0
    t1 = time.time()
    P = model.predict(featurize(cand, feature_kind))
    Sp = s_from_pred(P)
    t_pred = time.time() - t1
    order = np.argsort(Sp)[:top_k]
    t2 = time.time()
    Ytop = label_matrices(cand[order], c)
    t_eval = time.time() - t2
    return {
        "matrices": cand[order],
        "pred_S": Sp[order],
        "true": Ytop,
        "time": {"generate": t_gen, "predict": t_pred, "evaluate_top": t_eval},
    }


def random_search(n_eval, rng, c=sc.C_AES):
    """Baseline tanpa ML: evaluasi n_eval kandidat acak."""
    cand = random_invertible_matrices(n_eval, rng)
    Y = label_matrices(cand, c)
    return cand, Y


def circulant_exploration(c=sc.C_AES):
    """Eksplorasi persis seperti paper: 256 matriks sirkulan -> 128 valid."""
    rows = []
    idx_valid = 0
    for f in range(256):
        K = sc.circulant_from_first_row(f)
        inv = sc.is_invertible(K)
        rec = {"first_row": format(f, "08b"), "invertible": inv}
        if inv:
            idx_valid += 1
            s = sc.construct_sbox(K, c)
            e = sc.evaluate(s, full=False)
            rec.update({"K_index": idx_valid, "bijective": e["bijective"], "balanced": e["balanced"],
                        "SAC": e["SAC"], "BIC-SAC": e["BIC-SAC"], "S": e["S"]})
        rows.append(rec)
    return rows
