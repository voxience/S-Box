"""
sbox_core.py
Inti perhitungan S-box berdasarkan jurnal:
  Alamsyah et al. (2025) "AES S-box modification uses affine matrices exploration
  for increased S-box strength", Nonlinear Dynamics 113:3869-3890.
  https://doi.org/10.1007/s11071-024-10414-3

Isi:
  - Aritmetika GF(2^8) dengan polinomial irreducible x^8 + x^4 + x^3 + x + 1 (0x11B)
  - Tabel invers multiplikatif
  - Transformasi affine  B(X) = (K . X^-1 + C) mod 2
  - Uji validitas: balance & bijective
  - Uji kekuatan: NL, SAC, BIC-NL, BIC-SAC, LAP, DAP, nilai S (Eq. 9) dan SV (Eq. 20)
"""
from __future__ import annotations

import numpy as np

IRREDUCIBLE_POLY = 0x11B  # x^8 + x^4 + x^3 + x + 1
C_AES = 0x63              # konstanta 8-bit AES (01100011)

# Matriks affine AES (baris dibaca kiri->kanan = kolom b0..b7)
K_AES_ROWS = ["10001111", "11000111", "11100011", "11110001",
              "11111000", "01111100", "00111110", "00011111"]
# Matriks affine terbaik pada paper (K44)
K44_ROWS = ["01010111", "10101011", "11010101", "11101010",
            "01110101", "10111010", "01011101", "10101110"]


# ----------------------------------------------------------------------------
# GF(2^8)
# ----------------------------------------------------------------------------
def gf_mul(a: int, b: int, poly: int = IRREDUCIBLE_POLY) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        if a & 0x100:
            a ^= poly
        b >>= 1
    return r


def gf_inverse_table(poly: int = IRREDUCIBLE_POLY) -> np.ndarray:
    """Tabel invers multiplikatif (Tabel 1 di paper). inv(0) = 0."""
    inv = np.zeros(256, dtype=np.uint8)
    for a in range(1, 256):
        for b in range(1, 256):
            if gf_mul(a, b, poly) == 1:
                inv[a] = b
                break
    return inv


INV_TABLE = gf_inverse_table()

# bit matrix: BITS[x, i] = bit ke-i dari x (i=0 LSB)
BITS = ((np.arange(256)[:, None] >> np.arange(8)[None, :]) & 1).astype(np.uint8)


# ----------------------------------------------------------------------------
# Matriks affine
# ----------------------------------------------------------------------------
def rows_to_matrix(rows: list[str]) -> np.ndarray:
    return np.array([[int(c) for c in r] for r in rows], dtype=np.uint8)


def matrix_to_rows(K: np.ndarray) -> list[str]:
    return ["".join(str(int(v)) for v in row) for row in K]


def circulant_from_first_row(first_row: int) -> np.ndarray:
    """Matriks sirkulan: baris ke-i = baris pertama dirotasi kanan i kali.
    (K_AES dan K44 pada paper adalah matriks sirkulan.)"""
    bits = [(first_row >> (7 - j)) & 1 for j in range(8)]  # MSB ditulis paling kiri
    return np.array([np.roll(bits, i) for i in range(8)], dtype=np.uint8)


def gf2_rank(K: np.ndarray) -> int:
    M = K.copy().astype(np.uint8)
    rank, rows, cols = 0, M.shape[0], M.shape[1]
    for c in range(cols):
        piv = None
        for r in range(rank, rows):
            if M[r, c]:
                piv = r
                break
        if piv is None:
            continue
        M[[rank, piv]] = M[[piv, rank]]
        for r in range(rows):
            if r != rank and M[r, c]:
                M[r] ^= M[rank]
        rank += 1
    return rank


def is_invertible(K: np.ndarray) -> bool:
    return gf2_rank(K) == 8


# ----------------------------------------------------------------------------
# Konstruksi S-box
# ----------------------------------------------------------------------------
def construct_sbox(K: np.ndarray, c: int = C_AES) -> np.ndarray:
    """B(X) = (K . X^-1 + C) mod 2  untuk semua X in GF(2^8)."""
    inv_bits = BITS[INV_TABLE]                 # (256, 8)  bit dari X^-1
    c_bits = BITS[c]                           # (8,)
    out_bits = (inv_bits @ K.T.astype(np.int64) + c_bits) % 2
    return (out_bits * (1 << np.arange(8))).sum(axis=1).astype(np.uint8)


def is_bijective(sbox: np.ndarray) -> bool:
    return len(np.unique(sbox)) == 256


def is_balanced(sbox: np.ndarray) -> bool:
    ones = BITS[sbox].sum(axis=0)
    return bool(np.all(ones == 128))


# ----------------------------------------------------------------------------
# Uji kekuatan
# ----------------------------------------------------------------------------
def _walsh(f_pm: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard transform, input dalam bentuk +-1, shape (..., 256)."""
    a = f_pm.astype(np.int64).copy()
    h = 1
    while h < 256:
        a = a.reshape(*a.shape[:-1], -1, 2 * h)
        x, y = a[..., :h].copy(), a[..., h:].copy()
        a[..., :h], a[..., h:] = x + y, x - y
        a = a.reshape(*a.shape[:-2], 256)
        h *= 2
    return a


def _nl_of_bool(f: np.ndarray) -> np.ndarray:
    """Nonlinearity fungsi boolean (f dalam 0/1, shape (..., 256))."""
    w = _walsh(1 - 2 * f.astype(np.int64))
    return 128 - np.abs(w).max(axis=-1) // 2


def nonlinearity(sbox: np.ndarray) -> dict:
    fb = BITS[sbox].T                       # (8, 256) fungsi boolean per bit output
    nls = _nl_of_bool(fb)
    return {"min": int(nls.min()), "avg": float(nls.mean()), "per_bit": nls.tolist()}


def sac(sbox: np.ndarray) -> dict:
    x = np.arange(256)
    M = np.zeros((8, 8))
    for i in range(8):
        diff = sbox[x] ^ sbox[x ^ (1 << i)]
        M[i] = BITS[diff].mean(axis=0)      # P(bit output j berubah | bit input i dibalik)
    return {"avg": float(M.mean()), "matrix": M}


def bic_nl(sbox: np.ndarray) -> dict:
    fb = BITS[sbox].T
    vals, M = [], np.full((8, 8), np.nan)
    for j in range(8):
        for k in range(j + 1, 8):
            v = int(_nl_of_bool(fb[j] ^ fb[k]))
            M[j, k] = M[k, j] = v
            vals.append(v)
    return {"min": int(min(vals)), "avg": float(np.mean(vals)), "matrix": M}


def bic_sac(sbox: np.ndarray) -> dict:
    x = np.arange(256)
    fb = BITS[sbox]                          # (256, 8)
    M = np.full((8, 8), np.nan)
    vals = []
    for j in range(8):
        for k in range(j + 1, 8):
            g = fb[:, j] ^ fb[:, k]
            s = np.mean([np.mean(g[x] ^ g[x ^ (1 << i)]) for i in range(8)])
            M[j, k] = M[k, j] = s
            vals.append(s)
    return {"avg": float(np.mean(vals)), "matrix": M}


def lap(sbox: np.ndarray) -> float:
    # LAT via Walsh untuk semua mask output b (1..255)
    b = np.arange(1, 256)
    parity = (BITS[sbox[None, :] & b[:, None]].sum(axis=-1) & 1)   # (255, 256)
    w = _walsh(1 - 2 * parity)
    return float(np.abs(w).max() / 2 / 256)


def dap(sbox: np.ndarray) -> float:
    x = np.arange(256)
    best = 0
    for dx in range(1, 256):
        cnt = np.bincount(sbox[x] ^ sbox[x ^ dx], minlength=256)
        best = max(best, cnt.max())
    return float(best / 256)


def evaluate(sbox: np.ndarray, full: bool = True) -> dict:
    """Evaluasi lengkap. full=False -> hanya SAC & BIC-SAC (cepat, untuk eksplorasi)."""
    s = sac(sbox)
    bs = bic_sac(sbox)
    x = np.arange(256)
    res = {
        "bijective": is_bijective(sbox),
        "balanced": is_balanced(sbox),
        "fixed_points": int(np.sum(sbox == x)),
        "opposite_fixed_points": int(np.sum(sbox == (x ^ 0xFF))),
        "SAC": s["avg"],
        "BIC-SAC": bs["avg"],
        "S": (abs(s["avg"] - 0.5) + abs(bs["avg"] - 0.5)) / 2,      # Eq. (9)
        "_sac_matrix": s["matrix"],
        "_bicsac_matrix": bs["matrix"],
    }
    if full:
        nl = nonlinearity(sbox)
        bn = bic_nl(sbox)
        res.update({
            "NL": nl["min"],
            "BIC-NL": bn["min"],
            "LAP": lap(sbox),
            "DAP": dap(sbox),
        })
        res["SV"] = (120 - res["NL"]) + abs(0.5 - res["SAC"]) + \
                    (120 - res["BIC-NL"]) + abs(0.5 - res["BIC-SAC"])   # Eq. (20)
    return res


def sbox_to_grid(sbox: np.ndarray) -> np.ndarray:
    return np.asarray(sbox).reshape(16, 16)
