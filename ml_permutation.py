"""
ml_permutation.py
Mode 2 — Konstruksi S-box 8x8 dari PERMUTASI dengan optimasi berpandu ML (surrogate-assisted).

Berbeda dengan Mode 1 (B(X) = K·X^-1 + C), di sini TIDAK dipakai invers GF(2^8), matriks affine,
maupun S-box AES. Titik awal = permutasi acak; operator = swap dua nilai output (selalu bijektif).
Karena itu NL/DU/LAP TIDAK dijamin oleh konstruksi aljabar — nilainya murni hasil pencarian.

Komponen:
  - exact_eval()        : evaluasi exact "mahal" (255 komponen Walsh + DDT + LAT + SAC + BIC)
  - whs_cost()          : cost spektrum Walsh–Hadamard  sum_{b!=0} sum_a | |W_b(a)| - X |^r
  - SwapState / cheap_features() : fitur "murah" untuk ribuan kandidat swap sekaligus
                          (spektrum 8 fungsi koordinat + matriks SAC 8x8, dihitung inkremental)
  - run_random_search() : (A) permutasi acak
  - run_hill_climb()    : (B) hill-climbing tanpa ML, (C) berpandu ML, (D) ablasi heuristik
  - run_comparison()    : eksperimen adil (anggaran evaluasi exact sama) untuk banyak seed
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sbox_core as sc

MODEL_NAMES = ["KNN", "Random Forest", "Ridge Regression"]
# Default cost hasil studi parameter (README): r besar menekan puncak spektrum -> NL naik lebih jauh
DEFAULT_X = 0.0
DEFAULT_R = 8.0
DEFAULT_K = 15
FEATURE_NAMES = [
    "Δcost koordinat (relatif)", "Δ max|W| koordinat", "Δ #|W| ≥ puncak", "Δ #|W| ≥ puncak−4",
    "Δ #|W| ≥ puncak−8", "Δ Σ max|W_i|", "#koordinat berubah", "Δ Σ|SAC−0.5|", "Δ max|SAC−0.5|",
    "jarak Hamming x1⊕x2",
]
ProgressFn = Callable[[float, str], None] | None


# ----------------------------------------------------------------------------
# Evaluasi exact ("mahal")
# ----------------------------------------------------------------------------
def whs_cost(W: np.ndarray, X: float = DEFAULT_X, r: float = DEFAULT_R) -> float:
    """Cost WHS: sum_{b!=0} sum_a | |W_b(a)| - X |^r  (makin kecil makin baik)."""
    A = np.abs(np.abs(W[1:]).astype(np.float64) - X)
    return float((A ** r).sum())


def lex_key(rec: dict) -> tuple:
    """Kunci leksikografis (makin kecil makin baik):
    NL komponen tertinggi -> DU terendah -> LAP terendah -> |SAC-0.5| + |BIC-SAC-0.5| terkecil."""
    return (-rec["NL_comp"], rec["DU"], rec["LAP"], abs(rec["SAC"] - 0.5) + abs(rec["BIC-SAC"] - 0.5))


def exact_eval(sbox: np.ndarray, X: float = DEFAULT_X, r: float = DEFAULT_R, return_W: bool = False):
    """Evaluasi exact penuh 1 S-box: spektrum 255 komponen, DDT, LAT, SAC, BIC + cost WHS.
    Inilah satuan 'anggaran' yang dibatasi sama untuk semua metode.
    return_W=True -> juga mengembalikan spektrum Walsh penuh (hasil sampingan, tanpa biaya ekstra)."""
    W = sc.walsh_spectrum_all(sbox)
    e = sc.evaluate_full(sbox, W)
    keep = ("NL_comp", "NL_coord", "BIC-NL", "DU", "DAP", "LAP", "SAC", "SAC_std", "SAC_maxdev",
            "BIC-SAC", "fixed_points", "opposite_fixed_points", "bijective")
    rec = {k: e[k] for k in keep}
    rec["cost"] = whs_cost(W, X, r)
    return (rec, W) if return_W else rec


# ----------------------------------------------------------------------------
# Fitur murah untuk kandidat swap
# ----------------------------------------------------------------------------
FEATURE_SETS = {
    "Koordinat + SAC + puncak spektrum (14 fitur)": "full",
    "Koordinat + SAC saja (10 fitur)": "coord",
}
PEAK_FEATURE_NAMES = ["Δ #entri |W| ≥ puncak (255 komponen)", "Δ #entri |W| ≥ puncak−4",
                      "Δ max|W| (255 komponen)", "Δcost parsial entri puncak (relatif)"]


@dataclass
class SwapState:
    """Ringkasan murah dari S-box saat ini: spektrum Walsh 8 fungsi koordinat + hitungan SAC,
    dan (opsional) daftar entri puncak spektrum 255 komponen milik S-box SAAT INI.

    Spektrum penuh S-box saat ini sudah tersedia dari evaluasi exact solusi sekarang (sudah
    dihitung anggaran), jadi memakainya untuk fitur kandidat tidak menambah evaluasi exact.
    """
    sbox: np.ndarray
    X: float = DEFAULT_X
    r: float = DEFAULT_R
    W_full: np.ndarray | None = None          # (256, 256) spektrum S-box saat ini (opsional)
    full_cost: float = 1.0
    peak_k: int = 2048
    Wc: np.ndarray = field(init=False)       # (8, 256) spektrum koordinat
    sac_cnt: np.ndarray = field(init=False)  # (8, 8) hitungan perubahan bit (x2, karena pasangan dobel)

    def __post_init__(self) -> None:
        s = np.asarray(self.sbox, dtype=np.int64)
        self.sbox = s
        if self.W_full is not None:
            A = np.abs(self.W_full[1:])
            self.fpeak = float(A.max())
            flat = A.ravel()
            k = min(self.peak_k, flat.size)
            top = np.argpartition(-flat, k - 1)[:k]
            top = top[flat[top] >= self.fpeak - 16]           # entri di sekitar puncak
            self.pk_b = top // 256 + 1                         # indeks komponen b (1..255)
            self.pk_a = top % 256
            self.pk_w = self.W_full[self.pk_b, self.pk_a].astype(np.float64)
            self.n_peak = int((A == self.fpeak).sum())
            self.n_peak4 = int((A >= self.fpeak - 4).sum())
        self.Wc = sc.HAD[sc.COORD_MASKS][:, s] @ sc.HAD
        x = np.arange(256)
        diff = s[None, :] ^ s[x[None, :] ^ sc.COORD_MASKS[:, None]]
        self.sac_cnt = sc.BITS[diff].sum(axis=1).astype(np.int64)
        absW = np.abs(self.Wc)
        self.peak = float(absW.max())
        self.coord_cost = float((np.abs(absW - self.X) ** self.r).sum())
        self.rowmax_sum = float(absW.max(axis=1).sum())
        self.cnt = [int((absW >= self.peak - t).sum()) for t in (0, 4, 8)]
        M = self.sac_cnt / 256
        self.sac_dev = float(np.abs(M - 0.5).sum())
        self.sac_maxdev = float(np.abs(M - 0.5).max())


def cheap_features(st: SwapState, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Fitur murah untuk N kandidat swap (x1[n] <-> x2[n]), shape (N, 10).

    Swap mengubah fungsi koordinat i hanya jika bit i dari y1^y2 = 1, dengan
    ΔW_i(a) = -2·(-1)^{y1_i}·(H[a,x1] - H[a,x2]). Jadi spektrum baru 8 koordinat dan
    matriks SAC baru dihitung inkremental tanpa evaluasi 255 komponen / DDT / LAT.
    """
    s = st.sbox
    y1, y2 = s[x1], s[x2]
    d = y1 ^ y2
    chg = sc.BITS[d].astype(np.float32)                         # (N, 8) koordinat yang berubah
    sgn = 1 - 2 * sc.BITS[y1].astype(np.float32)                # (N, 8)
    hd = (sc.HAD[:, x1] - sc.HAD[:, x2]).T                      # (N, 256)
    Wn = st.Wc[None] - 2 * (chg * sgn)[:, :, None] * hd[:, None, :]   # (N, 8, 256)
    absW = np.abs(Wn)
    coord_cost = (np.abs(absW - st.X) ** st.r).sum(axis=(1, 2))
    rowmax = absW.max(axis=2)
    F = np.empty((len(x1), 10), dtype=np.float64)
    F[:, 0] = (coord_cost - st.coord_cost) / st.coord_cost
    F[:, 1] = rowmax.max(axis=1) - st.peak
    for j, t in enumerate((0, 4, 8)):
        F[:, 2 + j] = (absW >= st.peak - t).sum(axis=(1, 2)) - st.cnt[j]
    F[:, 5] = rowmax.sum(axis=1) - st.rowmax_sum
    F[:, 6] = chg.sum(axis=1)

    # --- SAC inkremental: pasangan (x1, x1^e_i) dan (x2, x2^e_i) yang berubah
    e = sc.COORD_MASKS[None, :]                                  # (1, 8)
    p1, p2 = x1[:, None] ^ e, x2[:, None] ^ e                    # (N, 8)
    valid = (p1 != x2[:, None]).astype(np.int64)                 # pasangan (x1,x2) tak berubah
    sp1, sp2 = s[p1], s[p2]
    B = sc.BITS
    delta = (B[y2[:, None] ^ sp1].astype(np.int64) - B[y1[:, None] ^ sp1]
             + B[y1[:, None] ^ sp2] - B[y2[:, None] ^ sp2]) * valid[:, :, None]   # (N, 8, 8)
    Mn = (st.sac_cnt[None] + 2 * delta) / 256
    dev = np.abs(Mn - 0.5)
    F[:, 7] = dev.sum(axis=(1, 2)) - st.sac_dev
    F[:, 8] = dev.max(axis=(1, 2)) - st.sac_maxdev
    F[:, 9] = sc.POPCOUNT[x1 ^ x2]
    if st.W_full is None:
        return F

    # --- delta entri puncak spektrum 255 komponen (dari spektrum S-box saat ini)
    # ΔW_b(a) = -2·[b·d = 1]·(-1)^{b·y1}·(H[a,x1] - H[a,x2])  ∈ {0, ±4}
    pb, pa = st.pk_b, st.pk_a
    coef = (sc.PARITY[pb][:, d].astype(np.float32) * (1 - 2 * sc.PARITY[pb][:, y1].astype(np.float32)))
    dW = -2 * coef * (sc.HAD[pa][:, x1] - sc.HAD[pa][:, x2])          # (K, N)
    newA = np.abs(st.pk_w[:, None] + dW)
    oldA = np.abs(st.pk_w)[:, None]
    P = np.empty((len(x1), 4))
    # entri < puncak-4 tidak mungkin mencapai puncak setelah 1 swap -> dua fitur ini EXACT
    P[:, 0] = (newA >= st.fpeak).sum(0) - st.n_peak
    P[:, 1] = (newA >= st.fpeak - 4).sum(0) - (oldA >= st.fpeak - 4).sum()
    P[:, 2] = newA.max(0) - st.fpeak
    P[:, 3] = ((np.abs(newA - st.X) ** st.r) - (np.abs(oldA - st.X) ** st.r)).sum(0) / st.full_cost
    return np.hstack([F, P])


def random_swaps(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """n pasangan posisi (x1, x2) acak dengan x1 != x2."""
    x1 = rng.integers(0, 256, n)
    x2 = (x1 + rng.integers(1, 256, n)) % 256
    return x1, x2


def apply_swap(sbox: np.ndarray, x1: int, x2: int) -> np.ndarray:
    s = sbox.copy()
    s[x1], s[x2] = s[x2], s[x1]
    return s


# ----------------------------------------------------------------------------
# Model surrogate
# ----------------------------------------------------------------------------
def make_surrogate(name: str, k: int = 7, seed: int = 0):
    """Model regresi untuk memprediksi Δcost relatif dari fitur murah."""
    if name == "KNN":
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k, weights="distance"))
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=60, max_depth=10, min_samples_leaf=2,
                                     n_jobs=1, random_state=seed)
    if name == "Ridge Regression":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    raise ValueError(name)


# ----------------------------------------------------------------------------
# Pencatat jalannya optimasi
# ----------------------------------------------------------------------------
class Tracker:
    """Mencatat evaluasi exact: S-box terbaik (leksikografis) & kurva konvergensi."""

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.n = 0
        self.best_rec: dict | None = None
        self.best_sbox: np.ndarray | None = None
        self.best_cost = np.inf
        self.curve_nl = np.zeros(budget, dtype=np.int32)
        self.curve_cost = np.zeros(budget)

    def done(self) -> bool:
        return self.n >= self.budget

    def record(self, sbox: np.ndarray, rec: dict) -> None:
        if self.best_rec is None or lex_key(rec) < lex_key(self.best_rec):
            self.best_rec, self.best_sbox = rec, sbox.copy()
        self.best_cost = min(self.best_cost, rec["cost"])
        self.curve_nl[self.n] = self.best_rec["NL_comp"]
        self.curve_cost[self.n] = self.best_cost
        self.n += 1


# ----------------------------------------------------------------------------
# Metode pencarian
# ----------------------------------------------------------------------------
def run_random_search(budget: int, seed: int, X: float = DEFAULT_X, r: float = DEFAULT_R,
                      progress: ProgressFn = None) -> dict:
    """(A) Random search: tiap evaluasi exact = 1 permutasi acak baru."""
    rng = np.random.default_rng(seed)
    tr = Tracker(budget)
    t0 = time.perf_counter()
    while not tr.done():
        s = rng.permutation(256)
        tr.record(s, exact_eval(s, X, r))
        if progress and tr.n % 100 == 0:
            progress(tr.n / budget, f"Random search {tr.n}/{budget}")
    return _result("A. Random search", tr, time.perf_counter() - t0)


def run_hill_climb(budget: int, seed: int, guide: str = "none", model_name: str = "KNN",
                   k: int = DEFAULT_K, n_cand: int = 1000, top_m: int = 10, warmup: int = 200,
                   X: float = DEFAULT_X, r: float = DEFAULT_R, temp0: float = 0.0, retrain_every: int = 1,
                   window: int = 3000, diag_points: int = 0, diag_size: int = 300,
                   feature_set: str = "full", label: str | None = None,
                   progress: ProgressFn = None) -> dict:
    """Hill-climbing / simulated annealing ringan dengan operator swap.

    guide = "none"      -> (B) tanpa ML: 1 swap acak per evaluasi exact
            "ml"        -> (C) n_cand kandidat -> surrogate -> top_m dievaluasi exact
            "heuristic" -> (D) ablasi tanpa ML: seperti (C) tapi kandidat diurutkan dengan SATU fitur
                           murah (Δcost parsial entri puncak; atau Δcost koordinat jika feature_set="coord")
    Semua metode berhenti tepat setelah `budget` evaluasi exact. temp0 > 0 mengaktifkan annealing
    (penerimaan berdasarkan Δcost relatif, suhu turun linear ke 0).
    Warm-up (C): `warmup` evaluasi pertama = swap acak (seperti B) untuk data latih awal.
    Fitur dinormalisasi z-score di dalam kolam kandidat tiap iterasi (mean/std kolam), sehingga
    data latih dari awal & akhir optimasi berada pada skala yang sama (mengatasi drift).
    diag_points > 0: di beberapa titik, `diag_size` kandidat dievaluasi exact khusus untuk mengukur
    akurasi surrogate (Spearman & precision@top-m). Evaluasi diagnostik TIDAK dihitung anggaran.
    """
    rng = np.random.default_rng(seed)
    tr = Tracker(budget)
    s = rng.permutation(256)
    cur, cur_W = exact_eval(s, X, r, return_W=True)
    tr.record(s, cur)
    use_peak = feature_set == "full"

    def state() -> SwapState:
        return SwapState(s, X, r, W_full=cur_W if use_peak else None, full_cost=cur["cost"])
    Xtr: list[np.ndarray] = []
    ytr: list[float] = []
    model, fitted, it = None, False, 0
    online_pred: list[float] = []
    online_true: list[float] = []
    diag: list[dict] = []
    diag_at = set(np.linspace(warmup + (budget - warmup) * 0.15, budget * 0.95,
                              diag_points).astype(int)) if diag_points else set()
    t_cheap = t_exact = t_model = 0.0
    n_cheap = 0
    accepted = 0
    t0 = time.perf_counter()

    def temperature() -> float:
        return temp0 * max(0.0, 1 - tr.n / budget)

    def accept(delta_rel: float) -> bool:
        if delta_rel < 0:
            return True
        T = temperature()
        return T > 0 and rng.random() < np.exp(-delta_rel / T)

    if guide == "ml":
        model = make_surrogate(model_name, k, seed)

    while not tr.done():
        it += 1
        use_filter = guide == "heuristic" or (guide == "ml" and tr.n >= warmup and fitted)
        if not use_filter:
            # satu swap acak, dievaluasi exact (metode B; juga fase warm-up C)
            if guide == "ml":
                # kolam kecil hanya untuk statistik normalisasi; kandidat ke-0 (acak) yang dievaluasi
                x1, x2 = random_swaps(rng, WARM_POOL)
                tc = time.perf_counter()
                f = _zscore(cheap_features(state(), x1, x2))[0]
                t_cheap += time.perf_counter() - tc
                n_cheap += WARM_POOL
            else:
                x1, x2 = random_swaps(rng, 1)
            cand = apply_swap(s, int(x1[0]), int(x2[0]))
            te = time.perf_counter()
            rec, W = exact_eval(cand, X, r, return_W=True)
            t_exact += time.perf_counter() - te
            tr.record(cand, rec)
            d = (rec["cost"] - cur["cost"]) / cur["cost"]
            if guide == "ml":
                Xtr.append(f)
                ytr.append(d)
            if accept(d):
                s, cur, cur_W = cand, rec, W
                accepted += 1
        else:
            x1, x2 = random_swaps(rng, n_cand)
            tc = time.perf_counter()
            F = cheap_features(state(), x1, x2)
            t_cheap += time.perf_counter() - tc
            n_cheap += n_cand
            tm = time.perf_counter()
            heur = F[:, 13] if use_peak else F[:, 0]
            Fz = _zscore(F)
            pred = model.predict(Fz) if guide == "ml" else heur
            t_model += time.perf_counter() - tm
            m = min(top_m, budget - tr.n)
            order = np.argsort(pred)[:m]

            # --- diagnostik akurasi surrogate (tidak dihitung anggaran)
            if guide == "ml" and any(tr.n <= p < tr.n + m for p in diag_at):
                idx = rng.choice(n_cand, min(diag_size, n_cand), replace=False)
                true_d = np.array([(whs_cost(sc.walsh_spectrum_all(apply_swap(s, int(x1[i]), int(x2[i]))), X, r)
                                    - cur["cost"]) / cur["cost"] for i in idx])
                p_sub = pred[idx]
                mm = min(top_m, len(idx))
                top_pred = set(np.argsort(p_sub)[:mm])
                top_true = set(np.argsort(true_d)[:mm])
                diag.append({
                    "eval": tr.n,
                    "spearman": float(spearmanr(p_sub, true_d).statistic),
                    "spearman_heur": float(spearmanr(heur[idx], true_d).statistic),
                    "precision_at_m": len(top_pred & top_true) / mm,
                    "precision_random": mm / len(idx),
                    "frac_improving_true": float((true_d < 0).mean()),
                    "frac_improving_topm": float((true_d[list(top_pred)] < 0).mean()),
                })

            best_c, best_rec, best_W, best_d = None, None, None, np.inf
            for j in order:
                cand = apply_swap(s, int(x1[j]), int(x2[j]))
                te = time.perf_counter()
                rec, W = exact_eval(cand, X, r, return_W=True)
                t_exact += time.perf_counter() - te
                tr.record(cand, rec)
                d = (rec["cost"] - cur["cost"]) / cur["cost"]
                if guide == "ml":
                    Xtr.append(Fz[j])
                    ytr.append(d)
                    online_pred.append(float(pred[j]))
                    online_true.append(d)
                if d < best_d:
                    best_c, best_rec, best_W, best_d = cand, rec, W, d
            if best_c is not None and accept(best_d):
                s, cur, cur_W = best_c, best_rec, best_W
                accepted += 1

        # --- (re)latih surrogate secara online (active learning)
        if guide == "ml" and tr.n >= min(warmup, budget) and len(ytr) >= 10 and \
                (not fitted or it % retrain_every == 0):
            tm = time.perf_counter()
            model.fit(np.array(Xtr[-window:]), np.array(ytr[-window:]))
            t_model += time.perf_counter() - tm
            fitted = True

        if progress and it % 25 == 0:
            progress(tr.n / budget, f"{label or _label(guide, model_name)}: {tr.n}/{budget} evaluasi exact")

    res = _result(label or _label(guide, model_name), tr, time.perf_counter() - t0)
    res.update({
        "accepted": accepted,
        "final_cost": cur["cost"],
        "time_exact_per_eval_ms": 1000 * t_exact / max(1, budget),
        "time_cheap_per_cand_ms": 1000 * t_cheap / max(1, n_cheap),
        "time_model_s": t_model,
        "n_cheap": n_cheap,
    })
    if guide == "ml":
        if len(online_true) > 5:
            res["online_spearman"] = float(spearmanr(online_pred, online_true).statistic)
        res["diag"] = diag
    return res


WARM_POOL = 64


def _zscore(F: np.ndarray) -> np.ndarray:
    """Normalisasi z-score per kolom di dalam satu kolam kandidat."""
    sd = F.std(axis=0)
    sd[sd == 0] = 1.0
    return (F - F.mean(axis=0)) / sd


def _label(guide: str, model_name: str) -> str:
    return {"none": "B. Hill-climbing tanpa ML",
            "heuristic": "D. Penyaringan heuristik (tanpa ML)",
            "ml": f"C. Hill-climbing + ML ({model_name})"}[guide]


def _result(name: str, tr: Tracker, elapsed: float) -> dict:
    assert tr.best_sbox is not None and sc.is_bijective(tr.best_sbox)
    return {"method": name, "best": tr.best_rec, "best_sbox": tr.best_sbox.astype(int).tolist(),
            "curve_nl": tr.curve_nl.tolist(), "curve_cost": tr.curve_cost.tolist(),
            "elapsed_s": elapsed, "budget": tr.budget}


# ----------------------------------------------------------------------------
# Eksperimen pembanding
# ----------------------------------------------------------------------------
def make_plan(budget: int, seeds: list[int], models: list[str], k: int = DEFAULT_K, n_cand: int = 1000,
              top_m: int = 10, warmup: int = 200, X: float = DEFAULT_X, r: float = DEFAULT_R, temp0: float = 0.0,
              include_heuristic: bool = True, include_random: bool = True, diag_points: int = 3,
              feature_set: str = "full", coord_ablation: bool = False) -> list[dict]:
    """Daftar job (metode x seed) dengan anggaran evaluasi exact yang SAMA.

    coord_ablation=True menambah run C dengan fitur 8 koordinat + SAC saja (tanpa fitur puncak)
    untuk model pertama, agar terlihat seberapa besar peran pilihan fitur.
    """
    methods: list[tuple[str, dict]] = []
    if include_random:
        methods.append(("random", {}))
    methods.append(("none", {}))
    if include_heuristic:
        methods.append(("heuristic", {"feature_set": feature_set}))
    for mname in models:
        methods.append(("ml", {"model_name": mname, "feature_set": feature_set}))
    if coord_ablation and models:
        methods.append(("ml", {"model_name": models[0], "feature_set": "coord",
                               "label": f"C'. Hill-climbing + ML ({models[0]}, fitur koordinat saja)"}))
    common = {"budget": budget, "X": X, "r": r}
    hc = {"k": k, "n_cand": n_cand, "top_m": top_m, "warmup": warmup, "temp0": temp0}
    jobs = []
    for seed in seeds:
        for kind, kw in methods:
            job = {"kind": kind, "seed": seed, **common}
            if kind != "random":
                job.update(hc, guide=kind, diag_points=diag_points if kind == "ml" else 0,
                           retrain_every=5 if kw.get("model_name") == "Random Forest" else 1, **kw)
            jobs.append(job)
    return jobs


def run_job(job: dict, progress: ProgressFn = None) -> dict:
    """Jalankan satu job dari make_plan()."""
    kw = {k_: v for k_, v in job.items() if k_ != "kind"}
    if job["kind"] == "random":
        res = run_random_search(kw["budget"], kw["seed"], kw["X"], kw["r"], progress=progress)
    else:
        res = run_hill_climb(progress=progress, **kw)
    res["seed"] = job["seed"]
    return res


def run_comparison(budget: int, seeds: list[int], models: list[str], progress: ProgressFn = None,
                   **kwargs) -> dict:
    """Eksperimen pembanding berurutan (untuk demo live di UI). kwargs diteruskan ke make_plan()."""
    jobs = make_plan(budget, seeds, models, **kwargs)
    runs: list[dict] = []
    for i, job in enumerate(jobs):
        def sub(p: float, msg: str, _i=i, _s=job["seed"]) -> None:
            if progress:
                progress((_i + p) / len(jobs), f"seed {_s} · {msg}")
        runs.append(run_job(job, sub))
    cfg = {"budget": budget, "seeds": seeds, "models": models, **kwargs}
    return {"config": cfg, "runs": runs, "summary": summarize(runs)}


def summarize(runs: list[dict]) -> dict:
    """Rata-rata ± std per metode + uji Mann–Whitney U (B vs tiap C)."""
    by: dict[str, list[dict]] = {}
    for r_ in runs:
        by.setdefault(r_["method"], []).append(r_)
    table = []
    for name, rs in by.items():
        row = {"Metode": name, "n_seed": len(rs)}
        for key in ("NL_comp", "NL_coord", "DU", "LAP", "SAC", "BIC-SAC", "BIC-NL"):
            v = np.array([x["best"][key] for x in rs], dtype=float)
            row[f"{key}_mean"], row[f"{key}_std"] = float(v.mean()), float(v.std())
        fc = [x.get("final_cost", x["best"]["cost"]) for x in rs]
        row["cost_mean"] = float(np.mean([x["curve_cost"][-1] for x in rs]))
        row["final_cost_mean"] = float(np.mean(fc))
        row["NL_comp_max"] = int(max(x["best"]["NL_comp"] for x in rs))
        for lvl in (100, 102, 104):
            hits = [next((i + 1 for i, v in enumerate(x["curve_nl"]) if v >= lvl), None) for x in rs]
            ok = [h for h in hits if h is not None]
            row[f"reach{lvl}_frac"] = len(ok) / len(rs)
            row[f"reach{lvl}_eval"] = float(np.mean(ok)) if ok else None
        row["time_mean_s"] = float(np.mean([x["elapsed_s"] for x in rs]))
        sp = [x["online_spearman"] for x in rs if "online_spearman" in x]
        diags = [d for x in rs for d in x.get("diag", [])]
        if diags:
            row["spearman_mean"] = float(np.mean([d["spearman"] for d in diags]))
            row["spearman_heur_mean"] = float(np.mean([d["spearman_heur"] for d in diags]))
            row["precision_at_m_mean"] = float(np.mean([d["precision_at_m"] for d in diags]))
            row["precision_random"] = float(np.mean([d["precision_random"] for d in diags]))
            row["frac_improving_true"] = float(np.mean([d["frac_improving_true"] for d in diags]))
            row["frac_improving_topm"] = float(np.mean([d["frac_improving_topm"] for d in diags]))
        if sp:
            row["online_spearman_mean"] = float(np.mean(sp))
        table.append(row)

    tests = []
    base = by.get("B. Hill-climbing tanpa ML", [])
    for name, rs in by.items():
        if name.startswith(("C.", "C'", "D.")):
            for key, alt in (("NL_comp", "greater"), ("cost", "less")):
                a = [x["best"]["NL_comp"] if key == "NL_comp" else x["curve_cost"][-1] for x in rs]
                b = [x["best"]["NL_comp"] if key == "NL_comp" else x["curve_cost"][-1] for x in base]
                if len(a) >= 2 and len(b) >= 2:
                    try:
                        p = float(mannwhitneyu(a, b, alternative=alt).pvalue)
                    except ValueError:
                        p = 1.0
                    tests.append({"Pembanding": f"{name} vs B", "Metrik": key,
                                  "Hipotesis": f"{name[:2]} {'>' if alt == 'greater' else '<'} B",
                                  "p-value": p})
    return {"table": table, "tests": tests}


def best_overall(result: dict) -> dict:
    """Run dengan S-box terbaik (leksikografis) dari seluruh run metode C."""
    cands = [r_ for r_ in result["runs"] if r_["method"].startswith("C.")] or result["runs"]
    return min(cands, key=lambda r_: lex_key(r_["best"]))
