"""
run_experiment.py
Eksperimen offline Mode 2 (anggaran besar, banyak seed, paralel) untuk S-box dari permutasi.

Hasil:
  results/mode2_results.json     ringkasan (rata-rata ± std, uji Mann–Whitney U, akurasi surrogate,
                                 kurva konvergensi rata-rata) + detail tiap run
  results/best_sbox_mode2.json   S-box terbaik (metode C) beserta metrik lengkap

Contoh:
  python run_experiment.py --budget 2000 --seeds 0 1 --jobs 4            # cek cepat
  python run_experiment.py --budget 20000 --seeds 0 1 2 3 4 --jobs 8     # eksperimen utama
"""
from __future__ import annotations

import os

# satu thread BLAS per proses, paralelisme lewat multiprocessing
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import ml_engine as ml
import ml_permutation as mp
import sbox_core as sc

CURVE_POINTS = 400


def downsample(curve: list, n: int = CURVE_POINTS) -> tuple[list[int], list]:
    """Ambil ≤ n titik kurva (indeks evaluasi 1-based) agar file JSON tetap kecil."""
    idx = np.unique(np.linspace(0, len(curve) - 1, min(n, len(curve))).astype(int))
    return (idx + 1).tolist(), [curve[i] for i in idx]


def public_metrics(sbox: np.ndarray) -> dict:
    """Metrik lengkap yang bisa diserialisasi JSON."""
    e = sc.evaluate_full(np.asarray(sbox))
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in e.items()}
    out["bijective"] = bool(out["bijective"])
    out["balanced"] = bool(out["balanced"])
    return out


def mode1_reference(seed: int = 0) -> dict:
    """S-box Mode 1 (ML-affine, Ridge + fitur struktur baris) sebagai pembanding."""
    K, s = ml.mode1_reference_sbox(seed)
    return {"sbox": s.astype(int).tolist(), "matrix": sc.matrix_to_rows(K), "metrics": public_metrics(s)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=20000, help="evaluasi exact per run (sama untuk semua metode)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="+", default=mp.MODEL_NAMES, choices=mp.MODEL_NAMES)
    ap.add_argument("--k", type=int, default=mp.DEFAULT_K, help="k untuk KNN")
    ap.add_argument("--n-cand", type=int, default=1000, help="kandidat swap per iterasi")
    ap.add_argument("--top-m", type=int, default=10, help="kandidat yang dievaluasi exact per iterasi")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--X", type=float, default=mp.DEFAULT_X)
    ap.add_argument("--r", type=float, default=mp.DEFAULT_R)
    ap.add_argument("--temp0", type=float, default=0.0, help=">0 = simulated annealing ringan")
    ap.add_argument("--features", choices=["full", "coord"], default="full")
    ap.add_argument("--no-random", action="store_true")
    ap.add_argument("--no-heuristic", action="store_true")
    ap.add_argument("--no-coord-ablation", action="store_true")
    ap.add_argument("--no-mode1", action="store_true", help="lewati pembuatan S-box Mode 1 pembanding")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    jobs = mp.make_plan(a.budget, a.seeds, a.models, k=a.k, n_cand=a.n_cand, top_m=a.top_m,
                        warmup=a.warmup, X=a.X, r=a.r, temp0=a.temp0,
                        include_heuristic=not a.no_heuristic, include_random=not a.no_random,
                        feature_set=a.features, coord_ablation=not a.no_coord_ablation)
    print(f"{len(jobs)} run × {a.budget} evaluasi exact, {a.jobs} proses paralel")
    t0 = time.time()
    runs: list[dict] = []
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(mp.run_job, j): j for j in jobs}
        for f in as_completed(futs):
            r_ = f.result()
            runs.append(r_)
            b = r_["best"]
            print(f"[{len(runs):3d}/{len(jobs)}] seed {r_['seed']} {r_['method']:<55s} "
                  f"NL={b['NL_comp']} DU={b['DU']} LAP={b['LAP']:.4f} ({r_['elapsed_s']:.0f}s)", flush=True)
    runs.sort(key=lambda x: (x["seed"], x["method"]))
    summary = mp.summarize(runs)

    # kurva konvergensi rata-rata per metode
    curves = {}
    for name in dict.fromkeys(r_["method"] for r_ in runs):
        rs = [r_ for r_ in runs if r_["method"] == name]
        nl = np.mean([r_["curve_nl"] for r_ in rs], axis=0).tolist()
        cost = np.mean([r_["curve_cost"] for r_ in rs], axis=0).tolist()
        x, nl_d = downsample(nl)
        _, cost_d = downsample(cost)
        curves[name] = {"eval": x, "NL_comp_mean": nl_d, "cost_mean": cost_d}

    slim_runs = []
    for r_ in runs:
        d = {k: v for k, v in r_.items() if k not in ("curve_nl", "curve_cost")}
        d["curve_eval"], d["curve_nl"] = downsample(r_["curve_nl"])
        _, d["curve_cost"] = downsample(r_["curve_cost"])
        slim_runs.append(d)

    best = mp.best_overall({"runs": runs})
    best_s = np.array(best["best_sbox"])
    assert sc.is_bijective(best_s), "S-box terbaik tidak bijektif!"
    for r_ in runs:
        assert sc.is_bijective(np.array(r_["best_sbox"])), r_["method"]

    refs = {
        "AES": public_metrics(sc.construct_sbox(sc.rows_to_matrix(sc.K_AES_ROWS))),
        "S-box44 (paper)": public_metrics(sc.construct_sbox(sc.rows_to_matrix(sc.K44_ROWS))),
    }
    mode1 = None if a.no_mode1 else mode1_reference()

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": time.time() - t0,
            "python": platform.python_version(), "cpu": platform.processor() or platform.machine(),
            "jobs": a.jobs}
    res = {"meta": meta, "config": {**vars(a), "seeds": a.seeds}, "summary": summary, "curves": curves,
           "references": refs, "mode1": mode1, "runs": slim_runs}
    (out / "mode2_results.json").write_text(json.dumps(res, indent=1))
    (out / "best_sbox_mode2.json").write_text(json.dumps({
        "method": best["method"], "seed": best["seed"], "config": res["config"],
        "sbox": best["best_sbox"], "metrics": public_metrics(best_s)}, indent=1))

    print(f"\nSelesai dalam {time.time() - t0:.0f}s. Ringkasan (rata-rata ± std atas {len(a.seeds)} seed):")
    for row in summary["table"]:
        print(f"  {row['Metode']:<58s} NL={row['NL_comp_mean']:.1f}±{row['NL_comp_std']:.1f} "
              f"(maks {row['NL_comp_max']})  DU={row['DU_mean']:.1f}  LAP={row['LAP_mean']:.4f}  "
              f"Spearman={row.get('spearman_mean', float('nan')):.2f}")
    for t in summary["tests"]:
        print(f"  Mann–Whitney {t['Pembanding']:<60s} {t['Metrik']:<8s} p={t['p-value']:.4f}")
    print(f"S-box terbaik: {best['method']} seed {best['seed']} -> NL={best['best']['NL_comp']}, "
          f"DU={best['best']['DU']}, LAP={best['best']['LAP']:.4f}")


if __name__ == "__main__":
    main()
