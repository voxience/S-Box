"""
mode2_ui.py
Halaman Streamlit "5. Mode 2 — Konstruksi Permutasi Berpandu ML".
Logika komputasi ada di ml_permutation.py; file ini hanya tampilan.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import ml_engine as ml
import ml_permutation as mp
import sbox_core as sc

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "mode2_results.json"
BEST_FILE = RESULTS_DIR / "best_sbox_mode2.json"


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


def load_saved() -> dict | None:
    """Hasil run_experiment.py jika ada (cache di-invalidate saat file berubah)."""
    if not RESULTS_FILE.exists():
        return None
    return _load_json(str(RESULTS_FILE), RESULTS_FILE.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _mode1_fallback(c: int) -> list[int]:
    _, s = ml.mode1_reference_sbox(0, n_data=800, n_screen=5000, c=c)
    return s.astype(int).tolist()


def _curves(res: dict) -> pd.DataFrame:
    """Kurva konvergensi rata-rata per metode (format panjang untuk Altair)."""
    rows = []
    if "curves" in res:
        for name, cv in res["curves"].items():
            for e, nl, c in zip(cv["eval"], cv["NL_comp_mean"], cv["cost_mean"]):
                rows.append({"Metode": name, "Evaluasi exact": e, "NL komponen terbaik": nl, "Cost terbaik": c})
        return pd.DataFrame(rows)
    names = dict.fromkeys(r["method"] for r in res["runs"])
    for name in names:
        rs = [r for r in res["runs"] if r["method"] == name]
        nl = np.mean([r["curve_nl"] for r in rs], axis=0)
        cost = np.mean([r["curve_cost"] for r in rs], axis=0)
        idx = np.unique(np.linspace(0, len(nl) - 1, min(300, len(nl))).astype(int))
        for i in idx:
            rows.append({"Metode": name, "Evaluasi exact": int(i + 1), "NL komponen terbaik": float(nl[i]),
                         "Cost terbaik": float(cost[i])})
    return pd.DataFrame(rows)


def _best_of(res: dict, prefix: str) -> dict | None:
    rs = [r for r in res["runs"] if r["method"].startswith(prefix)]
    return min(rs, key=lambda r: mp.lex_key(r["best"])) if rs else None


# ----------------------------------------------------------------------------
# Tampilan hasil eksperimen
# ----------------------------------------------------------------------------
def _pm(row: dict, key: str, fmt: str) -> str:
    return f"{row[f'{key}_mean']:{fmt}} ± {row[f'{key}_std']:{fmt}}"


def _reach(row: dict, lvl: int) -> str:
    frac, ev = row.get(f"reach{lvl}_frac"), row.get(f"reach{lvl}_eval")
    if frac is None:
        return "-"
    return f"{frac:.0%}" + (f" (≈{ev:,.0f} eval)" if ev else "")


def render_result(res: dict, key: str) -> None:
    """Tabel perbandingan metode, uji signifikansi, akurasi surrogate, dan grafik konvergensi."""
    cfg = res["config"]
    seeds = cfg["seeds"]
    st.caption(f"Anggaran: **{cfg['budget']:,} evaluasi exact per run** (sama untuk semua metode) · "
               f"{len(seeds)} seed {seeds} · kandidat/iterasi = {cfg.get('n_cand')} · top-m = {cfg.get('top_m')} · "
               f"cost X = {cfg.get('X')}, r = {cfg.get('r')} · k (KNN) = {cfg.get('k')}")
    tbl = res["summary"]["table"]
    df = pd.DataFrame([{
        "Metode": r["Metode"],
        "NL komponen (mean ± std)": _pm(r, "NL_comp", ".1f"),
        "NL komponen maks": r["NL_comp_max"],
        "NL koordinat": _pm(r, "NL_coord", ".1f"),
        "DU": _pm(r, "DU", ".1f"),
        "LAP": _pm(r, "LAP", ".4f"),
        "SAC": _pm(r, "SAC", ".4f"),
        "BIC-SAC": _pm(r, "BIC-SAC", ".4f"),
        "Cost akhir (mean)": f"{r['cost_mean']:.4g}",
        "Capai NL ≥ 102": _reach(r, 102),
        "Capai NL ≥ 104": _reach(r, 104),
        "Waktu/run (s)": round(r["time_mean_s"], 1),
    } for r in tbl])
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption("NL komponen = NL minimum atas semua 255 fungsi komponen b·S(x). Nilai pada baris adalah S-box "
               "terbaik (leksikografis: NL komponen → DU → LAP → |SAC−0,5|+|BIC-SAC−0,5|) dari tiap run. "
               "“Capai NL ≥ …” = persentase seed yang mencapai level itu & rata-rata evaluasi exact yang dibutuhkan.")

    ch = _curves(res)
    if not ch.empty:
        c1, c2 = st.columns(2)
        sel = alt.selection_point(fields=["Metode"], bind="legend")
        base = alt.Chart(ch).encode(
            x=alt.X("Evaluasi exact:Q"), color=alt.Color("Metode:N", legend=alt.Legend(orient="bottom", columns=2)),
            opacity=alt.condition(sel, alt.value(1), alt.value(0.15)))
        c1.altair_chart(base.mark_line(interpolate="step-after").encode(
            y=alt.Y("NL komponen terbaik:Q", scale=alt.Scale(zero=False), title="NL komponen terbaik (rata-rata seed)"),
            tooltip=["Metode", "Evaluasi exact", alt.Tooltip("NL komponen terbaik:Q", format=".2f")]
        ).add_params(sel).properties(height=340, title="Konvergensi NL vs jumlah evaluasi exact"), width="stretch")
        c2.altair_chart(base.mark_line().encode(
            y=alt.Y("Cost terbaik:Q", scale=alt.Scale(type="log", zero=False), title="Cost WHS terbaik (log)"),
            tooltip=["Metode", "Evaluasi exact", alt.Tooltip("Cost terbaik:Q", format=".4g")]
        ).add_params(sel).properties(height=340, title="Konvergensi cost WHS (makin kecil makin baik)"), width="stretch")

    tests = res["summary"]["tests"]
    if tests:
        st.markdown("**Uji signifikansi Mann–Whitney U (satu sisi) terhadap B (hill-climbing tanpa ML)**")
        tdf = pd.DataFrame(tests)
        tdf["Signifikan (α=0,05)"] = np.where(tdf["p-value"] < 0.05, "✅ ya", "—")
        st.dataframe(tdf.style.format({"p-value": "{:.4f}"}), hide_index=True, width="stretch")
        if len(seeds) < 4:
            st.caption("⚠️ Dengan < 4 seed, p-value minimum Mann–Whitney tidak bisa < 0,05. "
                       "Gunakan hasil offline (5 seed) untuk kesimpulan statistik.")
        else:
            st.caption("Metrik `NL_comp`: NL komponen terbaik lebih tinggi. Metrik `cost`: cost WHS terbaik lebih "
                       "rendah. NL bernilai diskrit (genap) sehingga sering seri — cost lebih peka.")

    acc = [r for r in tbl if "spearman_mean" in r]
    if acc:
        st.markdown("**Akurasi surrogate** (diukur pada kandidat acak yang dievaluasi exact khusus untuk "
                    "diagnostik — tidak dihitung anggaran)")
        st.dataframe(pd.DataFrame([{
            "Metode": r["Metode"],
            "Spearman prediksi vs Δcost aktual": round(r["spearman_mean"], 3),
            "Spearman heuristik 1 fitur (pembanding)": round(r["spearman_heur_mean"], 3),
            "Precision@top-m": round(r["precision_at_m_mean"], 3),
            "Precision@top-m acak": round(r["precision_random"], 3),
            "% kandidat acak yang memperbaiki": f"{r['frac_improving_true']:.1%}",
            "% top-m surrogate yang memperbaiki": f"{r['frac_improving_topm']:.1%}",
        } for r in acc]), hide_index=True, width="stretch")

    tm = [r for r in res["runs"] if "time_cheap_per_cand_ms" in r and r["method"].startswith(("C", "D"))]
    if tm:
        ex = np.mean([r["time_exact_per_eval_ms"] for r in tm])
        ch_ = np.mean([r["time_cheap_per_cand_ms"] for r in tm])
        c1, c2, c3 = st.columns(3)
        c1.metric("Evaluasi exact / S-box", f"{ex:.2f} ms", help="255 komponen Walsh + DDT + LAT + SAC + BIC")
        c2.metric("Fitur murah / kandidat", f"{ch_:.3f} ms", help="8 koordinat + SAC + entri puncak (inkremental)")
        c3.metric("Rasio", f"{ex / ch_:.0f}× lebih murah")


# ----------------------------------------------------------------------------
# S-box terbaik & tabel akhir
# ----------------------------------------------------------------------------
def render_best(res: dict, key: str, c_const: int, helpers: dict[str, Callable[..., Any]]) -> None:
    best = _best_of(res, "C.") or _best_of(res, "C'")
    if best is None:
        st.info("Belum ada run metode C (berpandu ML).")
        return
    s = np.array(best["best_sbox"], dtype=np.uint8)
    full = helpers["full_eval"](tuple(int(v) for v in s))
    assert sc.is_bijective(s)
    st.markdown(f"**Sumber:** {best['method']}, seed {best['seed']} — S-box bijektif ✅, dibangun dari "
                f"permutasi acak tanpa invers GF(2⁸) / matriks affine / S-box AES.")
    c1, c2 = st.columns([3, 2])
    with c1:
        hexmode = st.toggle("Tampilkan heksadesimal", key=f"m2hex_{key}")
        st.dataframe(helpers["sbox_df"](s, hexmode), width="stretch", height=600)
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Unduh S-box (.txt)", helpers["sbox_download"](s, "S-box Mode 2 (ML-permutasi)"),
                           file_name="sbox_mode2.txt", key=f"m2dl_{key}")
        d2.download_button("⬇️ Unduh S-box + metrik (.json)", json.dumps({
            "method": best["method"], "seed": best["seed"], "sbox": s.astype(int).tolist(),
            "metrics": {k: v for k, v in full.items() if not k.startswith("_")}}, indent=1, default=float),
            file_name="sbox_mode2.json", key=f"m2dlj_{key}")
    with c2:
        st.dataframe(pd.DataFrame([
            ("Bijektif / balance", f"{'Ya' if full['bijective'] else 'Tidak'} / {'Ya' if full['balanced'] else 'Tidak'}"),
            ("NL komponen (min 255 fungsi)", full["NL_comp"]),
            ("NL koordinat (min 8 bit, definisi paper)", full["NL"]),
            ("BIC-NL", full["BIC-NL"]),
            ("DU / DAP", f"{full['DU']} / {full['DAP']:.6f}"),
            ("LAP", f"{full['LAP']:.6f}"),
            ("SAC (mean / std / max dev)", f"{full['SAC']:.5f} / {full['SAC_std']:.5f} / {full['SAC_maxdev']:.5f}"),
            ("BIC-SAC", f"{full['BIC-SAC']:.5f}"),
            ("Fixed point / opposite", f"{full['fixed_points']} / {full['opposite_fixed_points']}"),
        ], columns=["Kriteria", "Nilai"]).astype(str), hide_index=True, width="stretch")
        st.altair_chart(helpers["heatmap"](full["_sac_matrix"], "Matriks SAC S-box Mode 2"), width="stretch")
        st.altair_chart(helpers["heatmap"](full["_bicsac_matrix"], "Matriks BIC-SAC S-box Mode 2"), width="stretch")

    st.subheader("Tabel perbandingan akhir")
    saes = sc.construct_sbox(sc.rows_to_matrix(sc.K_AES_ROWS), c_const)
    s44 = sc.construct_sbox(sc.rows_to_matrix(sc.K44_ROWS), c_const)
    if "search" in st.session_state:
        out = st.session_state["search"][0]
        Kb = out["matrices"][int(np.argmin(out["true"][:, 2]))]
        s_m1, m1_src = sc.construct_sbox(Kb, c_const), "dari menu 3 (sesi ini)"
    elif res.get("mode1") and c_const == sc.C_AES:
        s_m1, m1_src = np.array(res["mode1"]["sbox"], dtype=np.uint8), "dari run_experiment.py"
    else:
        with st.spinner("Membangun S-box Mode 1 pembanding (Ridge, fitur struktur baris)..."):
            s_m1, m1_src = np.array(_mode1_fallback(c_const), dtype=np.uint8), "dibangun otomatis"
    rows = [("AES S-box", saes, "Ya (invers GF(2⁸))"), ("S-box44 (paper)", s44, "Ya (invers GF(2⁸))"),
            (f"S-box Mode 1 — ML-affine ({m1_src})", s_m1, "Ya (invers GF(2⁸))"),
            (f"S-box Mode 2 — ML-permutasi ({best['method'][3:]})", s, "Tidak (hasil pencarian)")]
    for prefix, label in (("A.", "Random search (terbaik)"), ("B.", "Hill-climbing tanpa ML (terbaik)")):
        b = _best_of(res, prefix)
        if b:
            rows.append((label, np.array(b["best_sbox"], dtype=np.uint8), "Tidak (hasil pencarian)"))
    final = []
    for name, sb, guaranteed in rows:
        r = helpers["full_eval"](tuple(int(v) for v in sb))
        final.append({"S-box": name, "NL komponen": r["NL_comp"], "NL koordinat": r["NL"], "BIC-NL": r["BIC-NL"],
                      "DU": r["DU"], "DAP": round(r["DAP"], 6), "LAP": round(r["LAP"], 6),
                      "SAC": round(r["SAC"], 5), "BIC-SAC": round(r["BIC-SAC"], 5),
                      "Fixed point": r["fixed_points"], "Nilai dijamin konstruksi?": guaranteed})
    st.dataframe(pd.DataFrame(final), hide_index=True, width="stretch")
    st.caption("Nilai 112 / DU 4 pada tiga baris pertama berasal dari sifat aljabar fungsi invers GF(2⁸) — "
               "sama untuk matriks affine invertible mana pun. Baris Mode 2 / random / hill-climbing "
               "adalah hasil pencarian murni pada ruang permutasi (256! kemungkinan).")


# ----------------------------------------------------------------------------
# Halaman
# ----------------------------------------------------------------------------
def render(c_const: int, helpers: dict[str, Callable[..., Any]]) -> None:
    st.title("Mode 2 — Konstruksi S-box dari Permutasi dengan Optimasi Berpandu ML")
    st.markdown("""
**Mengapa perlu Mode 2?** Pada Mode 1, S-box dibangun sebagai $B(X) = K\\cdot X^{-1} + C$. Transformasi affine
yang invertible **tidak mengubah** NL, BIC-NL, LAP, DU/DAP dari fungsi invers GF(2⁸), sehingga *setiap* matriks
invertible otomatis memberi NL = 112, LAP = 0,0625, DAP = 0,015625. Nilai tinggi itu **dijamin oleh
matematika AES**, bukan oleh ML — ML di Mode 1 hanya bisa memilih SAC/BIC-SAC.

**Mode 2** memulai dari **permutasi acak 0..255** (tanpa invers GF(2⁸), tanpa matriks affine, tanpa S-box AES)
dan memperbaikinya dengan **swap dua nilai output** (selalu bijektif). Di sini NL/DU/LAP **tidak dijamin** —
nilainya murni hasil pencarian, sehingga kontribusi ML dapat diukur secara adil.
""")
    st.info("ℹ️ Ekspektasi jujur: S-box hasil pencarian heuristik dari permutasi biasanya mencapai NL ≈ 100–106. "
            "Angka 112 milik AES adalah hasil konstruksi aljabar dan tidak dicapai di sini; tidak ada S-box AES "
            "yang diselipkan. Yang diukur di Mode 2 adalah apakah ML membuat pencarian lebih baik/lebih cepat "
            "dibanding tanpa ML **pada jumlah evaluasi exact yang sama**.")

    with st.expander("⚙️ Cara kerja (cost, fitur murah, surrogate, aturan adil)"):
        st.markdown(f"""
**1. Evaluasi exact (“mahal”)** — untuk setiap S-box: spektrum Walsh 255 fungsi komponen (= LAT penuh),
DDT penuh, SAC, BIC-NL, BIC-SAC. Satu evaluasi exact = **1 unit anggaran**, sama untuk semua metode.

**2. Cost yang dioptimasi** (spektrum Walsh–Hadamard):
$\\text{{cost}} = \\sum_{{b\\neq 0}}\\sum_a \\big|\\,|W_b(a)| - X\\big|^r$. Pangkat besar menekan entri puncak
spektrum (yang menentukan NL) sambil tetap memberi gradien halus, sehingga hill-climbing bisa naik bertahap.
S-box terbaik dipilih **leksikografis**: NL komponen ↑ → DU ↓ → LAP ↓ → |SAC−0,5|+|BIC-SAC−0,5| ↓.

**3. Fitur murah per kandidat swap** (dihitung inkremental, ribuan kandidat sekaligus):
- {", ".join(mp.FEATURE_NAMES)};
- (set fitur lengkap) {", ".join(mp.PEAK_FEATURE_NAMES)} — dari spektrum S-box *saat ini* yang sudah tersedia
  dari evaluasi exact sebelumnya, jadi tidak menambah evaluasi exact.

**4. Loop surrogate-assisted (metode C):** bangkitkan *n* kandidat swap → hitung fitur murah → surrogate
memprediksi Δcost → hanya **top-m** yang dievaluasi exact → terima yang terbaik bila cost turun → data exact
ditambahkan & surrogate **dilatih ulang online** (active learning). Warm-up awal = swap acak (dihitung anggaran).
Fitur dinormalisasi z-score per kolam kandidat agar skala tetap stabil sepanjang optimasi.

**5. Pembanding (anggaran sama):** (A) random search, (B) hill-climbing tanpa ML (1 swap acak = 1 evaluasi),
(C) hill-climbing + ML (KNN / Random Forest / Ridge), dan ablasi (D) penyaringan dengan **satu** fitur murah
tanpa ML — untuk menguji apakah ML memberi nilai lebih dari aturan sederhana.
""")

    saved = load_saved()
    tabs = st.tabs(["📊 Hasil eksperimen offline", "▶️ Demo live", "🏆 S-box terbaik & perbandingan akhir"])

    with tabs[0]:
        if saved:
            m = saved.get("meta", {})
            st.caption(f"Dari `results/mode2_results.json` — dibuat {m.get('created', '?')}, "
                       f"total {m.get('elapsed_s', 0) / 60:.1f} menit ({m.get('jobs', '?')} proses paralel).")
            render_result(saved, "saved")
        else:
            st.info("Belum ada hasil tersimpan. Jalankan di komputer sendiri:\n\n"
                    "`python run_experiment.py --budget 20000 --seeds 0 1 2 3 4 --jobs 8`\n\n"
                    "lalu commit folder `results/`. Sementara itu gunakan tab **Demo live**.")

    with tabs[1]:
        st.markdown("Demo dengan anggaran kecil agar selesai ± 1 menit di Streamlit Community Cloud (1 CPU).")
        with st.form("mode2_form"):
            c1, c2, c3, c4 = st.columns(4)
            models = c1.multiselect("Model surrogate", mp.MODEL_NAMES, default=["KNN"])
            k = c1.slider("k (KNN)", 1, 40, mp.DEFAULT_K)
            budget = c2.slider("Anggaran evaluasi exact / run", 500, 4000, 1000, 250)
            n_cand = c2.slider("Kandidat swap / iterasi", 200, 2000, 500, 100)
            top_m = c3.slider("Top-m dievaluasi exact", 5, 20, 10)
            warmup = c3.slider("Warm-up (swap acak)", 50, 500, 100, 50)
            seed0 = c4.number_input("Seed awal", 0, 9999, 0)
            n_seeds = c4.slider("Jumlah seed", 1, 3, 1)
            c5, c6, c7, c8 = st.columns(4)
            X = c5.number_input("Parameter cost X", 0.0, 64.0, float(mp.DEFAULT_X), 4.0)
            r = c6.slider("Parameter cost r", 2.0, 10.0, float(mp.DEFAULT_R), 0.5)
            inc_rand = c7.checkbox("Sertakan (A) random search", True)
            inc_heur = c8.checkbox("Sertakan (D) ablasi heuristik", True)
            n_methods = 1 + inc_rand + inc_heur + len(models)
            est = n_seeds * budget * (0.004 * (1 + inc_rand) + (0.004 + 0.07 * n_cand / 1000 / top_m)
                                      * (inc_heur + len(models)))
            st.caption(f"{n_methods} metode × {n_seeds} seed. Perkiraan waktu: ± {est:.0f} s (laptop modern; "
                       "Streamlit Cloud bisa 1,5–2× lebih lambat).")
            go = st.form_submit_button("🚀 Jalankan demo", type="primary")
        if go:
            prog = st.progress(0.0, "Menyiapkan...")
            t0 = time.time()
            res = mp.run_comparison(int(budget), [int(seed0) + i for i in range(n_seeds)], models,
                                    k=int(k), n_cand=int(n_cand), top_m=int(top_m), warmup=int(warmup),
                                    X=float(X), r=float(r), include_heuristic=inc_heur,
                                    include_random=inc_rand, diag_points=2,
                                    progress=lambda p, msg: prog.progress(min(1.0, p), msg))
            prog.empty()
            res["meta"] = {"elapsed_s": time.time() - t0}
            st.session_state["mode2_live"] = res
        live = st.session_state.get("mode2_live")
        if live:
            st.success(f"Demo selesai dalam {live['meta']['elapsed_s']:.0f} s. "
                       "Hasil disimpan di sesi (tidak hilang saat widget berubah).")
            render_result(live, "live")

    with tabs[2]:
        sources = {}
        if st.session_state.get("mode2_live"):
            sources["Demo live (sesi ini)"] = st.session_state["mode2_live"]
        if saved:
            sources["Eksperimen offline (results/)"] = saved
        if not sources:
            st.info("Jalankan demo live atau sediakan `results/mode2_results.json` terlebih dahulu.")
            return
        pick = st.radio("Sumber hasil", list(sources), horizontal=True, key="m2_source")
        render_best(sources[pick], "src" + str(list(sources).index(pick)), c_const, helpers)
