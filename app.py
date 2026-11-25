"""
Streamlit app: Konstruksi S-box dengan Pendekatan Machine Learning
Basis jurnal: Alamsyah et al. (2025), Nonlinear Dynamics 113:3869-3890
"AES S-box modification uses affine matrices exploration for increased S-box strength"

Jalankan lokal:  streamlit run app.py
"""
import io
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import ml_engine as ml
import sbox_core as sc

st.set_page_config(page_title="S-Box ML Constructor", page_icon="🔐", layout="wide")

PAPER_REF = ("Alamsyah, Setiawan, A., Putra, A.T., Budiman, K., Muslim, M.A., Salahudin, S.N., "
             "Prasetiyo, B. (2025). AES S-box modification uses affine matrices exploration for "
             "increased S-box strength. *Nonlinear Dynamics*, 113, 3869–3890. "
             "https://doi.org/10.1007/s11071-024-10414-3")

# Nilai yang dilaporkan di paper (Tabel 19 & Tabel 20)
PAPER_VALUES = {
    "AES S-box": {"NL": 112, "SAC": 0.50488, "BIC-NL": 112, "BIC-SAC": 0.50460, "LAP": 0.0625, "DAP": 0.015625},
    "S-box44 (paper)": {"NL": 112, "SAC": 0.50073, "BIC-NL": 112, "BIC-SAC": 0.50237, "LAP": 0.0625,
                        "DAP": 0.015625, "SV": 16.0031},
}


# ----------------------------------------------------------------------------
# Helper tampilan
# ----------------------------------------------------------------------------
def sbox_df(sbox, hexmode=False):
    g = sc.sbox_to_grid(sbox)
    cols = [format(i, "X") for i in range(16)]
    if hexmode:
        g = np.vectorize(lambda v: format(int(v), "02X"))(g)
    return pd.DataFrame(g, index=cols, columns=cols)


def matrix_df(K):
    return pd.DataFrame({"Baris": [f"r{i}" for i in range(8)], "Bit (b0..b7)": sc.matrix_to_rows(K)})


def metrics_row(name, r):
    return {
        "S-box": name,
        "Bijektif": "✅" if r["bijective"] else "❌",
        "Balance": "✅" if r["balanced"] else "❌",
        "NL": r["NL"], "SAC": round(r["SAC"], 5), "BIC-NL": r["BIC-NL"],
        "BIC-SAC": round(r["BIC-SAC"], 5), "LAP": round(r["LAP"], 6), "DAP": round(r["DAP"], 6),
        "S (Eq.9)": round(r["S"], 6), "SV (Eq.20)": round(r["SV"], 6),
        "Fixed point": r["fixed_points"],
    }


def heatmap(M, title, fmt=".3f"):
    df = pd.DataFrame(M, index=[f"{i}" for i in range(8)], columns=[f"{j}" for j in range(8)])
    long = df.reset_index().melt(id_vars="index", var_name="kolom", value_name="nilai").dropna()
    long = long.rename(columns={"index": "baris"})
    base = alt.Chart(long).encode(x=alt.X("kolom:O", title="bit output"), y=alt.Y("baris:O", title="bit"))
    rect = base.mark_rect().encode(color=alt.Color("nilai:Q", scale=alt.Scale(scheme="blueorange", domainMid=0.5),
                                                   legend=None))
    text = base.mark_text(fontSize=10).encode(text=alt.Text("nilai:Q", format=fmt))
    return (rect + text).properties(title=title, height=260)


def sbox_download(sbox, name):
    buf = io.StringIO()
    buf.write(f"# {name}\n")
    for row in sc.sbox_to_grid(sbox):
        buf.write(", ".join(str(int(v)) for v in row) + "\n")
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def cached_full_eval(sbox_tuple):
    r = sc.evaluate(np.array(sbox_tuple, dtype=np.uint8), full=True)
    return r


@st.cache_data(show_spinner=False)
def cached_circulant(c):
    return pd.DataFrame(ml.circulant_exploration(c))


def show_sbox_detail(sbox, name, key):
    r = cached_full_eval(tuple(int(v) for v in sbox))
    c1, c2 = st.columns([3, 2])
    with c1:
        hexmode = st.toggle("Tampilkan heksadesimal", key=f"hex_{key}")
        st.dataframe(sbox_df(sbox, hexmode), width="stretch", height=600)
        st.download_button("⬇️ Unduh S-box (.txt)", sbox_download(sbox, name), file_name=f"{key}.txt",
                           key=f"dl_{key}")
    with c2:
        tbl = pd.DataFrame([
            ("Bijektif", "Ya" if r["bijective"] else "Tidak", "Ya"),
            ("Balance", "Ya" if r["balanced"] else "Tidak", "Ya"),
            ("Nonlinearity (NL)", f"{r['NL']}", "maks (112)"),
            ("SAC", f"{r['SAC']:.5f}", "0.5"),
            ("BIC-NL", f"{r['BIC-NL']}", "maks (112)"),
            ("BIC-SAC", f"{r['BIC-SAC']:.5f}", "0.5"),
            ("LAP", f"{r['LAP']:.6f}", "min"),
            ("DAP", f"{r['DAP']:.6f}", "min"),
            ("S (Eq. 9)", f"{r['S']:.6f}", "0"),
            ("SV (Eq. 20)", f"{r['SV']:.6f}", "16 (batas utk NL=112)"),
            ("Fixed point", f"{r['fixed_points']}", "0"),
        ], columns=["Kriteria", "Nilai", "Ideal"])
        st.dataframe(tbl, hide_index=True, width="stretch")
        st.altair_chart(heatmap(r["_sac_matrix"], "Matriks SAC (baris = bit input dibalik)"), width="stretch")
        st.altair_chart(heatmap(r["_bicsac_matrix"], "Matriks BIC-SAC"), width="stretch")
    return r


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🔐 S-Box ML Constructor")
page = st.sidebar.radio("Menu", [
    "1. Ringkasan & Reproduksi Paper",
    "2. Eksplorasi Affine Sirkulan (metode paper)",
    "3. Konstruksi S-box dengan Machine Learning",
    "4. Uji S-box Sendiri",
    "Tentang",
])
st.sidebar.markdown("---")
c_hex = st.sidebar.text_input("Konstanta affine C (hex)", "63", help="Default C_AES = 0x63 (01100011)")
try:
    C_CONST = int(c_hex, 16) & 0xFF
except ValueError:
    C_CONST = sc.C_AES
    st.sidebar.error("Hex tidak valid, memakai 63.")
st.sidebar.caption(f"C = {C_CONST} = {C_CONST:08b}b")
st.sidebar.caption("Polinomial irreducible: x⁸ + x⁴ + x³ + x + 1 (0x11B)")


# ============================================================================
# 1. Reproduksi paper
# ============================================================================
if page.startswith("1"):
    st.title("Reproduksi S-box AES & S-box44")
    st.markdown(f"**Jurnal acuan:** {PAPER_REF}")
    st.markdown(r"""
Konstruksi S-box mengikuti persamaan (2) pada paper:

$$B(X) = \left(K \cdot X^{-1} + C\right) \bmod 2$$

dengan $X^{-1}$ = invers multiplikatif di GF(2⁸) modulo $x^8+x^4+x^3+x+1$,
$K$ = matriks affine 8×8, dan $C$ = konstanta 8-bit (0x63).
""")
    k_aes = sc.rows_to_matrix(sc.K_AES_ROWS)
    k_44 = sc.rows_to_matrix(sc.K44_ROWS)
    col1, col2, col3 = st.columns(3)
    col1.markdown("**Matriks affine K_AES**"); col1.dataframe(matrix_df(k_aes), hide_index=True)
    col2.markdown("**Matriks affine K44 (terbaik di paper)**"); col2.dataframe(matrix_df(k_44), hide_index=True)
    with col3:
        st.markdown("**Contoh perhitungan (paper Eq. 3)**")
        x = 0x0F
        xi = int(sc.INV_TABLE[x])
        s44 = sc.construct_sbox(k_44, C_CONST)
        st.code(f"X      = {x:3d} = {x:08b}\nX^-1   = {xi:3d} = {xi:08b}\nB(X)   = {int(s44[x]):3d} = {int(s44[x]):08b}",
                language="text")
        st.caption("Paper: X=15 → X⁻¹=199 → B(X)=214 (dengan C=0x63).")

    with st.expander("Tabel invers multiplikatif GF(2⁸) (Tabel 1 paper)"):
        st.dataframe(sbox_df(sc.INV_TABLE), width="stretch")

    s_aes = sc.construct_sbox(k_aes, C_CONST)
    tab_a, tab_b, tab_c = st.tabs(["S-box44", "AES S-box", "Perbandingan dengan nilai paper"])
    with tab_a:
        r44 = show_sbox_detail(s44, "S-box44", "sbox44")
    with tab_b:
        raes = show_sbox_detail(s_aes, "AES S-box", "sboxaes")
    with tab_c:
        r44 = cached_full_eval(tuple(int(v) for v in s44))
        raes = cached_full_eval(tuple(int(v) for v in s_aes))
        rows = []
        for label, r, ref in [("AES S-box", raes, PAPER_VALUES["AES S-box"]),
                              ("S-box44", r44, PAPER_VALUES["S-box44 (paper)"])]:
            for k, v in ref.items():
                rows.append({"S-box": label, "Kriteria": k, "Nilai paper": v,
                             "Hasil aplikasi": round(r[k], 6),
                             "Cocok": "✅" if abs(r[k] - v) < 6e-5 else "⚠️"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if C_CONST != sc.C_AES:
            st.info("Konstanta C diubah dari 0x63; nilai S-box berbeda dari paper, "
                    "tetapi seluruh metrik kekuatan tetap sama (konstanta tidak memengaruhi NL/SAC/BIC/LAP/DAP).")


# ============================================================================
# 2. Eksplorasi sirkulan
# ============================================================================
elif page.startswith("2"):
    st.title("Eksplorasi Matriks Affine Sirkulan (metode paper)")
    st.markdown("""
Matriks affine pada paper (K_AES, K4, K44, K81, K111, K128) berbentuk **sirkulan**: setiap baris adalah
rotasi kanan 1 bit dari baris sebelumnya, sehingga matriks ditentukan oleh **baris pertama (8 bit)**.
Dari **256** kemungkinan baris pertama, hanya **128** yang menghasilkan S-box **bijektif & balance**
(yaitu matriks invertible di GF(2)). Ini sesuai dengan klaim paper "128 affine matrices meet the criteria".
""")
    with st.spinner("Mengevaluasi 256 matriks sirkulan..."):
        df = cached_circulant(C_CONST)
    valid = df[df["invertible"]].copy()
    c1, c2, c3 = st.columns(3)
    c1.metric("Kandidat", len(df)); c2.metric("Valid (bijektif & balance)", len(valid))
    best = valid.loc[valid["S"].idxmin()]
    c3.metric("Terbaik", f"K{int(best['K_index'])}  (S = {best['S']:.5f})")

    chart = alt.Chart(valid).mark_bar().encode(
        x=alt.X("K_index:O", title="Indeks matriks valid (K1..K128)", axis=alt.Axis(labelOverlap=True)),
        y=alt.Y("S:Q", title="Nilai S (makin kecil makin baik)"),
        color=alt.condition(alt.datum.K_index == int(best["K_index"]), alt.value("#d62728"), alt.value("#4c78a8")),
        tooltip=["K_index", "first_row", alt.Tooltip("SAC:Q", format=".5f"),
                 alt.Tooltip("BIC-SAC:Q", format=".5f"), alt.Tooltip("S:Q", format=".6f")],
    ).properties(height=320)
    st.altair_chart(chart, width="stretch")

    show = valid[["K_index", "first_row", "SAC", "BIC-SAC", "S"]].sort_values("S")
    st.dataframe(show.style.format({"SAC": "{:.5f}", "BIC-SAC": "{:.5f}", "S": "{:.6f}"}),
                 hide_index=True, width="stretch", height=350)
    st.download_button("⬇️ Unduh hasil eksplorasi (.csv)", df.to_csv(index=False), "eksplorasi_sirkulan.csv")

    st.subheader("Lihat S-box dari matriks tertentu")
    pick = st.selectbox("Pilih indeks K", valid["K_index"].astype(int).tolist(),
                        index=int(best["K_index"]) - 1)
    fr = valid[valid["K_index"] == pick]["first_row"].iloc[0]
    K = sc.circulant_from_first_row(int(fr, 2))
    st.dataframe(matrix_df(K), hide_index=True)
    show_sbox_detail(sc.construct_sbox(K, C_CONST), f"S-box K{pick}", f"circ_{pick}")


# ============================================================================
# 3. ML
# ============================================================================
elif page.startswith("3"):
    st.title("Konstruksi S-box dengan Pendekatan Machine Learning")
    st.markdown("""
**Masalah:** paper hanya menjelajah matriks sirkulan (128 valid). Jika ruang pencarian diperluas ke
**semua matriks affine 8×8 yang invertible (± 5,3 × 10¹⁸ matriks)**, evaluasi brute-force tidak mungkin.

**Solusi ML (surrogate model):**
1. **Dataset** – bangkitkan matriks affine acak yang invertible, konstruksi S-box-nya, hitung SAC & BIC-SAC (label).
2. **Training** – latih model (KNN, Random Forest, Ridge) untuk memprediksi SAC & BIC-SAC dari matriks affine.
3. **Screening** – model memprediksi nilai S untuk puluhan ribu kandidat baru dalam hitungan detik.
4. **Verifikasi** – hanya *top-k* kandidat hasil prediksi yang dikonstruksi & diuji penuh (NL, SAC, BIC, LAP, DAP).
""")
    with st.expander("💡 Mengapa hanya SAC & BIC-SAC yang diprediksi?"):
        st.markdown("""
- Transformasi affine yang invertible **tidak mengubah** NL, BIC-NL, LAP, dan DAP dari fungsi invers
  GF(2⁸). Semua kandidat valid selalu NL = 112, BIC-NL = 112, LAP = 0,0625, DAP = 0,015625
  (lihat juga Tabel 9, 11, 13, 14 di paper: semua S-box usulan nilainya sama).
- Karena itu paper memilih S-box terbaik dengan **nilai S = (|SAC−0,5| + |BIC-SAC−0,5|)/2** (Eq. 9).
- Bit ke-j output = (baris ke-j matriks K)·X⁻¹ ⊕ c_j, sehingga **SAC hanya bergantung pada nilai tiap baris**
  dan **BIC-SAC hanya pada XOR pasangan baris**. Fitur "struktur baris" memanfaatkan sifat ini.
""")

    st.subheader("Langkah 1 — Bangkitkan dataset")
    c1, c2 = st.columns(2)
    n_data = c1.slider("Jumlah matriks affine (data latih + uji)", 500, 6000, 2500, 500)
    seed = c2.number_input("Random seed", 0, 99999, 42)
    if st.button("🎲 Bangkitkan dataset", type="primary"):
        rng = np.random.default_rng(int(seed))
        prog = st.progress(0.0, "Membangkitkan matriks invertible...")
        Ks = ml.random_invertible_matrices(n_data, rng)
        Y = np.zeros((n_data, 3))
        step = max(1, n_data // 50)
        t = time.time()
        for i in range(0, n_data, step):
            Y[i:i + step] = ml.label_matrices(Ks[i:i + step], C_CONST)
            prog.progress(min(1.0, (i + step) / n_data), f"Mengonstruksi & menguji S-box {min(i + step, n_data)}/{n_data}")
        prog.empty()
        st.session_state["data"] = (Ks, Y)
        st.session_state.pop("models", None)
        st.session_state.pop("search", None)
        st.success(f"Dataset siap: {n_data} matriks ({time.time() - t:.1f} detik).")

    if "data" in st.session_state:
        Ks, Y = st.session_state["data"]
        d = pd.DataFrame(Y, columns=["SAC", "BIC-SAC", "S"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah data", len(d)); c2.metric("Rata-rata S", f"{d['S'].mean():.5f}")
        c3.metric("S terbaik di dataset", f"{d['S'].min():.6f}")
        hist = alt.Chart(d).mark_bar().encode(x=alt.X("S:Q", bin=alt.Bin(maxbins=50), title="Nilai S"),
                                              y=alt.Y("count()", title="Jumlah")).properties(height=200)
        st.altair_chart(hist, width="stretch")

        st.subheader("Langkah 2 — Latih & bandingkan model")
        c1, c2, c3 = st.columns(3)
        feat_label = c1.selectbox("Representasi fitur", list(ml.FEATURE_SETS.keys()))
        feat_kind = ml.FEATURE_SETS[feat_label]
        model_names = c2.multiselect("Model", ml.MODEL_NAMES, default=ml.MODEL_NAMES)
        k_nn = c3.slider("k (untuk KNN)", 1, 25, 7)
        if st.button("🧠 Latih model", type="primary", disabled=not model_names):
            with st.spinner("Melatih model..."):
                res, models, split, _ = ml.train_and_compare(Ks, Y, feat_kind, model_names, k=k_nn)
            st.session_state["models"] = (models, feat_kind, res)
            st.session_state.pop("search", None)

        if "models" in st.session_state:
            models, feat_kind_tr, res = st.session_state["models"]
            rdf = pd.DataFrame(res)
            st.dataframe(rdf.style.format({c: "{:.4f}" for c in rdf.columns if c != "Model"}),
                         hide_index=True, width="stretch")
            st.caption("R² mendekati 1 = prediksi akurat. Korelasi peringkat & Precision@10% mengukur kemampuan "
                       "model menemukan kandidat terbaik — yang paling penting untuk screening.")
            if feat_kind_tr == "raw":
                st.warning("Dengan 64 bit mentah, model hampir tidak bisa memprediksi (R² ≈ 0). "
                           "Coba fitur *struktur baris* untuk melihat perbedaannya.")

            st.subheader("Langkah 3 — Pencarian S-box berpandu ML")
            c1, c2, c3 = st.columns(3)
            model_pick = c1.selectbox("Model untuk screening", list(models.keys()),
                                      index=len(models) - 1)
            n_screen = c2.slider("Jumlah kandidat yang disaring", 2000, 50000, 20000, 2000)
            top_k = c3.slider("Top-k yang diuji penuh", 5, 200, 50, 5)
            if st.button("🚀 Cari S-box terbaik", type="primary"):
                rng = np.random.default_rng(int(seed) + 1)
                with st.spinner(f"Menyaring {n_screen} kandidat dengan {model_pick}..."):
                    out = ml.ml_guided_search(models[model_pick], feat_kind_tr, n_screen, top_k, rng, C_CONST)
                with st.spinner(f"Baseline: {top_k} kandidat acak tanpa ML..."):
                    _, Yr = ml.random_search(top_k, np.random.default_rng(int(seed) + 2), C_CONST)
                st.session_state["search"] = (out, Yr, model_pick, n_screen, top_k)

            if "search" in st.session_state:
                out, Yr, model_pick, n_screen, top_k = st.session_state["search"]
                best_i = int(np.argmin(out["true"][:, 2]))
                Kb = out["matrices"][best_i]
                sbest = sc.construct_sbox(Kb, C_CONST)
                t = out["time"]
                st.success(f"Selesai. Bangkitkan {t['generate']:.1f}s · prediksi {t['predict']:.2f}s · "
                           f"uji top-{top_k} {t['evaluate_top']:.2f}s")

                circ = cached_circulant(C_CONST)
                cmp = pd.DataFrame([
                    {"Metode": "AES S-box", "Kandidat dievaluasi": 1, "S terbaik": 0.004742},
                    {"Metode": "Paper: eksplorasi sirkulan (S-box44)", "Kandidat dievaluasi": 128,
                     "S terbaik": circ["S"].min()},
                    {"Metode": f"Acak tanpa ML ({top_k} kandidat)", "Kandidat dievaluasi": top_k,
                     "S terbaik": Yr[:, 2].min()},
                    {"Metode": f"ML ({model_pick}) — saring {n_screen}, uji {top_k}",
                     "Kandidat dievaluasi": top_k, "S terbaik": out["true"][:, 2].min()},
                ])
                st.dataframe(cmp.style.format({"S terbaik": "{:.6f}"}), hide_index=True, width="stretch")

                top_df = pd.DataFrame({
                    "Peringkat prediksi": np.arange(1, len(out["pred_S"]) + 1),
                    "S prediksi": out["pred_S"], "S aktual": out["true"][:, 2],
                    "SAC aktual": out["true"][:, 0], "BIC-SAC aktual": out["true"][:, 1],
                })
                sc_chart = alt.Chart(top_df).mark_circle(size=60).encode(
                    x=alt.X("S prediksi:Q"), y=alt.Y("S aktual:Q"),
                    tooltip=["Peringkat prediksi", alt.Tooltip("S prediksi:Q", format=".6f"),
                             alt.Tooltip("S aktual:Q", format=".6f")]).properties(
                    height=260, title="Prediksi vs aktual untuk kandidat top-k")
                st.altair_chart(sc_chart, width="stretch")
                with st.expander("Tabel kandidat top-k"):
                    st.dataframe(top_df.style.format({c: "{:.6f}" for c in top_df.columns if c != "Peringkat prediksi"}),
                                 hide_index=True, width="stretch")

                st.subheader("🏆 S-box terbaik hasil ML")
                st.markdown("**Matriks affine:**")
                st.dataframe(matrix_df(Kb), hide_index=True)
                rbest = show_sbox_detail(sbest, "S-box ML", "sbox_ml")

                s44 = sc.construct_sbox(sc.rows_to_matrix(sc.K44_ROWS), C_CONST)
                saes = sc.construct_sbox(sc.rows_to_matrix(sc.K_AES_ROWS), C_CONST)
                final = pd.DataFrame([
                    metrics_row("AES S-box", cached_full_eval(tuple(int(v) for v in saes))),
                    metrics_row("S-box44 (paper)", cached_full_eval(tuple(int(v) for v in s44))),
                    metrics_row(f"S-box ML ({model_pick})", rbest),
                ])
                st.markdown("**Perbandingan akhir**")
                st.dataframe(final, hide_index=True, width="stretch")


# ============================================================================
# 4. Uji S-box sendiri
# ============================================================================
elif page.startswith("4"):
    st.title("Uji S-box Sendiri")
    st.markdown("Tempel 256 nilai (desimal, atau hex dengan awalan 0x), dipisah koma/spasi/baris baru, "
                "atau unggah file .txt/.csv.")
    up = st.file_uploader("Unggah file", type=["txt", "csv"])
    default = " ".join(str(int(v)) for v in sc.construct_sbox(sc.rows_to_matrix(sc.K44_ROWS)))
    text = up.read().decode("utf-8", errors="ignore") if up else st.text_area("Nilai S-box", default, height=200)
    if st.button("🧪 Uji", type="primary"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
        tokens = " ".join(lines).replace(",", " ").replace(";", " ").split()
        try:
            vals = [int(t, 0) for t in tokens]
        except ValueError:
            st.error("Ada nilai yang bukan angka.")
            st.stop()
        if len(vals) != 256 or not all(0 <= v <= 255 for v in vals):
            st.error(f"Harus tepat 256 nilai di rentang 0..255 (terbaca {len(vals)}).")
            st.stop()
        show_sbox_detail(np.array(vals, dtype=np.uint8), "S-box pengguna", "user")


# ============================================================================
# Tentang
# ============================================================================
else:
    st.title("Tentang Aplikasi")
    st.markdown(f"""
**Jurnal acuan:** {PAPER_REF}

**Kriteria pengujian** (sesuai paper):
| Kriteria | Ideal | Keterangan |
|---|---|---|
| Nonlinearity (NL) | maks. (120 batas teoretis, 112 untuk invers GF(2⁸)) | jarak Hamming minimum ke fungsi affine |
| SAC | 0,5 | peluang bit output berubah jika 1 bit input dibalik |
| BIC-NL | maks. | NL dari XOR dua bit output |
| BIC-SAC | 0,5 | SAC dari XOR dua bit output |
| LAP | min. | peluang aproksimasi linear terbesar |
| DAP | min. | peluang diferensial terbesar |
| S (Eq. 9) | 0 | (abs(SAC−0,5) + abs(BIC-SAC−0,5)) / 2 |
| SV (Eq. 20) | 0 | (120−NL) + abs(0,5−SAC) + (120−BIC-NL) + abs(0,5−BIC-SAC) |

**Catatan:** SAC dan BIC-SAC yang dilaporkan adalah *rata-rata* matriks. Rata-rata 0,5 tidak berarti
setiap entri bernilai 0,5 — lihat heatmap untuk sebaran per-entri.

**Modul:** `sbox_core.py` (GF(2⁸), konstruksi, pengujian), `ml_engine.py` (dataset, fitur, model, pencarian),
`app.py` (antarmuka Streamlit).
""")
