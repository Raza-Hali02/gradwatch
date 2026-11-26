import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# ==============================================================================
# 1. PAGE CONFIG & WHITE / BLUE / GRAY THEME
# ==============================================================================
st.set_page_config(
    page_title="GradWatch",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Clean white & soft slate-gray base */
    .stApp {
        background-color: #ffffff;
        color: #0f172a;
    }
    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 2.5rem;
        max-width: 1180px;
    }

    /* Inline code pills */
    code {
        background-color: #eff6ff !important;
        color: #1e40af !important;
        border: 1px solid #bfdbfe;
        border-radius: 6px;
        padding: 0.1rem 0.35rem;
        font-size: 0.85em;
    }

    /* Sidebar white/gray surface with high-contrast dark text */
    section[data-testid="stSidebar"] {
        background-color: #f1f5f9 !important;
        border-right: 1px solid #cbd5e1;
    }
    section[data-testid="stSidebar"] * {
        color: #0f172a;
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #334155 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding: 0.5rem 0.9rem;
        border-radius: 999px;
        margin-bottom: 0.3rem;
        background-color: transparent;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background-color: #1d4ed8 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked),
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) * {
        color: #ffffff !important;
        font-weight: 600;
    }

    /* Metric labels and values */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #0f172a !important;
    }

    /* Widget labels, radio options, and slider text */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label *,
    [data-testid="stSlider"] [data-testid="stThumbValue"],
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"] {
        color: #0f172a !important;
    }

    /* Blue accent for sliders & radio inputs instead of default red */
    [data-testid="stSlider"] [role="slider"] {
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
    }
    [data-testid="stSlider"] [data-testid="stSliderTrackActive"] {
        background-color: #1d4ed8 !important;
    }
    input[type="radio"] {
        accent-color: #1d4ed8 !important;
    }

    /* Sub-tabs inside container */
    button[data-baseweb="tab"],
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span {
        color: #334155 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] span {
        color: #1d4ed8 !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #1d4ed8 !important;
    }

    /* High-contrast light blue/white info & alert banners */
    [data-testid="stAlert"],
    [data-testid="stAlert"] > div,
    [data-testid="stNotification"] {
        background-color: #eff6ff !important;
        border: 1px solid #93c5fd !important;
        border-left: 4px solid #1d4ed8 !important;
        border-radius: 10px !important;
        color: #0f172a !important;
    }
    [data-testid="stAlert"] *,
    [data-testid="stNotification"] * {
        color: #0f172a !important;
    }

    /* Expander container, header bar (all states), and body */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] details[open] {
        background-color: #ffffff !important;
    }
    [data-testid="stExpander"] details summary,
    [data-testid="stExpander"] details[open] summary,
    [data-testid="stExpander"] details summary:hover,
    [data-testid="stExpander"] details[open] summary:hover,
    [data-testid="stExpander"] summary > div,
    [data-testid="stExpander"] details[open] summary > div {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    [data-testid="stExpander"] summary svg,
    [data-testid="stExpander"] details[open] summary svg {
        color: #0f172a !important;
        fill: #0f172a !important;
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    /* Light white/gray DataFrames */
    [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
    }
    [data-testid="stDataFrame"] * {
        color: #0f172a !important;
    }

    /* Primary & secondary buttons in blue / white / gray */
    div.stButton > button[kind="primary"] {
        background-color: #1d4ed8 !important;
        border-color: #1d4ed8 !important;
        color: #ffffff !important;
        border-radius: 999px;
        font-weight: 600;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1e40af !important;
        border-color: #1e40af !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #0f172a !important;
        border-radius: 999px;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #eff6ff !important;
        border-color: #93c5fd !important;
        color: #1d4ed8 !important;
    }

    /* Slate-blue & white Hero Banner (Beranda) */
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        color: #f8fafc;
        border: 1px solid #1e40af;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    }
    .hero-eyebrow {
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #bfdbfe;
        margin-bottom: 0.6rem;
        font-weight: 600;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 0.6rem 0;
        line-height: 1.15;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #e2e8f0;
        max-width: 650px;
        margin: 0 auto 1.35rem auto;
        line-height: 1.5;
    }
    .hero-badge {
        display: inline-block;
        padding: 0.48rem 1.15rem;
        border-radius: 999px;
        border: 1px solid #93c5fd;
        color: #eff6ff;
        font-weight: 600;
        font-size: 0.88rem;
        background: rgba(37, 99, 235, 0.35);
    }

    /* White & Gray Metric Cards + Subtle Blue Accent */
    .info-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-top: 3px solid #2563eb;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        height: 100%;
    }
    .info-card-label {
        font-size: 0.82rem;
        color: #475569;
        margin-bottom: 0.4rem;
        font-weight: 500;
    }
    .info-card-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1d4ed8;
    }

    /* Soft Gray & Blue Step Cards */
    .step-card {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        height: 100%;
    }
    .step-num {
        font-size: 1.3rem;
        font-weight: 800;
        color: #2563eb;
        margin-bottom: 0.3rem;
    }
    .step-desc {
        font-size: 0.9rem;
        color: #334155;
        line-height: 1.45;
    }

    /* Footer */
    .app-footer {
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #cbd5e1;
        text-align: center;
        font-size: 0.8rem;
        color: #64748b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. CONSTANTS, PATHS & TRANSLATION MAPS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "student-mat.csv")
MODEL_PATH = os.path.join(BASE_DIR, "student_performance_model.pkl")
FIG_1_PATH = os.path.join(BASE_DIR, "Figure_1.png")
FIG_2_PATH = os.path.join(BASE_DIR, "Figure_2.png")

SKEWED_COLS = ["absences", "failures", "traveltime", "Dalc", "Walc"]
LEAKAGE_COLS = ["G1", "G2", "G3", "target"]

CAT_OPTIONS = {
    "school": {"GP": "GP — Gabriel Pereira", "MS": "MS — Mousinho da Silveira"},
    "sex": {"F": "Perempuan (F)", "M": "Laki-laki (M)"},
    "address": {"U": "Perkotaan (Urban)", "R": "Pedesaan (Rural)"},
    "famsize": {"LE3": "≤ 3 Anggota Keluarga", "GT3": "> 3 Anggota Keluarga"},
    "Pstatus": {"T": "Tinggal Bersama (T)", "A": "Berpisah (A)"},
    "Mjob": {
        "teacher": "Guru",
        "health": "Tenaga Kesehatan",
        "services": "Layanan Publik / Sipil",
        "at_home": "Ibu Rumah Tangga",
        "other": "Lainnya",
    },
    "Fjob": {
        "teacher": "Guru",
        "health": "Tenaga Kesehatan",
        "services": "Layanan Publik / Sipil",
        "at_home": "Di Rumah",
        "other": "Lainnya",
    },
    "reason": {
        "home": "Dekat dengan Rumah",
        "reputation": "Reputasi Sekolah",
        "course": "Preferensi Mata Pelajaran",
        "other": "Lainnya",
    },
    "guardian": {"mother": "Ibu", "father": "Ayah", "other": "Wali Lainnya"},
    "schoolsup": {"yes": "Ya", "no": "Tidak"},
    "famsup": {"yes": "Ya", "no": "Tidak"},
    "paid": {"yes": "Ya", "no": "Tidak"},
    "activities": {"yes": "Ya", "no": "Tidak"},
    "nursery": {"yes": "Ya", "no": "Tidak"},
    "higher": {"yes": "Ya", "no": "Tidak"},
    "internet": {"yes": "Ya", "no": "Tidak"},
    "romantic": {"yes": "Ya", "no": "Tidak"},
}

ORD_OPTIONS = {
    "Medu": {
        0: "0 — Tidak Sekolah",
        1: "1 — Pendidikan Dasar (SD)",
        2: "2 — SMP (Kelas 5–9)",
        3: "3 — SMA / Sederajat",
        4: "4 — Perguruan Tinggi",
    },
    "Fedu": {
        0: "0 — Tidak Sekolah",
        1: "1 — Pendidikan Dasar (SD)",
        2: "2 — SMP (Kelas 5–9)",
        3: "3 — SMA / Sederajat",
        4: "4 — Perguruan Tinggi",
    },
    "traveltime": {
        1: "1 — < 15 menit",
        2: "2 — 15–30 menit",
        3: "3 — 30–60 menit",
        4: "4 — > 60 menit",
    },
    "studytime": {
        1: "1 — < 2 jam / minggu",
        2: "2 — 2–5 jam / minggu",
        3: "3 — 5–10 jam / minggu",
        4: "4 — > 10 jam / minggu",
    },
    "failures": {
        0: "0 — Tidak Pernah Gagal",
        1: "1 — 1 Kali Gagal",
        2: "2 — 2 Kali Gagal",
        3: "3 — ≥ 3 Kali Gagal",
    },
    "famrel": {
        1: "1 — Sangat Buruk",
        2: "2 — Buruk",
        3: "3 — Cukup",
        4: "4 — Baik",
        5: "5 — Sangat Baik",
    },
    "freetime": {
        1: "1 — Sangat Sedikit",
        2: "2 — Sedikit",
        3: "3 — Sedang",
        4: "4 — Banyak",
        5: "5 — Sangat Banyak",
    },
    "goout": {
        1: "1 — Sangat Jarang",
        2: "2 — Jarang",
        3: "3 — Sedang",
        4: "4 — Sering",
        5: "5 — Sangat Sering",
    },
    "Dalc": {
        1: "1 — Sangat Rendah",
        2: "2 — Rendah",
        3: "3 — Sedang",
        4: "4 — Tinggi",
        5: "5 — Sangat Tinggi",
    },
    "Walc": {
        1: "1 — Sangat Rendah",
        2: "2 — Rendah",
        3: "3 — Sedang",
        4: "4 — Tinggi",
        5: "5 — Sangat Tinggi",
    },
    "health": {
        1: "1 — Sangat Buruk",
        2: "2 — Buruk",
        3: "3 — Cukup",
        4: "4 — Baik",
        5: "5 — Sangat Baik",
    },
}


# ==============================================================================
# 3. CACHED DATA, PREPROCESSING & MODEL LOADERS
# ==============================================================================
@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    df = pd.read_csv(path, sep=sep)
    df["target"] = (df["G3"] >= 10).astype(int)
    return df


@st.cache_resource
def load_trained_artifacts(model_path: str, df: pd.DataFrame):
    artifact = joblib.load(model_path)

    X_raw = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
    y = df["target"]

    X_encoded = pd.get_dummies(X_raw, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )

    if isinstance(artifact, dict):
        model = artifact.get("model") or artifact.get("estimator")
        scaler = artifact.get("scaler")
        feature_cols = list(artifact.get("feature_columns", X_train.columns))
    else:
        model = artifact
        scaler = None
        feature_cols = list(getattr(model, "feature_names_in_", X_train.columns))

    if scaler is None:
        scaler = RobustScaler()
        scaler.fit(X_train[SKEWED_COLS])

    X_test_scaled = X_test.copy()
    X_test_scaled[SKEWED_COLS] = scaler.transform(X_test_scaled[SKEWED_COLS])
    X_test_aligned = X_test_scaled.reindex(columns=feature_cols, fill_value=0)

    return model, scaler, feature_cols, X_raw, X_test_aligned, y_test


def preprocess_single_row(
    row_dict: dict, X_raw_ref: pd.DataFrame, scaler: RobustScaler, feature_cols: list
) -> pd.DataFrame:
    single_df = pd.DataFrame([row_dict])
    combined = pd.concat([X_raw_ref, single_df], ignore_index=True)
    combined_encoded = pd.get_dummies(combined, drop_first=True)

    row_encoded = combined_encoded.iloc[[-1]].copy()
    row_encoded[SKEWED_COLS] = scaler.transform(row_encoded[SKEWED_COLS])
    return row_encoded.reindex(columns=feature_cols, fill_value=0)


# ==============================================================================
# 4. INITIALIZE STATE & LOAD RESOURCES
# ==============================================================================
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Beranda"
if "show_csv" not in st.session_state:
    st.session_state.show_csv = False
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
    st.error(
        "File `student-mat.csv` atau `student_performance_model.pkl` tidak ditemukan "
        f"di folder `{BASE_DIR}`."
    )
    st.stop()

df = load_dataset(DATA_PATH)
model, scaler, feature_cols, X_raw_ref, X_test_aligned, y_test = load_trained_artifacts(
    MODEL_PATH, df
)

y_pred_test = model.predict(X_test_aligned)
y_prob_test = model.predict_proba(X_test_aligned)[:, 1]
metrics_summary = {
    "logloss": log_loss(y_test, y_prob_test),
    "accuracy": accuracy_score(y_test, y_pred_test),
    "f1_macro": f1_score(y_test, y_pred_test, average="macro"),
    "roc_auc": roc_auc_score(y_test, y_prob_test),
    "cm": confusion_matrix(y_test, y_pred_test),
}


# ==============================================================================
# 5. SIDEBAR NAVIGATION
# ==============================================================================
menu_options = ["Beranda", "Dataset", "Prediksi", "Evaluasi Model"]

with st.sidebar:
    st.markdown("### GradWatch")
    st.caption("Sistem Peringatan Dini Siswa")
    st.markdown("")

    selected_menu = st.radio(
        "Navigasi",
        options=menu_options,
        key="nav_page",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption(
        "Hanya untuk deteksi dini akademik (G3 ≥ 10 = Lulus), "
        "bukan penilaian akhir mutlak."
    )


# ==============================================================================
# 6. PAGE 1: BERANDA
# ==============================================================================
if selected_menu == "Beranda":
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-eyebrow">STUDENT PERFORMANCE (UCI MAT-395)</div>
            <h1 class="hero-title">Prediksi Kelulusan Siswa.</h1>
            <p class="hero-subtitle">
                Klasifikasi risiko kelulusan siswa (Lulus vs Berisiko Tidak Lulus) dari 30 fitur
                demografis, akademik, dan perilaku tanpa kebocoran nilai ujian sebelumnya.
            </p>
            <div class="hero-badge">ROC-AUC {metrics_summary['roc_auc']:.4f} &bull; Log-Loss {metrics_summary['logloss']:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_cta1, col_cta2, col_cta3 = st.columns([1.2, 1, 1.2])
    with col_cta2:
        if st.button("Coba di menu Prediksi", type="primary", use_container_width=True):
            st.session_state["nav_page"] = "Prediksi"
            st.rerun()

    st.markdown("")
    st.markdown(
        "Aplikasi ini mengklasifikasikan siswa ke dalam kategori **Lulus (`G3 ≥ 10`)** atau "
        "**Berisiko Tidak Lulus (`G3 < 10`)** menggunakan pipeline `RobustScaler` + "
        "`SGDClassifier (log_loss, penalty='l1', alpha=0.01, class_weight='balanced')`."
    )
    st.markdown("")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-card-label">Jumlah sampel siswa</div>
                <div class="info-card-value">{len(df)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-card-label">Fitur awal dipakai</div>
                <div class="info-card-value">{X_raw_ref.shape[1]} Fitur</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-card-label">Target kelas</div>
                <div class="info-card-value">Lulus / Berisiko</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-num">1</div>
                <div class="step-desc">Pilih profil sampel siswa (0–394) atau atur parameter klik-saja di menu Prediksi.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-num">2</div>
                <div class="step-desc">Tekan <b>Jalankan Prediksi</b> untuk memproses encoding &amp; scaling secara otomatis.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-num">3</div>
                <div class="step-desc">Baca status kelulusan, probabilitas, dan perbandingan nilai G3 aktual.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    with st.expander("Tentang dataset & aturan preprocessing (Option B)"):
        st.markdown(
            """
- **Sumber Data:** `student-mat.csv` berisi **395 observasi** siswa dan **33 atribut asli** (0 missing value).
- **Aturan Target:** Nilai akhir `G3 ≥ 10` dikodekan sebagai **1 (Lulus)** sebanyak **265 siswa (67.1%)**, sedangkan `G3 < 10` sebagai **0 (Tidak Lulus)** sebanyak **130 siswa (32.9%)**.
- **Pencegahan Data Leakage:** Kolom `G1` dan `G2` dihapus agar model berfungsi sebagai sistem peringatan dini murni sejak awal semester.
- **Tanpa PCA:** Reduksi dimensi PCA tidak digunakan karena jumlah fitur (`~39` setelah One-Hot Encoding) jauh lebih kecil dari jumlah sampel (`n = 395`) dan interpretabilitas fitur tetap terjaga.
            """
        )


# ==============================================================================
# 7. PAGE 2: DATASET
# ==============================================================================
elif selected_menu == "Dataset":
    st.title("Dataset Student Performance")
    st.caption("Eksplorasi data historis 395 siswa matematika (UCI Machine Learning Repository).")

    tab_table, tab_stats, tab_viz = st.tabs(["Tabel Data", "Statistik", "Visualisasi"])

    with tab_table:
        btn_label = (
            "Sembunyikan CSV (student-mat.csv)"
            if st.session_state.show_csv
            else "Tampilkan CSV (student-mat.csv)"
        )
        if st.button(btn_label, type="secondary"):
            st.session_state.show_csv = not st.session_state.show_csv
            st.rerun()

        if st.session_state.show_csv:
            st.dataframe(df, use_container_width=True, height=430)
        else:
            st.info(
                "Tabel CSV disembunyikan agar tampilan tetap ringan. "
                "Klik tombol di atas untuk menampilkan seluruh 395 baris data."
            )

    with tab_stats:
        num_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "absences", "G1", "G2", "G3"]
        st.subheader("Ringkasan Statistik Variabel Numerik & Ordinal")
        st.dataframe(df[num_cols].describe().T.round(2), use_container_width=True)

        st.subheader("Distribusi Kelas Target (G3 ≥ 10)")
        target_counts = pd.DataFrame(
            {
                "Kategori": ["Lulus (G3 ≥ 10)", "Tidak Lulus (G3 < 10)"],
                "Jumlah Siswa": [(df["target"] == 1).sum(), (df["target"] == 0).sum()],
                "Persentase (%)": [
                    round((df["target"] == 1).mean() * 100, 1),
                    round((df["target"] == 0).mean() * 100, 1),
                ],
            }
        )
        st.dataframe(target_counts, use_container_width=True, hide_index=True)

    with tab_viz:
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("**Distribusi Nilai Akhir (G3)**")
            g3_hist = df["G3"].value_counts().sort_index()
            st.bar_chart(g3_hist)
        with v2:
            st.markdown("**Rata-Rata Nilai G3 Berdasarkan Waktu Belajar (`studytime`)**")
            study_map = {1: "< 2 jam", 2: "2–5 jam", 3: "5–10 jam", 4: "> 10 jam"}
            avg_study = df.groupby("studytime")["G3"].mean().rename(index=study_map)
            st.bar_chart(avg_study)

        st.markdown("**Distribusi Ketidakhadiran (`absences`) — Alasan Penggunaan `RobustScaler`**")
        abs_counts = df["absences"].value_counts().sort_index()
        st.bar_chart(abs_counts)

        if os.path.exists(FIG_1_PATH):
            st.markdown("---")
            st.markdown("**Visualisasi Eksplorasi Tambahan (`Figure_1.png`)**")
            st.image(
                FIG_1_PATH,
                caption="Distribusi G3, sebaran G3 berdasarkan waktu belajar, dan boxplot outlier ketidakhadiran (absences).",
                use_container_width=True,
            )


# ==============================================================================
# 8. PAGE 3: PREDIKSI
# ==============================================================================
elif selected_menu == "Prediksi":
    st.title("Parameter Profil Siswa")
    st.caption("Pilih sampel siswa atau atur parameter klik-saja, lalu tekan Jalankan Prediksi.")

    preset_col1, preset_col2 = st.columns([2, 1])
    with preset_col1:
        preset_idx = st.slider(
            "Profil Sampel Siswa (Baris CSV 0 – 394)",
            min_value=0,
            max_value=len(df) - 1,
            value=0,
            help="Mengisi otomatis seluruh input di bawah sesuai baris data pada student-mat.csv.",
        )
    sample_row = df.iloc[preset_idx]
    actual_g3 = int(sample_row["G3"])
    actual_status = "Lulus (G3 ≥ 10)" if actual_g3 >= 10 else "Tidak Lulus (G3 < 10)"

    with preset_col2:
        st.metric(
            label=f"Referensi Aktual Baris #{preset_idx}",
            value=f"G3 = {actual_g3}/20",
            delta=actual_status,
            delta_color="normal" if actual_g3 >= 10 else "inverse",
        )

    def radio_cat(col_name: str, label: str, horizontal: bool = True):
        options = list(CAT_OPTIONS[col_name].keys())
        default_val = sample_row[col_name]
        default_idx = options.index(default_val) if default_val in options else 0
        return st.radio(
            label,
            options=options,
            index=default_idx,
            format_func=lambda k: CAT_OPTIONS[col_name][k],
            horizontal=horizontal,
            key=f"in_{col_name}_{preset_idx}",
        )

    def radio_ord(col_name: str, label: str, horizontal: bool = False):
        options = list(ORD_OPTIONS[col_name].keys())
        default_val = int(sample_row[col_name])
        default_idx = options.index(default_val) if default_val in options else 0
        return st.radio(
            label,
            options=options,
            index=default_idx,
            format_func=lambda k: ORD_OPTIONS[col_name][k],
            horizontal=horizontal,
            key=f"in_{col_name}_{preset_idx}",
        )

    with st.container(border=True):
        tab_demo, tab_acad, tab_life = st.tabs(
            ["1. Demografi & Keluarga", "2. Akademik & Dukungan", "3. Gaya Hidup & Absensi"]
        )

        user_inputs = {}

        with tab_demo:
            d1, d2 = st.columns(2)
            with d1:
                user_inputs["school"] = radio_cat("school", "Sekolah")
                user_inputs["sex"] = radio_cat("sex", "Jenis Kelamin")
                user_inputs["age"] = st.slider(
                    "Usia Siswa (tahun)",
                    min_value=15,
                    max_value=22,
                    value=int(sample_row["age"]),
                    key=f"in_age_{preset_idx}",
                )
                user_inputs["address"] = radio_cat("address", "Area Tempat Tinggal")
                user_inputs["famsize"] = radio_cat("famsize", "Ukuran Keluarga")
                user_inputs["Pstatus"] = radio_cat("Pstatus", "Status Tinggal Orang Tua")
            with d2:
                user_inputs["Medu"] = radio_ord("Medu", "Pendidikan Ibu")
                user_inputs["Fedu"] = radio_ord("Fedu", "Pendidikan Ayah")
                user_inputs["Mjob"] = radio_cat("Mjob", "Pekerjaan Ibu", horizontal=False)
                user_inputs["Fjob"] = radio_cat("Fjob", "Pekerjaan Ayah", horizontal=False)
                user_inputs["guardian"] = radio_cat("guardian", "Wali Utama Siswa")
                user_inputs["famrel"] = radio_ord("famrel", "Kualitas Hubungan Keluarga", horizontal=True)

        with tab_acad:
            a1, a2 = st.columns(2)
            with a1:
                user_inputs["reason"] = radio_cat("reason", "Alasan Memilih Sekolah", horizontal=False)
                user_inputs["traveltime"] = radio_ord("traveltime", "Waktu Tempuh ke Sekolah")
                user_inputs["studytime"] = radio_ord("studytime", "Waktu Belajar Mingguan")
                user_inputs["failures"] = radio_ord("failures", "Riwayat Kegagalan Kelas Sebelumnya")
            with a2:
                user_inputs["schoolsup"] = radio_cat("schoolsup", "Dukungan Belajar Ekstra Sekolah")
                user_inputs["famsup"] = radio_cat("famsup", "Dukungan Belajar Keluarga")
                user_inputs["paid"] = radio_cat("paid", "Les Tambahan Berbayar (Matematika)")
                user_inputs["activities"] = radio_cat("activities", "Kegiatan Ekstrakurikuler")
                user_inputs["nursery"] = radio_cat("nursery", "Pernah Mengikuti PAUD / TK")
                user_inputs["higher"] = radio_cat("higher", "Ingin Melanjutkan ke Perguruan Tinggi")

        with tab_life:
            l1, l2 = st.columns(2)
            with l1:
                user_inputs["internet"] = radio_cat("internet", "Akses Internet di Rumah")
                user_inputs["romantic"] = radio_cat("romantic", "Memiliki Hubungan Romantis")
                user_inputs["freetime"] = radio_ord("freetime", "Waktu Luang Setelah Sekolah")
                user_inputs["goout"] = radio_ord("goout", "Frekuensi Keluar Bersama Teman")
            with l2:
                user_inputs["Dalc"] = radio_ord("Dalc", "Konsumsi Alkohol Hari Kerja")
                user_inputs["Walc"] = radio_ord("Walc", "Konsumsi Alkohol Akhir Pekan")
                user_inputs["health"] = radio_ord("health", "Status Kesehatan Saat Ini")
                user_inputs["absences"] = st.slider(
                    "Jumlah Ketidakhadiran (Absensi)",
                    min_value=0,
                    max_value=75,
                    value=int(sample_row["absences"]),
                    key=f"in_absences_{preset_idx}",
                )

        st.markdown("")
        if st.button("Jalankan Prediksi", type="primary"):
            X_single = preprocess_single_row(user_inputs, X_raw_ref, scaler, feature_cols)
            pred_class = int(model.predict(X_single)[0])
            proba = model.predict_proba(X_single)[0]
            st.session_state.last_prediction = {
                "pred_class": pred_class,
                "prob_fail": float(proba[0]),
                "prob_pass": float(proba[1]),
                "preset_idx": preset_idx,
                "actual_g3": actual_g3,
            }

    res = st.session_state.last_prediction
    if res is None:
        st.info("Belum ada prediksi. Atur parameter profil siswa lalu tekan **Jalankan Prediksi**.")
    else:
        st.markdown("### Hasil Prediksi Kelulusan")
        r1, r2 = st.columns([1.2, 1])
        with r1:
            if res["pred_class"] == 1:
                st.success(
                    f"**Status Prediksi: LULUS (`G3 ≥ 10`)**\n\n"
                    f"Probabilitas Lulus: **{res['prob_pass']*100:.1f}%** | "
                    f"Probabilitas Berisiko: **{res['prob_fail']*100:.1f}%**"
                )
            else:
                st.error(
                    f"**Status Prediksi: BERISIKO TIDAK LULUS (`G3 < 10`)**\n\n"
                    f"Probabilitas Berisiko: **{res['prob_fail']*100:.1f}%** | "
                    f"Probabilitas Lulus: **{res['prob_pass']*100:.1f}%**"
                )
            st.caption(
                f"Referensi Baris #{res['preset_idx']}: Nilai G3 aktual = **{res['actual_g3']}** "
                f"({'Lulus' if res['actual_g3'] >= 10 else 'Tidak Lulus'})."
            )
        with r2:
            prob_df = pd.DataFrame(
                {
                    "Kategori": ["Tidak Lulus (0)", "Lulus (1)"],
                    "Probabilitas": [res["prob_fail"], res["prob_pass"]],
                }
            ).set_index("Kategori")
            st.bar_chart(prob_df)


# ==============================================================================
# 9. PAGE 4: EVALUASI MODEL
# ==============================================================================
elif selected_menu == "Evaluasi Model":
    st.title("Performansi Model")
    st.caption("Dihitung pada data uji (N = 79, stratified 20% split, tanpa fitur G1 & G2).")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-card-label">Log-Loss (Metrik Utama)</div>
                <div class="info-card-value">{metrics_summary['logloss']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-card-label">Akurasi Mentah</div>
                <div class="info-card-value">{metrics_summary['accuracy']*100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-card-label">Macro F1-Score</div>
                <div class="info-card-value">{metrics_summary['f1_macro']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-card-label">ROC-AUC</div>
                <div class="info-card-value">{metrics_summary['roc_auc']:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Confusion Matrix (N = 79)")
        cm = metrics_summary["cm"]
        cm_df = pd.DataFrame(
            cm,
            index=["Aktual Tidak Lulus (0)", "Aktual Lulus (1)"],
            columns=["Prediksi Tidak Lulus (0)", "Prediksi Lulus (1)"],
        )
        st.dataframe(cm_df, use_container_width=True)
        st.caption(
            f"TN = {cm[0,0]} | FP = {cm[0,1]} | FN = {cm[1,0]} | TP = {cm[1,1]}. "
            "Penalti L1 (`alpha=0.01`) & `class_weight='balanced'` menjaga deteksi kelas minoritas."
        )

    with right_col:
        st.subheader("Kurva ROC")
        fpr, tpr, _ = roc_curve(y_test, y_prob_test)
        roc_df = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr}).set_index(
            "False Positive Rate"
        )
        st.line_chart(roc_df)

    if os.path.exists(FIG_2_PATH):
        st.markdown("")
        st.image(
            FIG_2_PATH,
            caption="Visualisasi Confusion Matrix model peringatan dini pada data uji (N = 79).",
            use_container_width=False,
            width=540,
        )

    st.markdown("")
    with st.expander("Classification report lengkap"):
        report_dict = classification_report(
            y_test,
            y_pred_test,
            target_names=["Tidak Lulus (0)", "Lulus (1)"],
            output_dict=True,
        )
        report_df = pd.DataFrame(report_dict).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    with st.expander("Bobot fitur aktif (L1 Sparse Coefficients)"):
        coefs = model.coef_[0]
        coef_df = pd.DataFrame({"Fitur": feature_cols, "Koefisien": coefs})
        active_df = (
            coef_df[coef_df["Koefisien"].abs() > 1e-6]
            .assign(Abs=lambda d: d["Koefisien"].abs())
            .sort_values("Abs", ascending=False)
            .drop(columns=["Abs"])
        )
        st.caption(
            f"Dari **{len(feature_cols)}** fitur hasil encoding, L1 regularization mempertahankan "
            f"**{len(active_df)}** fitur aktif dan mengenolkan **{len(feature_cols) - len(active_df)}** fitur noise."
        )
        st.dataframe(active_df.round(4), use_container_width=True, hide_index=True)

st.markdown(
    '<div class="app-footer">© 2026 GradWatch (Streamlit + Scikit-Learn).</div>',
    unsafe_allow_html=True,
)