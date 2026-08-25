"""
Aplikasi Streamlit — Klasifikasi Jenis Cyberbullying
Alur: Disclaimer -> (Classifier | EDA | Perbandingan Model), navigasi via tombol (bukan multipage)
"""

import json
import os
import re
import string

import joblib
import pandas as pd
import streamlit as st
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect, LangDetectException

st.set_page_config(page_title="Deteksi Cyberbullying", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
        .stApp { background-color: #FFFFFF; }
        .main-title { font-size: 2.1rem; font-weight: 800; color: #1A1A2E; margin-bottom: 0.1rem; }
        .sub-title { color: #5A5A72; font-size: 1rem; margin-bottom: 1.2rem; }
        .result-card { padding: 1.1rem 1.4rem; border-radius: 14px; margin-top: 0.8rem; border: 1px solid #E4E8EF; }
        .result-safe   { background-color: #E9F9EF; border-color: #A6E9BE; }
        .result-danger { background-color: #FDECEC; border-color: #F5B4B4; }
        .stTextArea textarea { border-radius: 10px; }
        div[data-testid="stMetricValue"] { color: #2E86DE; }
        .disclaimer-box {
            background-color: #FFF4E5; border: 1px solid #F5C989; border-radius: 12px;
            padding: 1.3rem 1.6rem; margin: 1rem 0 1.5rem 0;
        }
        .wc-gate-box {
            background-color: #FFF9E6; border: 1px solid #F0D875; border-radius: 12px;
            padding: 1rem 1.2rem; text-align: center;
        }
        .low-confidence-badge {
            background-color: #FFF4E5; border: 1px solid #F5C989; border-radius: 8px;
            padding: 0.5rem 0.8rem; margin-top: 0.5rem; font-size: 0.9rem; color: #8A5A00;
        }
        .stat-box {
            background-color: #F0F4F8; border-radius: 12px; padding: 1rem 1.2rem;
            text-align: center; border: 1px solid #E4E8EF;
        }
        .stat-box .stat-number { font-size: 1.6rem; font-weight: 800; color: #2E86DE; }
        .stat-box .stat-label { font-size: 0.85rem; color: #5A5A72; }
        .nav-active button { background-color: #2E86DE !important; color: white !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in {
    "show_classifier": False,
    "current_view": "classifier",   # "classifier" | "eda" | "models"
    "input_text": "",
    "pred_label": None,
    "pred_confidence": None,
    "pred_cleaned": "",
    "wc_revealed": False,
    "detected_lang": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

CONFIDENCE_THRESHOLD = 60.0

LABEL_INFO = {
    "not_cyberbullying": ("✅ Bukan Cyberbullying", "safe"),
    "age": ("⚠️ Cyberbullying — Usia", "danger"),
    "ethnicity": ("⚠️ Cyberbullying — Etnis", "danger"),
    "gender": ("⚠️ Cyberbullying — Gender", "danger"),
    "religion": ("⚠️ Cyberbullying — Agama", "danger"),
    "other_cyberbullying": ("⚠️ Cyberbullying — Lainnya", "danger"),
}

EXAMPLES = [
    "You are so stupid because of your religion",
    "Happy birthday my friend, hope you have a great day!",
    "I hate people of that ethnicity, they should leave",
    "Great match yesterday, the team played really well",
]

# ===========================================================================
# HALAMAN 0 — DISCLAIMER (gerbang awal, sebelum apapun bisa diakses)
# ===========================================================================
if not st.session_state.show_classifier:
    st.markdown('<div class="main-title">🛡️ Deteksi Jenis Cyberbullying</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Final Project — Klasifikasi Cyberbullying pada Tweet Berbahasa Inggris</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="disclaimer-box">
        <h4>⚠️ Peringatan Konten</h4>
        <p>Aplikasi ini dilatih menggunakan dataset publik <b>Cyberbullying Classification</b> (Kaggle)
        yang berisi tweet asli, termasuk bahasa kasar, slur, dan ujaran kebencian yang eksplisit.</p>
        <p>Fitur <b>WordCloud</b> menampilkan kata-kata apa adanya dari data asli untuk tujuan transparansi
        analisis — bukan menormalisasi bahasa tersebut. WordCloud disembunyikan secara default dan baru
        muncul setelah kamu memilih untuk menampilkannya.</p>
        <p>Dengan melanjutkan, kamu memahami bahwa kamu mungkin akan melihat kata-kata yang
        ofensif atau tidak nyaman.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("✅ Mulai Aplikasi", type="primary", use_container_width=True):
        st.session_state.show_classifier = True
        st.rerun()
    st.stop()  # hentikan render di sini, jangan lanjut ke bawah sebelum disclaimer di-acknowledge

# ===========================================================================
# Setup NLTK, clean_text, deteksi bahasa, load model (dipakai lintas halaman)
# ===========================================================================
@st.cache_resource
def setup_nltk():
    resources = ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "stopwords", "wordnet", "omw-1.4"]
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass
    return set(stopwords.words("english")), WordNetLemmatizer()


stop_words, lemmatizer = setup_nltk()


def get_wordnet_pos(tag):
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("N"): return wordnet.NOUN
    if tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN


def lemmatize_with_pos(tokens):
    tagged = pos_tag(tokens)
    return [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = lemmatize_with_pos(tokens)
    return " ".join(tokens)


def detect_language_safe(text):
    try:
        return detect(text)
    except LangDetectException:
        return None


@st.cache_resource
def load_artifacts():
    model = joblib.load("model_terbaik.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, tfidf, le


model, tfidf, le = load_artifacts()

with st.sidebar:
    st.header("ℹ️ Tentang Model")
    st.write("**Algoritma:** XGBoost")
    st.write("**Fitur:** TF-IDF")
    st.write(f"**Jumlah kategori:** {len(le.classes_)}")
    st.write("**Kategori:**")
    for c in le.classes_:
        st.markdown(f"- {c}")
    st.divider()
    st.caption("Model dilatih pada dataset publik *Cyberbullying Classification* (Kaggle) — teks Bahasa Inggris.")

# ---------------------------------------------------------------------------
# Navigasi (tombol, bukan sidebar multipage)
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🛡️ Deteksi Jenis Cyberbullying</div>', unsafe_allow_html=True)

nav_cols = st.columns(3)
nav_items = [("classifier", "🔍 Classifier"), ("eda", "📊 EDA Dataset"), ("models", "🧪 Perbandingan Model")]
for (view_key, view_label), ncol in zip(nav_items, nav_cols):
    with ncol:
        is_active = st.session_state.current_view == view_key
        if st.button(view_label, use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.current_view = view_key
            st.rerun()

st.divider()

# ===========================================================================
# HALAMAN A — CLASSIFIER
# ===========================================================================
if st.session_state.current_view == "classifier":
    st.markdown(
        '<div class="sub-title">Masukkan teks tweet berbahasa Inggris untuk mendeteksi apakah termasuk '
        'cyberbullying, dan jenis kategorinya.</div>',
        unsafe_allow_html=True,
    )

    st.write("**Coba contoh cepat:**")
    ex_cols = st.columns(len(EXAMPLES))
    for i, ex in enumerate(EXAMPLES):
        with ex_cols[i]:
            label = ex if len(ex) <= 28 else ex[:26] + "…"
            if st.button(label, key=f"ex_{i}", use_container_width=True):
                st.session_state.input_text = ex

    col_input, col_wordcloud = st.columns([1, 1], gap="large")

    with col_input:
        user_input = st.text_area(
            "Teks tweet:",
            value=st.session_state.input_text,
            height=150,
            placeholder="Ketik atau tempel teks tweet di sini...",
            key="text_area_input",
        )
        predict_btn = st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True)
        show_clean = st.toggle("Tampilkan hasil cleaning teks", value=False)

        if st.session_state.detected_lang and st.session_state.detected_lang != "en":
            st.warning(
                f"⚠️ Input terakhir terdeteksi sebagai bahasa `{st.session_state.detected_lang}`, "
                f"bukan Bahasa Inggris. Akurasi prediksi tidak terjamin untuk bahasa selain Inggris."
            )

        if st.session_state.pred_label:
            display_text, style = LABEL_INFO.get(st.session_state.pred_label, (st.session_state.pred_label, "danger"))
            st.markdown(
                f'<div class="result-card result-{style}"><h3 style="margin:0;">{display_text}</h3></div>',
                unsafe_allow_html=True,
            )
            if st.session_state.pred_confidence is not None:
                st.metric("Tingkat keyakinan", f"{st.session_state.pred_confidence:.1f}%")
                if st.session_state.pred_confidence < CONFIDENCE_THRESHOLD:
                    st.markdown(
                        '<div class="low-confidence-badge">⚠️ Keyakinan model rendah — '
                        'hasil ini sebaiknya ditinjau manual, jangan dijadikan keputusan otomatis.</div>',
                        unsafe_allow_html=True,
                    )
            if show_clean:
                st.text_area("Teks setelah cleaning:", value=st.session_state.pred_cleaned, height=80, disabled=True)
        else:
            st.info("Hasil prediksi akan muncul di sini setelah kamu klik **Prediksi Sekarang**.")

    with col_wordcloud:
        st.markdown("**🔤 Gambaran kata kunci:**")
        if st.session_state.pred_label:
            wc_path = f"wordcloud_{st.session_state.pred_label}.png"
            caption = f"Kategori: {st.session_state.pred_label}"
        else:
            wc_path = "wordcloud_all.png"
            caption = "Seluruh kategori (sebelum ada prediksi)"

        if not st.session_state.wc_revealed:
            st.markdown(
                """
                <div class="wc-gate-box">
                ⚠️ <b>WordCloud dapat berisi konten sensitif</b><br>
                <span style="font-size:0.85rem; color:#7A6A00;">
                Kata-kata di bawah diambil apa adanya dari data tweet asli.
                </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            if st.button("👁️ Tampilkan WordCloud", use_container_width=True):
                st.session_state.wc_revealed = True
                st.rerun()
        else:
            if os.path.exists(wc_path):
                st.image(wc_path, caption=caption, use_container_width=True)
                if st.button("🙈 Sembunyikan WordCloud", use_container_width=True):
                    st.session_state.wc_revealed = False
                    st.rerun()
            else:
                st.caption(f"File `{wc_path}` tidak ditemukan di folder deployment.")

    if predict_btn:
        if not user_input.strip():
            st.warning("Teks tweet masih kosong, silakan isi dulu.")
        else:
            detected_lang = detect_language_safe(user_input)

            cleaned = clean_text(user_input)
            if not cleaned.strip():
                st.error(
                    "Setelah cleaning, teks menjadi kosong (kemungkinan hanya berisi "
                    "URL/mention/angka/stopword). Coba teks yang lebih deskriptif."
                )
            else:
                vect = tfidf.transform([cleaned])
                pred_num = model.predict(vect)[0]
                pred_label = le.inverse_transform([pred_num])[0]

                st.session_state.pred_label = pred_label
                st.session_state.pred_cleaned = cleaned
                st.session_state.wc_revealed = False
                st.session_state.detected_lang = detected_lang

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(vect)[0]
                    st.session_state.pred_confidence = proba[pred_num] * 100
                else:
                    st.session_state.pred_confidence = None

                st.rerun()

    st.divider()
    st.caption("⚠️ Prediksi bersifat otomatis dan bisa saja keliru — gunakan sebagai alat bantu, bukan keputusan final.")

# ===========================================================================
# HALAMAN B — EDA DATASET
# ===========================================================================
elif st.session_state.current_view == "eda":
    st.markdown(
        '<div class="sub-title">Eksplorasi dataset yang digunakan untuk melatih model.</div>',
        unsafe_allow_html=True,
    )

    if os.path.exists("eda_stats.json"):
        with open("eda_stats.json") as f:
            stats = json.load(f)

        s1, s2, s3, s4 = st.columns(4)
        stat_items = [
            (s1, f"{stats['total_setelah_dedup']:,}", "Total Data (final)"),
            (s2, f"{stats['duplikat_dihapus']:,}", "Duplikat Dihapus"),
            (s3, f"{stats['jumlah_kategori']}", "Jumlah Kategori"),
            (s4, f"{stats['avg_tweet_length']:.1f}", "Rata-rata Kata/Tweet"),
        ]
        for col, number, label in stat_items:
            with col:
                st.markdown(
                    f'<div class="stat-box"><div class="stat-number">{number}</div>'
                    f'<div class="stat-label">{label}</div></div>',
                    unsafe_allow_html=True,
                )

        st.write("")
    else:
        st.warning("File `eda_stats.json` tidak ditemukan — jalankan skrip export EDA di Colab terlebih dahulu.")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("**Distribusi Jumlah Tweet per Kategori**")
        if os.path.exists("distribusi_kategori.png"):
            st.image("distribusi_kategori.png", use_container_width=True)
        else:
            st.caption("File `distribusi_kategori.png` tidak ditemukan.")

    with col_b:
        st.markdown("**Distribusi Panjang Tweet**")
        if os.path.exists("distribusi_panjang_tweet.png"):
            st.image("distribusi_panjang_tweet.png", use_container_width=True)
        else:
            st.caption("File `distribusi_panjang_tweet.png` tidak ditemukan.")

    st.divider()
    st.markdown("**WordCloud per Kategori**")
    st.caption("⚠️ Gambar di bawah berisi kata-kata apa adanya dari data asli, termasuk kemungkinan bahasa kasar.")

    if "eda_wc_revealed" not in st.session_state:
        st.session_state.eda_wc_revealed = False

    if not st.session_state.eda_wc_revealed:
        if st.button("👁️ Tampilkan Semua WordCloud Kategori"):
            st.session_state.eda_wc_revealed = True
            st.rerun()
    else:
        if st.button("🙈 Sembunyikan"):
            st.session_state.eda_wc_revealed = False
            st.rerun()

        wc_cols = st.columns(3)
        for i, cat in enumerate(le.classes_):
            wc_path = f"wordcloud_{cat}.png"
            with wc_cols[i % 3]:
                if os.path.exists(wc_path):
                    st.image(wc_path, caption=cat, use_container_width=True)
                else:
                    st.caption(f"`{wc_path}` tidak ditemukan.")

# ===========================================================================
# HALAMAN C — PERBANDINGAN MODEL
# ===========================================================================
elif st.session_state.current_view == "models":
    st.markdown(
        '<div class="sub-title">Perbandingan algoritma yang diuji sebelum XGBoost dipilih sebagai model final.</div>',
        unsafe_allow_html=True,
    )

    if os.path.exists("model_comparison.csv"):
        comp_df = pd.read_csv("model_comparison.csv", index_col=0)
        comp_df_display = comp_df.sort_values("f1_macro", ascending=False).round(4)
        st.dataframe(comp_df_display, use_container_width=True)

        if os.path.exists("model_comparison_notes.json"):
            with open("model_comparison_notes.json") as f:
                notes = json.load(f)
            st.success(f"**Model terpilih: {notes['best_model']}**\n\n{notes['alasan_pemilihan']}")
    else:
        st.warning("File `model_comparison.csv` tidak ditemukan — jalankan skrip export di Colab terlebih dahulu.")

    if os.path.exists("perbandingan_model.png"):
        st.markdown("**Visualisasi Perbandingan**")
        st.image("perbandingan_model.png", use_container_width=True)

    if os.path.exists("flat_vs_hierarchical.csv"):
        st.divider()
        st.markdown("**Flat Classification vs Hierarchical Classification**")
        hier_df = pd.read_csv("flat_vs_hierarchical.csv", index_col=0)
        st.dataframe(hier_df.round(4), use_container_width=True)
        st.caption(
            "Hierarchical classification diuji sebagai pendekatan alternatif (Level 1: bullying/tidak, "
            "Level 2: jenis bullying), namun flat classification tetap dipilih untuk deployment karena "
            "performa akhirnya sedikit lebih tinggi dan pipeline-nya lebih sederhana."
        )
