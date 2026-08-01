"""
Aplikasi Streamlit — Klasifikasi Jenis Cyberbullying (Tema Terang, Interaktif)
Model: TF-IDF + XGBoost (model terbaik hasil perbandingan 7-8 algoritma)
"""

import re
import string

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Konfigurasi halaman & tema terang
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Deteksi Cyberbullying",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp { background-color: #FFFFFF; }
        .main-title {
            font-size: 2.1rem; font-weight: 800; color: #1A1A2E;
            margin-bottom: 0.1rem;
        }
        .sub-title { color: #5A5A72; font-size: 1rem; margin-bottom: 1.2rem; }
        .result-card {
            padding: 1.1rem 1.4rem; border-radius: 14px; margin-top: 0.8rem;
            border: 1px solid #E4E8EF;
        }
        .result-safe   { background-color: #E9F9EF; border-color: #A6E9BE; }
        .result-danger { background-color: #FDECEC; border-color: #F5B4B4; }
        .example-btn button { width: 100%; }
        .stTextArea textarea { border-radius: 10px; }
        div[data-testid="stMetricValue"] { color: #2E86DE; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Setup NLTK (cache biar tidak download ulang tiap run)
# ---------------------------------------------------------------------------
@st.cache_resource
def setup_nltk():
    resources = [
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "stopwords",
        "wordnet",
        "omw-1.4",
    ]
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass
    return set(stopwords.words("english")), WordNetLemmatizer()


stop_words, lemmatizer = setup_nltk()


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_with_pos(tokens):
    tagged = pos_tag(tokens)
    return [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged]


def clean_text(text):
    """Persis sama dengan fungsi cleaning yang dipakai saat training."""
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


# ---------------------------------------------------------------------------
# Load model, vectorizer, label encoder
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_terbaik.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, tfidf, le


model, tfidf, le = load_artifacts()

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

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ Tentang Model")
    st.write("**Algoritma:** XGBoost")
    st.write("**Fitur:** TF-IDF")
    st.write(f"**Jumlah kategori:** {len(le.classes_)}")
    st.write("**Kategori:**")
    for c in le.classes_:
        st.markdown(f"- {c}")
    st.divider()
    st.caption(
        "Model dilatih pada dataset publik *Cyberbullying Classification* "
        "(Kaggle) — teks Bahasa Inggris."
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🛡️ Deteksi Jenis Cyberbullying</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Masukkan teks tweet berbahasa Inggris untuk mendeteksi apakah termasuk '
    'cyberbullying, dan jenis kategorinya.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Contoh cepat (tombol)
# ---------------------------------------------------------------------------
st.write("**Coba contoh cepat:**")
ex_cols = st.columns(len(EXAMPLES))
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

for i, ex in enumerate(EXAMPLES):
    with ex_cols[i]:
        label = ex if len(ex) <= 28 else ex[:26] + "…"
        if st.button(label, key=f"ex_{i}", use_container_width=True):
            st.session_state.input_text = ex

# ---------------------------------------------------------------------------
# Input & prediksi
# ---------------------------------------------------------------------------
col_input, col_result = st.columns([1, 1], gap="large")

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

with col_result:
    if predict_btn:
        if not user_input.strip():
            st.warning("Teks tweet masih kosong, silakan isi dulu.")
        else:
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
                display_text, style = LABEL_INFO.get(pred_label, (pred_label, "danger"))

                st.markdown(
                    f'<div class="result-card result-{style}">'
                    f'<h3 style="margin:0;">{display_text}</h3></div>',
                    unsafe_allow_html=True,
                )

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(vect)[0]
                    conf = proba[pred_num] * 100
                    st.metric("Tingkat keyakinan", f"{conf:.1f}%")

                    # Grafik interaktif: probabilitas semua kategori
                    proba_df = pd.DataFrame({
                        "Kategori": le.classes_,
                        "Probabilitas (%)": proba * 100,
                    }).sort_values("Probabilitas (%)", ascending=True)

                    fig = px.bar(
                        proba_df, x="Probabilitas (%)", y="Kategori", orientation="h",
                        color="Probabilitas (%)", color_continuous_scale="Blues",
                        text="Probabilitas (%)",
                    )
                    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig.update_layout(
                        plot_bgcolor="white", paper_bgcolor="white",
                        height=280, margin=dict(l=0, r=10, t=10, b=0),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if show_clean:
                    st.text_area("Teks setelah cleaning:", value=cleaned, height=80, disabled=True)
    else:
        st.info("Hasil prediksi akan muncul di sini setelah kamu klik **Prediksi Sekarang**.")

st.divider()
st.caption(
    "⚠️ Prediksi bersifat otomatis dan bisa saja keliru — gunakan sebagai alat bantu, "
    "bukan keputusan final."
)
