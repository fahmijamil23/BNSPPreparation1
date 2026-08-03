"""
Aplikasi Streamlit — Klasifikasi Jenis Cyberbullying
Single-page flow: Disclaimer -> Classifier, dengan double-safety reveal untuk WordCloud
"""

import os
import re
import string

import joblib
import streamlit as st
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer

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
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in {
    "show_classifier": False,
    "input_text": "",
    "pred_label": None,
    "pred_confidence": None,
    "pred_cleaned": "",
    "wc_revealed": False,   # status reveal wordcloud (reset tiap ada prediksi baru)
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

CONFIDENCE_THRESHOLD = 60.0  # di bawah ini dianggap "keyakinan rendah"

# ===========================================================================
# HALAMAN 1 — DISCLAIMER
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
        <p>Fitur <b>WordCloud</b> pada aplikasi ini menampilkan kata-kata yang paling sering muncul
        di setiap kategori — apa adanya dari data asli, tanpa disensor — karena tujuannya adalah
        transparansi analisis, bukan menormalisasi bahasa tersebut. WordCloud akan
        <b>disembunyikan secara default</b> dan baru muncul setelah kamu memilih untuk menampilkannya.</p>
        <p>Dengan melanjutkan, kamu memahami bahwa kamu mungkin akan melihat kata-kata yang
        ofensif atau tidak nyaman.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("✅ Mulai Classifier", type="primary", use_container_width=True):
        st.session_state.show_classifier = True
        st.rerun()

# ===========================================================================
# HALAMAN 2 — CLASSIFIER
# ===========================================================================
else:

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

    if st.button("← Kembali", type="secondary"):
        st.session_state.show_classifier = False
        st.rerun()

    st.markdown('<div class="main-title">🛡️ Deteksi Jenis Cyberbullying</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Masukkan teks tweet berbahasa Inggris untuk mendeteksi apakah termasuk '
        'cyberbullying, dan jenis kategorinya.</div>',
        unsafe_allow_html=True,
    )

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

        if st.session_state.pred_label:
            display_text, style = LABEL_INFO.get(st.session_state.pred_label, (st.session_state.pred_label, "danger"))
            st.markdown(
                f'<div class="result-card result-{style}"><h3 style="margin:0;">{display_text}</h3></div>',
                unsafe_allow_html=True,
            )
            if st.session_state.pred_confidence is not None:
                st.metric("Tingkat keyakinan", f"{st.session_state.pred_confidence:.1f}%")
                # --- Poin 2: badge keyakinan rendah ---
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

        # --- Double safety: wordcloud disembunyikan di balik tombol reveal ---
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
                st.session_state.wc_revealed = False  # reset -> wordcloud kategori baru wajib di-reveal ulang

                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(vect)[0]
                    st.session_state.pred_confidence = proba[pred_num] * 100
                else:
                    st.session_state.pred_confidence = None

                st.rerun()

    st.divider()
    st.caption("⚠️ Prediksi bersifat otomatis dan bisa saja keliru — gunakan sebagai alat bantu, bukan keputusan final.")
