import streamlit as st
import joblib

# =========================
# LOAD MODEL & VECTORIZER
# =========================
model = joblib.load('svm_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# =========================
# TAMPILAN APLIKASI
# =========================
st.set_page_config(page_title="MBTI Predictor", page_icon="🧠")

st.title("🧠 MBTI Personality Predictor")
st.write("Aplikasi ini memprediksi tipe kepribadian MBTI berdasarkan teks bahasa Inggris menggunakan model **SVM + TF-IDF**.")

# =========================
# INPUT TEXT
# =========================
user_input = st.text_area(
    "Masukkan teks dalam Bahasa Inggris:",
    placeholder="Example: I enjoy being alone and thinking deeply about life..."
)

# =========================
# TOMBOL PREDIKSI
# =========================
if st.button("Prediksi MBTI"):
    if user_input.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong!")
    else:
        # Preprocessing ke TF-IDF
        input_tfidf = tfidf.transform([user_input])

        # Prediksi
        prediction = model.predict(input_tfidf)

        # Tampilkan hasil
        st.success(f"✅ Prediksi Tipe MBTI: **{prediction[0]}**")
