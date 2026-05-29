import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import os
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="AI Voice Detector Pro", page_icon="🎤")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('/content/drive/MyDrive/audio_ai_project/final_model_v2.h5')

model = load_my_model()

def extract_features(audio_path):
    audio, sr = librosa.load(audio_path, sr=22050)
    target_len = sr * 5
    # Sessizlik temizleme ve padding
    audio, _ = librosa.effects.trim(audio)
    if len(audio) > target_len:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)))
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
    mfcc = mfcc[np.newaxis, ..., np.newaxis] 
    return mfcc

st.title("🎤 AI Voice Detector Pro")

if 'audio_path' not in st.session_state:
    st.session_state.audio_path = None

# Arayüz
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Ses Yükle", type=['wav', 'mp3', 'm4a'])
    if uploaded_file:
        with open("temp.wav", "wb") as f: f.write(uploaded_file.getbuffer())
        st.session_state.audio_path = "temp.wav"
with col2:
    audio_bytes = audio_recorder(text="Kaydet", pause_threshold=2.0)
    if audio_bytes:
        with open("temp.wav", "wb") as f: f.write(audio_bytes)
        st.session_state.audio_path = "temp.wav"

if st.session_state.audio_path:
    st.audio(st.session_state.audio_path)
    
    if st.button("🚀 Modeli Test Et"):
        features = extract_features(st.session_state.audio_path)
        prediction = float(model.predict(features)[0][0])
        prediction = max(0.0, min(1.0, prediction))
        
        st.divider()
        
        # --- ŞÜPHELİ KATEGORİSİ EKLENMİŞ MANTIK ---
        if prediction > 0.65:
            st.error(f"### 🚨 SONUÇ: YAPAY ZEKA (AI)")
            st.progress(prediction)
            st.write(f"**Yapaylık Analizi:** Model yüksek olasılıkla sentetik tını algıladı (%{prediction*100:.2f})")
            
        elif 0.40 <= prediction <= 0.65:
            st.warning(f"### ⚠️ ANALİZ EDİLİYOR: İNSAN OLMA İHTİMALİ YÜKSEK")
            st.progress(prediction)
            st.write("**Model Notu:** Ses karakteristiği insan tınısına çok yakın ancak ortam gürültüsü veya kayıt kalitesi nedeniyle düşük seviyeli yapaylık sinyalleri tespit edildi.")
            st.info("Bu durum genellikle mikrofon yankısı veya oda akustiğinden kaynaklanır.")
            
        else:
            st.success(f"### ✅ SONUÇ: GERÇEK İNSAN")
            st.progress(1.0 - prediction)
            st.write(f"**Doğallık Analizi:** Model insan sesine özgü frekans düzensizliklerini başarıyla doğruladı (%{(1-prediction)*100:.2f})")
        
        st.divider()

if st.button("🔄 Sıfırla"):
    st.session_state.audio_path = None
    st.rerun()
