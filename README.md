# 🎤 AI Voice Detector Pro (Deep Learning Project)

Bu proje, ses kayıtlarını analiz ederek sesin **gerçek bir insana** mı ait olduğunu yoksa **yapay zeka (AI/Deepfake)** tarafından mı üretildiğini tespit etmek amacıyla geliştirilmiştir.

## 🚀 Projenin Amacı
Gelişen yapay zeka teknolojileri ile birlikte ses taklidi ve sentetik ses üretimi oldukça gerçekçi bir noktaya ulaşmıştır. Bu çalışma, dijital güvenlik ve içerik doğrulama süreçlerine katkı sağlamak için derin öğrenme tekniklerini kullanarak bir savunma mekanizması sunar.

## 📊 Veri Seti (Datasets)
Modelin eğitimi ve testi için iki ana kaynak birleştirilmiştir:
1. **Ana Veri Seti:** [Kaggle Deepfake Audio Dataset](1000 Gerçek, 1000 Yapay ses örneği).
2. **Özel Veri Seti (Custom):** Tarafımca kaydedilen gerçek insan sesleri ile güncel asistanlardan (Siri, Gemini, vb.) alınan örneklerden oluşan hibrit bir test seti.

## 🧠 Teknik Mimari
Proje, ses sinyallerini görsel birer veri gibi işleyen **CNN (Convolutional Neural Networks)** mimarisi üzerine kurulmuştur.

- **Özellik Çıkarımı:** Ses dosyaları `librosa` kütüphanesi ile **MFCC (Mel-Frequency Cepstral Coefficients)** özniteliklerine dönüştürülmüştür.
- **Model Yapısı:** - 3 adet Convolutional Katmanı (Özellik yakalama)
  - Batch Normalization & Dropout (Aşırı öğrenmeyi engelleme)
  - Dense & Sigmoid Çıkış Katmanı (Sınıflandırma)
- **Başarı Oranı:** %98 Accuracy.



## 🛠️ Uygulama Özellikleri
Uygulama **Streamlit** üzerinden kullanıcı dostu bir arayüz sunar:
- **Dosya Yükleme:** `.wav`, `.mp3` veya `.m4a` formatlarını destekler.
- **Canlı Kayıt:** Mikrofon üzerinden anlık ses analizi yapabilir.
- **Üçlü Analiz Sistemi:** - ✅ **Gerçek İnsan:** Düşük yapaylık skoru.
  - ⚠️ **Analiz Ediliyor/Şüpheli:** Modelin %40-%65 arası kararsız kaldığı, ancak insan tınısının yüksek olduğu durumlar.
  - 🚨 **Yapay Zeka (AI):** Yüksek sentetik sinyal tespiti.

## ⚙️ Kurulum ve Çalıştırma
Projeyi yerelde çalıştırmak için:
```bash
pip install -r requirements.txt
streamlit run app.py
