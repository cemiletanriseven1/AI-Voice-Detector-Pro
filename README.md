# 🎤 AI Voice Detector Pro

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Bu proje, bir ses kaydının gerçek bir insana mı ait olduğunu yoksa yapay zeka (AI/Deepfake) tarafından mı üretildiğini frekans bazında analiz ederek tespit eden **derin öğrenme tabanlı bir güvenlik mekanizmasıdır**. Proje kapsamında ses sinyallerinden öznitelik çıkarımı yapılmış, bir Evrişimsel Sinir Ağı (CNN) modeli eğitilmiş ve son kullanıcı için interaktif bir web arayüzü entegre edilmiştir.

---

## 🚀 Proje Özellikleri
* **Gelişmiş Öznitelik Çıkarımı:** Ses sinyallerinden 40 kanallı MFCC (Mel Frequency Cepstral Coefficients) dönüşümü ve Z-Score normalizasyonu.
* **Güçlü CNN Mimarisi:** Yaklaşık 2.3 milyon parametreye sahip, aşırı öğrenmeyi (overfitting) engelleyen Dropout ve BatchNormalization katmanlı derin öğrenme modeli.
* **Canlı Web Arayüzü:** Streamlit tabanlı, sürükle-bırak ses yükleme desteğine ve tarayıcı tabanlı canlı mikrofon kaydı özelliğine sahip modern arayüz.
* **Üçlü Karar Mekanizması:** Klasik ikili eşikler yerine olasılık tabanlı belirsizlik tahmini yönetimi.

---

## 📊 Veri Seti Dağılımı (Dataset Summary)

Modelin tarafsız eğitilmesi için Kaggle üzerindeki devasa havuzdan dengeli bir alt küme (1000 Real / 1000 Fake) seçilmiş, ayrıca modern TTS motorlarına karşı direnci ölçmek adına özel kayıtlar eklenmiştir.

| Veri Türü | Kaynak / Açıklama | Adet |
| :--- | :--- | :--- |
| **Real** | Kaggle Deepfake Audio Dataset (Dengeli Set) | 1000 |
| **Fake** | Kaggle Deepfake Audio Dataset (Dengeli Set) | 1000 |
| **Real (Özel)** | Yakın Çevre ve Doğal Ortam Ses Kayıtları | 68 |
| **Fake (Özel)** | Microsoft Edge TTS, Siri ve Canlı Asistanlar | 190 |
| **TOPLAM** | **Genel Veri Hacmi** | **2258** |

---

## 📉 Model Performansı & Başarı Metrikleri

Model, daha önce hiç görmediği 451 adet test verisi üzerinde **%98 Genel Doğruluk (Accuracy)** oranına ulaşmıştır. 

### Sınıflandırma Raporu (Classification Report)
* **Sınıf 0 (Real - Gerçek İnsan):** Precision: 0.96 | **Recall: 1.00** | F1-Score: 0.98
* **Sınıf 1 (Fake - Yapay Zeka):** **Precision: 1.00** | Recall: 0.96 | F1-Score: 0.98

> **Kritik Analiz:** Gerçek insan seslerindeki Recall değerinin **1.00** olması, modelin insan seslerini ayırt etmede kusursuzlaştığını; yapay zeka seslerindeki Precision değerinin **1.00** olması ise "Yapay Zeka" olarak etiketlenen seslerde sıfır hatalı pozitif (False Positive) payı olduğunu gösterir.

---

## 🛠️ Kurulum ve Yerel Çalıştırma (Installation)

Projeyi kendi bilgisayarınızda veya Google Colab üzerinde çalıştırmak için aşağıdaki adımları takip edebilirsiniz.

### 1. Depoyu Klonlayın
```bash
git clone [https://github.com/cemiletanriseven1/AI-Voice-Detector-Pro.git](https://github.com/cemiletanriseven1/AI-Voice-Detector-Pro.git)
cd AI-Voice-Detector-Pro
