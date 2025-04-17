import streamlit as st
import zipfile
import os
import shutil
import time
import cv2
import pandas as pd
from process_damar_analizi import process_images  # process_damar_analizi.py dosyasından fonksiyonlar

# Başlık
st.title("Damar Görüntüleme ve Kümeleme")

# Kullanıcıdan zip dosyası alma
uploaded_file = st.file_uploader("Zip dosyasını yükleyin", type=["zip"])

# Zip dosyasını açma ve içerikleri işleme
if uploaded_file is not None:
    st.write("Zip dosyası alındı. İşlem başlatılıyor...")

    # Yüklenen dosyayı geçici bir dosyaya kaydedelim
    temp_zip_path = "/tmp/uploaded_file.zip"  # Geçici dosya yolu
    with open(temp_zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Zip dosyasını çıkar
    unzip_folder = "input_images"
    os.makedirs(unzip_folder, exist_ok=True)

    try:
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_folder)
        st.write("Görseller başarıyla çıkarıldı!")
    except Exception as e:
        st.error(f"Zip dosyası çıkarılırken hata oluştu: {str(e)}")

    # Görseller çıkarıldıktan sonra analiz butonunu ekleyelim
    if st.button("Analiz Et"):
        # Progress bar'ı başlat
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        st.write("Görseller işleniyor...")
        
        # Görselleri işleme fonksiyonunu burada çağırıyoruz
        analyze_images(unzip_folder, "final_vessel_analysis", progress_bar, status_text)

        st.write("Analiz tamamlandı!")

# Görselleri analiz etme fonksiyonu
def analyze_images(input_folder, output_folder, progress_bar, status_text):
    # Çıktı klasörünü oluştur
    os.makedirs(output_folder, exist_ok=True)
    results = []
    total_images = len([f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png'))])
    
    # Görselleri sırayla işle
    for i, filename in enumerate(os.listdir(input_folder)):
        if filename.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            
            # Görseli işleme ve analiz fonksiyonu
            if img is not None:
                try:
                    # process_images fonksiyonu, damar tespiti, analiz, görselleştirme ve CSV kaydetme işlemleri içeriyor
                    result = process_images(img, filename, output_folder)

                    # Sonuçları kaydetme (isteğe bağlı olarak görselleştirme)
                    results.append(result)

                except Exception as e:
                    st.error(f"{filename} işlenirken hata oluştu: {str(e)}")

            # Progress bar'ı güncelle
            progress = (i + 1) / total_images
            progress_bar.progress(progress)
            status_text.text(f"İşlenen Görsel: {i + 1}/{total_images}")
            time.sleep(0.1)  # Simülasyon için küçük bir uyku süresi

    # Sonuçları bir CSV'ye kaydet
    if results:
        df = pd.DataFrame(results)
        output_csv = os.path.join(output_folder, 'results.csv')
        df.to_csv(output_csv, index=False)
        st.write(f"Sonuçlar {output_csv} dosyasına kaydedildi.")
