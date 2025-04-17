import os
import time
import zipfile
import cv2
import streamlit as st
from process_damar_analizi import process_images  # Burada process_images fonksiyonunu içeri alıyoruz

# Başlık
st.title("Damar Görüntüleme ve Kümeleme")

# Zip dosyasını yükleme
uploaded_file = st.file_uploader("Zip dosyasını yükleyin", type=["zip"])

# Zip dosyasını çıkarma ve görselleri işleme
if uploaded_file is not None:
    st.write("Zip dosyası alındı. İşlem başlatılıyor...")

    # Geçici zip dosyasını kaydedelim
    temp_zip_path = "/tmp/uploaded_file.zip"
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

    # Görselleri işleme butonu
    if st.button("Analiz Et"):
        # Progress bar'ı başlat
        progress_bar = st.progress(0)
        status_text = st.empty()

        st.write("Görseller işleniyor...")

        # Görselleri işleme işlemini başlat
        analyze_images(unzip_folder, "final_vessel_analysis", progress_bar, status_text)

        st.write("Analiz tamamlandı!")
