import streamlit as st
from process_damar_analizi import process_images, cluster_analysis
import os

# Başlık
st.title("Damar Görüntüleme ve Kümeleme")

# Kullanıcıdan zip dosyası alma
uploaded_file = st.file_uploader("Zip dosyasını yükleyin", type=["zip"])

# Zip dosyasını açma ve içerikleri işleme
if uploaded_file is not None:
    st.write("Zip dosyası alındı. İşlem başlatılıyor...")
    
    # Zip dosyasını aç
    with open("uploaded_file.zip", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Dosyayı çıkar
    unzip_folder = "input_images"
    os.makedirs(unzip_folder, exist_ok=True)
    with zipfile.ZipFile("uploaded_file.zip", 'r') as zip_ref:
        zip_ref.extractall(unzip_folder)
    st.write("Görseller başarıyla çıkarıldı!")

    # Görselleri işleme
    output_folder = "final_vessel_analysis"
    os.makedirs(output_folder, exist_ok=True)
    
    # Damar analizi
    process_images(unzip_folder, output_folder)

    # Kümeleme analizi
    cluster_analysis(os.path.join(output_folder, "results.csv"))
    
    st.write("Analiz tamamlandı.")
else:
    st.write("Lütfen bir zip dosyası yükleyin.")
