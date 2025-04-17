import streamlit as st
import zipfile
import os
import shutil

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
