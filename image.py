import os
import zipfile
import shutil
import streamlit as st
import time  # İlerleme çubuğu ve spinner için zaman gecikmesi ekleyebilmek için
import pandas as pd

def unzip_file(uploaded_zip, extract_to):
    """Zip dosyasını çıkartan fonksiyon"""
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        return extract_to

def create_zip_from_folder(folder):
    """Bir klasörden zip dosyası oluşturur."""
    zip_name = os.path.join(os.getcwd(), "output_results")  # Zip dosyasının adı
    shutil.make_archive(zip_name, 'zip', folder)  # Klasörün içeriğini zip'e çevir
    return zip_name + '.zip'  # Tam yol ile zip dosyasının yolu döndürülür

# Streamlit arayüzü
st.title("Damar Analiz Aracı")

# Zip dosyasını yüklemek
uploaded_zip = st.file_uploader("Bir zip dosyası yükleyin", type=['zip'])

if uploaded_zip is not None:
    # Zip dosyasını geçici bir klasöre çıkart
    temp_dir = os.path.join(os.getcwd(), 'uploaded_images')
    os.makedirs(temp_dir, exist_ok=True)
    
    st.write("Zip dosyası alındı ve çözüldü...")
    unzip_file(uploaded_zip, temp_dir)
    
    # Zip içindeki dosyaları gör
    image_files = [f for f in os.listdir(temp_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif'))]
    st.write(f"Zip dosyasındaki görseller: {image_files}")
    
    # Analiz et butonu
    if st.button("Analiz Et"):
        with st.spinner("Analiz yapılıyor... Lütfen bekleyin."):
            # Analiz için bir progress bar
            progress_bar = st.progress(0)
            
            # Görseller üzerinde işlem yap
            results = []
            total_files = len(image_files)
            for idx, image_file in enumerate(image_files):
                # Görselleri işle
                img_path = os.path.join(temp_dir, image_file)
                st.image(img_path, caption=f"Orijinal Görüntü: {image_file}")
                
                # Burada gerçek analiz işlemleri yapılacak
                # Örneğin damar tespiti ve metrik hesaplama
                # Şu an sadece bir örnek işlem yapıyoruz
                results.append({
                    'Görüntü Adı': image_file,
                    'Uzunluk': 123.45,  # Örnek uzunluk değeri
                    'Kalınlık': 6.78,   # Örnek kalınlık değeri
                    'Dallanma': 12      # Örnek dallanma değeri
                })
                
                # Progress bar'ı güncelle
                progress_bar.progress((idx + 1) / total_files)
                time.sleep(1)  # Her analiz için kısa bir gecikme ekledik (gerçek analiz için kaldırılabilir)
        
        # Sonuçları göster
        results_df = pd.DataFrame(results)
        st.write("Analiz Sonuçları:", results_df)
        
        # Sonuçları zip dosyası olarak oluştur
        zip_output = create_zip_from_folder(temp_dir)
        st.write(f"Sonuçlar zip dosyası olarak oluşturuldu: {zip_output}")

        # Zip dosyasını indirme linki
        with open(zip_output, "rb") as f:
            st.download_button(
                label="Sonuçları İndir",
                data=f,
                file_name="sonuclar.zip",
                mime="application/zip"
            )
