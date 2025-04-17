import os
import zipfile
import shutil
import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Fotoğraf işleme fonksiyonları (Örneğin, damar analizi gibi)
def process_image(image_path):
    """Bir görsel üzerinde analiz yapar ve işlenmiş görseli döndürür"""
    # Burada örnek bir işleme yapılır, siz kendi kodunuzu ekleyebilirsiniz
    img = Image.open(image_path)
    img = img.convert('RGB')  # Örneğin, RGB'ye dönüştürme
    # Basit bir işleme örneği (gri tonlara dönüştürme)
    img = img.convert('L')
    return img

def create_figure_from_images(images):
    """Görselleri 6'lı bir figürde düzenler ve döndürür"""
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.ravel()
    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].axis('off')  # Eksenleri gizle
    plt.tight_layout()
    return fig

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

def save_cluster_image(images, cluster_path):
    """Kümeleme sonuçları için bir görsel kaydeder (örneğin 6'lı görsel grid)"""
    fig = create_figure_from_images(images)
    fig.savefig(cluster_path)
    plt.close(fig)

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
            
            # Görselleri işleme
            processed_images = []
            results = []
            total_files = len(image_files)
            
            # Görselleri işleyerek analiz yap
            for idx, image_file in enumerate(image_files):
                img_path = os.path.join(temp_dir, image_file)
                processed_img = process_image(img_path)
                processed_images.append(processed_img)
                
                # Her görselin analizi için sonuç ekleyin (Örnek metrikler)
                results.append({
                    'Görüntü Adı': image_file,
                    'Uzunluk': 123.45,  # Örnek uzunluk değeri
                    'Kalınlık': 6.78,   # Örnek kalınlık değeri
                    'Dallanma': 12      # Örnek dallanma değeri
                })
                
                # Progress bar'ı güncelle
                progress_bar.progress((idx + 1) / total_files)
                time.sleep(1)  # Gerçek analizde bu gecikme kaldırılabilir
        
        # Sonuçları DataFrame olarak göster
        results_df = pd.DataFrame(results)
        st.write("Analiz Sonuçları:", results_df)

        # Kümeleme sonuçlarını görsel olarak oluştur
        cluster_path = os.path.join(os.getcwd(), 'cluster.png')
        save_cluster_image(processed_images, cluster_path)
        st.image(cluster_path, caption="Kümeleme Sonuçları")

        # Sonuçları zip dosyasına ekle
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
        
        # İşlenmiş görsellerin küçük versiyonlarını simge olarak göster
        st.write("İşlenmiş Görseller:")
        for idx, img in enumerate(processed_images):
            st.image(img, caption=f"İşlenmiş Görüntü {image_files[idx]}", width=150, use_column_width=False)

            # Simgeye tıklandığında büyük versiyonu yeni sekmede açmak için link
            img_path = os.path.join(temp_dir, image_files[idx])
            st.markdown(f'<a href="file://{img_path}" target="_blank">Büyük Görüntü</a>', unsafe_allow_html=True)
