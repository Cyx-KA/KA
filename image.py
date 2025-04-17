import os
import time
import zipfile
import cv2
import streamlit as st
from process_damar_analizi import process_images  # process_images fonksiyonunuz
import tempfile
import shutil

# Eksik olan analyze_images fonksiyonunu tanımlayalım
def analyze_images(input_folder, output_folder, progress_bar=None, status_text=None):
    """Görüntüleri işleyip analiz eden ana fonksiyon"""
    try:
        # Çıktı klasörünü oluştur
        os.makedirs(output_folder, exist_ok=True)
        
        # Görüntü dosyalarını listele
        image_files = [f for f in os.listdir(input_folder) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        if not image_files:
            if status_text:
                status_text.text("Hiçbir görsel bulunamadı!")
            return False

        results = []
        total = len(image_files)
        
        for i, filename in enumerate(image_files):
            # İlerleme durumunu güncelle
            if progress_bar:
                progress_bar.progress((i + 1) / total)
            if status_text:
                status_text.text(f"İşleniyor: {filename} ({i+1}/{total})")
            
            # Görseli işle
            img_path = os.path.join(input_folder, filename)
            result_img, csv_data = process_images(img_path)  # Sizin fonksiyonunuz
            
            # Sonuçları kaydet
            output_path = os.path.join(output_folder, f"result_{filename}")
            cv2.imwrite(output_path, result_img)
            results.append(csv_data.iloc[0].to_dict())
        
        # CSV'yi kaydet
        if results:
            pd.DataFrame(results).to_csv(os.path.join(output_folder, "results.csv"), index=False)
            return True
        return False
        
    except Exception as e:
        if status_text:
            status_text.text(f"Hata: {str(e)}")
        return False

# Streamlit arayüzü
st.title("Damar Görüntüleme ve Kümeleme")

# Zip yükleme
uploaded_file = st.file_uploader("Zip dosyasını yükleyin", type=["zip"])

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Zip'i geçici dizine çıkar
        try:
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
            st.success("Görseller başarıyla çıkarıldı!")
            
            # Analiz butonu
            if st.button("Analiz Et"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Çıktı klasörü
                output_dir = "final_vessel_analysis"
                os.makedirs(output_dir, exist_ok=True)
                
                # Analiz işlemi
                if analyze_images(tmp_dir, output_dir, progress_bar, status_text):
                    st.success("Analiz tamamlandı!")
                    
                    # Sonuçları göster
                    st.subheader("İşlenmiş Görseller")
                    result_files = [f for f in os.listdir(output_dir) 
                                  if f.startswith('result_') and f.lower().endswith(('.png', '.jpg'))]
                    
                    cols = st.columns(3)
                    for idx, img_file in enumerate(result_files):
                        with cols[idx % 3]:
                            st.image(os.path.join(output_dir, img_file), 
                                   caption=img_file.replace('result_', ''))
                    
                    # CSV indirme butonu
                    with open(os.path.join(output_dir, "results.csv"), "rb") as f:
                        st.download_button(
                            label="Sonuçları İndir (CSV)",
                            data=f,
                            file_name="damar_analiz_sonuclari.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("Analiz sırasında hata oluştu!")
                    
        except Exception as e:
            st.error(f"Zip işlenirken hata: {str(e)}")
