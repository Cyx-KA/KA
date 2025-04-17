import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from skimage.filters import frangi, meijering, threshold_multiotsu
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from zipfile import ZipFile
import shutil

# Gelişmiş damar tespiti
def enhanced_vessel_detection(image):
    """Gelişmiş damar tespiti fonksiyonu"""
    # Renk kanallarını iyileştir
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    l_channel = clahe.apply(lab[:,:,0])
    
    # Çoklu ölçekli damar tespiti
    thin_vessels = frangi(l_channel, sigmas=range(1,4), alpha=0.5, beta=0.5, gamma=15)
    thick_vessels = frangi(l_channel, sigmas=range(3,8), alpha=0.5, beta=0.5, gamma=10)
    meijering_img = meijering(l_channel, sigmas=range(1,4), black_ridges=False)
    
    # Kombine sonuç
    combined = 0.5*thin_vessels + 0.3*thick_vessels + 0.2*meijering_img
    
    # Çoklu Otsu thresholding
    try:
        thresholds = threshold_multiotsu(combined)
        regions = np.digitize(combined, bins=thresholds)
    except:
        # Eğer threshold bulunamazsa varsayılan değer
        regions = np.digitize(combined, bins=[combined.mean()])
    
    # İnce ve kalın damar maskeleri
    thin_mask = (regions == 2) if len(np.unique(regions)) > 1 else (regions == 1)
    thick_mask = (regions >= 1)
    
    return thin_mask, thick_mask, combined

# Kullanıcı fotoğraflarını analiz etme fonksiyonu
def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    results = []
    processed_count = 0
    
    # ZIP içerisindeki her görseli işle
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
            continue
        
        try:
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # 1. Damar tespiti
            thin_mask, thick_mask, combined = enhanced_vessel_detection(img)
            
            # 2. Kalınlık haritası
            distance = np.zeros_like(thin_mask)
            thickness_map = distance * np.logical_or(thin_mask, thick_mask)
            
            # 3. Metrik hesaplama
            thin_length = np.sum(thin_mask)
            thick_length = np.sum(thick_mask)
            avg_thickness = np.mean(thickness_map[np.logical_or(thin_mask, thick_mask)]) * 2
            total_branches = len(np.unique(thin_mask)) + len(np.unique(thick_mask))
                
            results.append({
                'Image': filename,
                'Total_Vessel_Length': thin_length + thick_length,
                'Thin_Vessel_Length': thin_length,
                'Thick_Vessel_Length': thick_length,
                'Avg_Thickness': avg_thickness,
                'Total_Branches': total_branches
            })
            processed_count += 1
            
            # 4. Görselleştirme
            fig, ax = plt.subplots(2, 3, figsize=(20, 12))
            
            # Orijinal görüntü
            ax[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax[0,0].set_title('Orijinal Görüntü')
            
            # Kırmızı işaretli damarlar
            combined_vessels = np.logical_or(thin_mask, thick_mask)
            red_vessels_image = img.copy()
            red_vessels_image[combined_vessels] = [0, 0, 255]
            ax[0,1].imshow(cv2.cvtColor(red_vessels_image, cv2.COLOR_BGR2RGB))
            ax[0,1].set_title('Kırmızı İşaretli Damarlar')
            
            # Kalınlık haritası
            ax[0,2].imshow(thickness_map, cmap='jet')
            ax[0,2].set_title('Kalınlık Haritası')
            
            # İnce damarlar
            ax[1,0].imshow(thin_mask, cmap='gray')
            ax[1,0].set_title(f'İnce Damarlar ({len(np.unique(thin_mask))} dal)')
            
            # Kalın damarlar
            ax[1,1].imshow(thick_mask, cmap='gray')
            ax[1,1].set_title(f'Kalın Damarlar ({len(np.unique(thick_mask))} dal)')
            
            # Kalınlık haritası 2
            ax[1,2].imshow(np.log1p(thickness_map), cmap='jet')
            ax[1,2].set_title('Logaritmik Kalınlık Haritası')
            
            # Dikey eksende boş alan yok
            for i in range(2):
                for j in range(3):
                    ax[i, j].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f'result_{filename}.png'), dpi=150)
            plt.close()
        
        except Exception as e:
            print(f"Hata: {filename} - {str(e)}")
    
    if processed_count > 0:
        pd.DataFrame(results).to_csv(os.path.join(output_folder, 'results.csv'), index=False)
        print(f"\n✅ {processed_count} görsel başarıyla işlendi! Sonuçlar: {output_folder}")
    else:
        print("\n⚠️ Hiçbir görsel işlenemedi! Girdi klasörünü ve görselleri kontrol edin.")
    
    return results, output_folder

# Streamlit arayüzü
def main():
    st.title('Damar Analizi')
    st.write('Lütfen analiz için görsellerinizi içeren bir klasör yükleyin.')
    
    uploaded_folder = st.file_uploader("Görsel klasörünü yükleyin (ZIP formatında)", type=['zip'], accept_multiple_files=False)
    
    if uploaded_folder is not None:
        # ZIP dosyasını çözmek
        with open("uploaded_folder.zip", "wb") as f:
            f.write(uploaded_folder.getbuffer())
        
        shutil.unpack_archive("uploaded_folder.zip", "uploaded_folder")
        st.write("Görseller başarıyla yüklendi. Şimdi analiz için butona tıklayın.")
        
        # Kullanıcı analize başlamak için buton tıklamalı
        if st.button("Analizi Başlat"):
            input_folder = "uploaded_folder"
            output_folder = "final_vessel_analysis"
            
            # Görselleri işle
            results, output_folder = process_images(input_folder, output_folder)
            
            # Sonuçları göster
            st.write("Sonuçlar:")
            st.dataframe(results)
            
            # Görselleri göster
            for result in results:
                img_path = os.path.join(output_folder, f'result_{result["Image"]}.png')
                st.image(img_path, caption=result["Image"], use_container_width=True)
            
            # Zip dosyasını oluştur ve indir
            zip_filename = f"{output_folder}.zip"
            with ZipFile(zip_filename, 'w') as zipf:
                for filename in os.listdir(output_folder):
                    zipf.write(os.path.join(output_folder, filename), filename)
            
            st.download_button(
                label="Sonuçları İndir",
                data=open(zip_filename, "rb").read(),
                file_name=zip_filename,
                mime="application/zip"
            )

if __name__ == "__main__":
    main()
