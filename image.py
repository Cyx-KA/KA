import streamlit as st
import os
import cv2
import numpy as np
from skimage.filters import frangi, meijering, threshold_multiotsu
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from io import BytesIO

# Gelişmiş damar tespiti fonksiyonu
def enhanced_vessel_detection(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    l_channel = clahe.apply(lab[:,:,0])
    
    thin_vessels = frangi(l_channel, sigmas=range(1,4), alpha=0.5, beta=0.5, gamma=15)
    thick_vessels = frangi(l_channel, sigmas=range(3,8), alpha=0.5, beta=0.5, gamma=10)
    meijering_img = meijering(l_channel, sigmas=range(1,4), black_ridges=False)
    
    combined = 0.5 * thin_vessels + 0.3 * thick_vessels + 0.2 * meijering_img
    
    try:
        thresholds = threshold_multiotsu(combined)
        regions = np.digitize(combined, bins=thresholds)
    except:
        regions = np.digitize(combined, bins=[combined.mean()])
    
    thin_mask = (regions == 2) if len(np.unique(regions)) > 1 else (regions == 1)
    thick_mask = (regions >= 1)
    
    return thin_mask, thick_mask, combined

# Görseli Streamlit'te gösterme fonksiyonu
def show_image(image, caption=""):
    st.image(image, caption=caption, use_column_width=True)

# Ana UI fonksiyonu
def main():
    st.title("Damar Tespiti ve Analiz Sistemi")
    st.write("Görselleri yükleyerek damar tespiti ve analizlerini yapabilirsiniz.")
    
    uploaded_files = st.file_uploader("Görselleri Yükleyin", accept_multiple_files=True, type=['jpg', 'jpeg', 'png', 'tif', 'tiff'])
    
    if uploaded_files:
        for file in uploaded_files:
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            if image is not None:
                # Damar tespiti
                thin_mask, thick_mask, combined = enhanced_vessel_detection(image)
                
                # Görselleri oluşturma
                red_vessels_image = image.copy()
                combined_vessels = np.logical_or(thin_mask, thick_mask)
                red_vessels_image[combined_vessels] = [0, 0, 255]  # Kırmızı işaretleme
                
                # Kalınlık haritası
                distance = distance_transform_edt(np.logical_or(thin_mask, thick_mask))
                thickness_map = distance * np.logical_or(thin_mask, thick_mask)
                
                # Gösterimler
                st.subheader(f"{file.name} Görseli")
                show_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "Orijinal Görüntü")
                show_image(cv2.cvtColor(red_vessels_image, cv2.COLOR_BGR2RGB), "Kırmızı İşaretli Damarlar")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(thickness_map, cmap='jet')
                plt.colorbar(im, ax=ax)
                ax.set_title("Kalınlık Haritası")
                st.pyplot(fig)
                
                # Daha fazla analiz eklenebilir
                # Örneğin; damar uzunluğu, branş sayısı gibi metrikler
            else:
                st.write(f"{file.name} görseli okunamadı.")

if __name__ == "__main__":
    main()
