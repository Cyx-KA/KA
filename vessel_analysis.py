import cv2
import numpy as np
import pandas as pd
import os
from skimage.filters import frangi, meijering
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from skan import Skeleton
import logging

# --- 1. HATA YÖNETİMİ ---
logging.basicConfig(filename='vessel_analysis.log', level=logging.ERROR)

# --- 2. PERFORMANS İYİLEŞTİRMELERİ ---
@st.cache_data
def process_single_image(img_path):
    """Tek görüntü için optimize edilmiş damar analizi"""
    try:
        # --- GÖRÜNTÜ YÜKLEME ---
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Geçersiz görüntü: {img_path}")
        
        # --- BOYUT STANDARDİZASYONU ---
        img = cv2.resize(img, (1024, 1024))  # Tüm görseller aynı boyutta
        
        # --- DAMAR TESPİTİ ---
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_channel = clahe.apply(lab[:,:,0])
        
        # Çoklu ölçekli damar tespiti
        thin = frangi(l_channel, sigmas=range(1,3), gamma=0.8)
        thick = frangi(l_channel, sigmas=range(3,6), gamma=0.5)
        combined = 0.6*thin + 0.4*thick
        
        # --- MASKELER ---
        thresh = np.percentile(combined, 95)  # 95. persentil eşik
        thin_mask = combined > thresh
        thick_mask = combined > (thresh * 1.2)
        
        # --- METRİK HESAPLAMA ---
        thin_skel = skeletonize(thin_mask)
        thick_skel = skeletonize(thick_mask)
        
        metrics = {
            "Image": os.path.basename(img_path),
            "Total_Vessel_Length": np.sum(thin_skel) + np.sum(thick_skel),
            "Thin_Vessel_Length": np.sum(thin_skel),
            "Thick_Vessel_Length": np.sum(thick_skel),
            "Avg_Thickness": distance_transform_edt(thick_mask).mean(),
            "Total_Branches": len(np.unique(Skeleton(thick_skel).graph['edge'].ravel()))
        }
        
        # --- GÖRSELLEŞTİRME ---
        result_img = img.copy()
        result_img[thin_mask] = [255, 0, 0]  # Mavi (BGR format)
        result_img[thick_mask] = [0, 0, 255]  # Kırmızı
        
        return result_img, pd.DataFrame([metrics])
        
    except Exception as e:
        logging.error(f"{img_path} işlenirken hata: {str(e)}")
        # Hata durumunda boş bir görsel ve sıfır değerler döndür
        return np.zeros((256,256,3), dtype=np.uint8), pd.DataFrame({
            "Image": [os.path.basename(img_path)],
            "Total_Vessel_Length": [0],
            "Thin_Vessel_Length": [0],
            "Thick_Vessel_Length": [0],
            "Avg_Thickness": [0],
            "Total_Branches": [0]
        })
