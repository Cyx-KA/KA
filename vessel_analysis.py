import cv2
import numpy as np
import pandas as pd
import os
from skimage.filters import frangi
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from skan import Skeleton
import logging

# Streamlit cache yerine basit bir cache mekanizması
_cache = {}

def process_single_image(img_path):
    """Optimize edilmiş damar analizi fonksiyonu"""
    try:
        # Önbellek kontrolü
        if img_path in _cache:
            return _cache[img_path]
            
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Geçersiz görüntü: {img_path}")

        # Boyut standardizasyonu
        img = cv2.resize(img, (1024, 1024))
        
        # Damar tespiti
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_channel = clahe.apply(lab[:,:,0])
        
        thin = frangi(l_channel, sigmas=range(1,3))
        thick = frangi(l_channel, sigmas=range(3,6))
        combined = 0.6*thin + 0.4*thick
        
        # Maskeler
        thresh = np.percentile(combined, 95)
        thin_mask = combined > thresh
        thick_mask = combined > (thresh * 1.2)
        
        # Metrikler
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
        
        # Görselleştirme
        result_img = img.copy()
        result_img[thin_mask] = [255, 0, 0]  # BGR: Mavi
        result_img[thick_mask] = [0, 0, 255]  # BGR: Kırmızı
        
        # Önbelleğe al
        _cache[img_path] = (result_img, pd.DataFrame([metrics]))
        return result_img, pd.DataFrame([metrics])
        
    except Exception as e:
        logging.error(f"Hata: {str(e)}")
        return np.zeros((256,256,3), pd.DataFrame({
            "Image": [os.path.basename(img_path)],
            "Total_Vessel_Length": [0],
            "Thin_Vessel_Length": [0],
            "Thick_Vessel_Length": [0],
            "Avg_Thickness": [0],
            "Total_Branches": [0]
        })
