import cv2
import numpy as np
import pandas as pd
from skimage.filters import frangi, meijering
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from skan import Skeleton, summarize

def process_single_image(img_path):
    """Tek bir görüntü için damar analizi yapar"""
    img = cv2.imread(img_path)
    
    # Damar tespiti (sizin kodunuz)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    l_channel = clahe.apply(lab[:,:,0])
    
    thin_vessels = frangi(l_channel, sigmas=range(1,4))
    thick_vessels = frangi(l_channel, sigmas=range(3,8))
    combined = 0.5*thin_vessels + 0.5*thick_vessels
    
    # Maskeler
    thin_mask = combined > combined.mean()
    thick_mask = combined > combined.mean() * 1.5
    
    # İskelet analizi
    thin_skeleton = skeletonize(thin_mask)
    thick_skeleton = skeletonize(thick_mask)
    
    # Metrikler
    thin_length = np.sum(thin_skeleton)
    thick_length = np.sum(thick_skeleton)
    
    # Sonuç DataFrame
    csv_data = pd.DataFrame({
        "Image": [os.path.basename(img_path)],
        "Total_Vessel_Length": [thin_length + thick_length],
        "Thin_Vessel_Length": [thin_length],
        "Thick_Vessel_Length": [thick_length],
        "Avg_Thickness": [np.mean(combined[combined > 0])],
        "Total_Branches": [np.sum(thin_skeleton) + np.sum(thick_skeleton)]
    })
    
    # Görselleştirme
    result_img = img.copy()
    result_img[thin_mask] = [0, 0, 255]  # İnce damarlar kırmızı
    result_img[thick_mask] = [0, 255, 0]  # Kalın damarlar yeşil
    
    return result_img, csv_data
