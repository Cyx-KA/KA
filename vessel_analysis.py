import cv2
import numpy as np
import pandas as pd
from skimage.filters import frangi, meijering, threshold_multiotsu
from skimage.morphology import skeletonize
from skimage.morphology import remove_small_objects, disk
from skan import Skeleton, summarize
from scipy.ndimage import distance_transform_edt
import os
from tqdm import tqdm

def enhanced_vessel_detection(image):
    """Gelişmiş damar tespiti fonksiyonu"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    l_channel = clahe.apply(lab[:,:,0])
    
    thin_vessels = frangi(l_channel, sigmas=range(1,4), alpha=0.5, beta=0.5, gamma=15)
    thick_vessels = frangi(l_channel, sigmas=range(3,8), alpha=0.5, beta=0.5, gamma=10)
    meijering_img = meijering(l_channel, sigmas=range(1,4), black_ridges=False)
    
    combined = 0.5*thin_vessels + 0.3*thick_vessels + 0.2*meijering_img
    try:
        thresholds = threshold_multiotsu(combined)
        regions = np.digitize(combined, bins=thresholds)
    except:
        regions = np.digitize(combined, bins=[combined.mean()])
    
    thin_mask = (regions == 2) if len(np.unique(regions)) > 1 else (regions == 1)
    thick_mask = (regions >= 1)
    
    return thin_mask, thick_mask, combined

def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    results = []
    processed_count = 0
    
    for filename in tqdm(sorted(os.listdir(input_folder)), desc="Damar Analizi"):
        if not filename.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
            continue
        
        try:
            img = cv2.imread(os.path.join(input_folder, filename))
            if img is None:
                continue
            
            thin_mask, thick_mask, combined = enhanced_vessel_detection(img)
            
            distance = distance_transform_edt(np.logical_or(thin_mask, thick_mask))
            thickness_map = distance * np.logical_or(thin_mask, thick_mask)
            
            thin_length = np.sum(thin_mask)
            thick_length = np.sum(thick_mask)
            
            results.append({
                'Image': filename,
                'Total_Vessel_Length': thin_length + thick_length,
                'Thin_Vessel_Length': thin_length,
                'Thick_Vessel_Length': thick_length,
                'Avg_Thickness': np.mean(thickness_map[np.logical_or(thin_mask, thick_mask)]) * 2,
                'Total_Branches': len(np.unique(thin_mask)) + len(np.unique(thick_mask))
            })
            processed_count += 1
                
        except Exception as e:
            print(f"Hata: {filename} - {str(e)}")
    
    if processed_count > 0:
        pd.DataFrame(results).to_csv(os.path.join(output_folder, 'results.csv'), index=False)
        print(f"\n✅ {processed_count} görsel başarıyla işlendi! Sonuçlar: {output_folder}")
    else:
        print("\n⚠️ Hiçbir görsel işlenemedi! Girdi klasörünü ve görselleri kontrol edin.")
