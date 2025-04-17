import cv2
import numpy as np
import os
from tqdm import tqdm
from skimage.filters import frangi, meijering, threshold_multiotsu
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from skan import Skeleton, summarize
import pandas as pd
import matplotlib.pyplot as plt

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

def safe_skeleton_analysis(mask, min_length=500):
    try:
        skeleton = skeletonize(mask)
        skeleton_obj = Skeleton(skeleton)
        stats = summarize(skeleton_obj, separator='-')
        long_branches = stats[stats['branch-distance'] > min_length]
        long_vessels_mask = np.zeros_like(mask, dtype=bool)
        for _, branch in long_branches.iterrows():
            coords = branch['image-coord']
            for coord in coords:
                long_vessels_mask[int(coord[0]), int(coord[1])] = True
        return stats, long_vessels_mask
    except Exception as e:
        print(f"Skeleton analiz hatası: {str(e)}")
        return None, None

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
            thin_stats, _ = safe_skeleton_analysis(thin_mask)
            thick_stats, _ = safe_skeleton_analysis(thick_mask)
            if thin_stats is not None and thick_stats is not None:
                thin_length = thin_stats['branch-distance'].sum()
                thick_length = thick_stats['branch-distance'].sum()
                distance = distance_transform_edt(np.logical_or(thin_mask, thick_mask))
                thickness_map = distance * np.logical_or(thin_mask, thick_mask)
                results.append({
                    'Image': filename,
                    'Total_Vessel_Length': thin_length + thick_length,
                    'Thin_Vessel_Length': thin_length,
                    'Thick_Vessel_Length': thick_length,
                    'Avg_Thickness': np.mean(thickness_map[np.logical_or(thin_mask, thick_mask)]) * 2,
                    'Total_Branches': len(thin_stats) + len(thick_stats)
                })
                processed_count += 1
        except Exception as e:
            print(f"Hata: {filename} - {str(e)}")
    if processed_count > 0:
        pd.DataFrame(results).to_csv(os.path.join(output_folder, 'results.csv'), index=False)
        print(f"\n✅ {processed_count} görsel başarıyla işlendi! Sonuçlar: {output_folder}")
    else:
        print("\n⚠️ Hiçbir görsel işlenemedi!")

# Ana işlem akışını başlatın
input_folder = "input_images"
output_folder = "final_vessel_analysis"
process_images(input_folder, output_folder)
