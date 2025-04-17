import os
import zipfile
import pandas as pd
import cv2
import numpy as np
from skimage.filters import frangi, meijering, threshold_multiotsu
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
import streamlit as st
import shutil
from tqdm import tqdm
from io import BytesIO

def enhanced_vessel_detection(image):
    """Gelişmiş damar tespiti fonksiyonu"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    l_channel = clahe.apply(lab[:,:,0])
    
    thin_vessels = frangi(l_channel, sigmas=range(1,4), alpha=0.5, beta=0.5, gamma=15)
    thick_vessels = frangi(l_channel, sigmas=range(3,8), alpha=0.5, beta=0.5, gamma=10)
    meijering_img = meijering(l_channel, sigmas=range(1,4), black_ridges=False)
    
    combined = 0.5*thin_vessels + 0.3*thick_vessels + 0.2*meijering_img
    thresholds = threshold_multiotsu(combined)
    regions = np.digitize(combined, bins=thresholds)
    
    thin_mask = (regions == 2) if len(np.unique(regions)) > 1 else (regions == 1)
    thick_mask = (regions >= 1)
    
    return thin_mask, thick_mask, combined

def safe_skeleton_analysis(mask, min_length=500):
    """Hata toleranslı iskelet analizi"""
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
                    'Görüntü': filename,
                    'Toplam_Uzunluk': thin_length + thick_length,
                    'İnce_Uzunluk': thin_length,
                    'Kalın_Uzunluk': thick_length,
                    'Ortalama_Kalınlık': np.mean(thickness_map[np.logical_or(thin_mask, thick_mask)]) * 2,
                    'Toplam_Dallanma': len(thin_stats) + len(thick_stats)
                })
                processed_count += 1
                
        except Exception as e:
            print(f"Hata: {filename} - {str(e)}")
    
    if processed_count > 0:
        pd.DataFrame(results).to_csv(os.path.join(output_folder, 'sonuclar.csv'), index=False)
        print(f"\n✅ {processed_count} görsel başarıyla işlendi! Sonuçlar: {output_folder}")
    else:
        print("\n⚠️ Hiçbir görsel işlenemedi! Girdi klasörünü ve görselleri kontrol edin.")

def extract_zip(uploaded_file):
    """ZIP dosyasını çıkarma fonksiyonu"""
    with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
        zip_ref.extractall("input_images")
    return "input_images"

def create_zip_from_folder(folder):
    """Klasörden ZIP dosyası oluşturma fonksiyonu"""
    output = BytesIO()
    shutil.make_archive(output, 'zip', folder)
    return output.getvalue()

# Streamlit arayüzü
st.title("Damar Analizi Uygulaması")

uploaded_file = st.file_uploader("Zip dosyasını yükleyin", type=["zip"])

if uploaded_file:
    # ZIP dosyasını çıkar
    folder_path = extract_zip(uploaded_file)
    st.write("Yüklenen fotoğraflar analiz edilecek...")
    
    # 'Analiz et' butonu
    if st.button("Analiz Et"):
        output_folder = "final_vessel_analysis"
        process_images(folder_path, output_folder)
        
        # Analiz tamamlandığında zip dosyası oluşturulacak
        zip_data = create_zip_from_folder(output_folder)
        
        # Kullanıcıya zip dosyasını indirme linki sun
        st.download_button("Sonuçları indir (ZIP formatında)", zip_data, file_name="sonuc.zip", mime="application/zip")
        
        # Sonuçları csv olarak göster
        results_df = pd.read_csv(os.path.join(output_folder, 'sonuclar.csv'))
        st.write(results_df)
