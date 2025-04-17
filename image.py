import cv2
import numpy as np
import pandas as pd
from skimage.filters import frangi, meijering, threshold_multiotsu
from skimage.morphology import skeletonize, remove_small_objects, disk
from scipy.ndimage import distance_transform_edt
from skan import Skeleton, summarize
import matplotlib.pyplot as plt
import os
import streamlit as st
from tqdm import tqdm
import tempfile
import shutil
from sklearn.cluster import KMeans

# Gelişmiş damar tespiti fonksiyonu
def enhanced_vessel_detection(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
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

# Hata toleranslı iskelet analizi fonksiyonu
def safe_skeleton_analysis(mask, min_length=500):
    """Hata toleranslı iskelet analizi (orijinal çıktıları koruyarak)"""
    try:
        skeleton = skeletonize(mask)
        skeleton_obj = Skeleton(skeleton)
        
        # Tüm skan versiyonları için uyumlu çağrı
        stats = summarize(skeleton_obj)
        
        # Koordinatları almak için esnek yöntem
        if hasattr(skeleton_obj, 'coordinates'):  # Yeni skan versiyonları
            coords = skeleton_obj.coordinates
            paths = skeleton_obj.paths_list()
            
            # Dalların uzunluklarını hesapla
            branch_info = []
            for path in paths:
                length = 0
                path_coords = []
                for i in range(1, len(path)):
                    p1 = coords[path[i-1]]
                    p2 = coords[path[i]]
                    length += np.linalg.norm(p1 - p2)
                    path_coords.extend([p1, p2])
                branch_info.append({'length': length, 'coords': path_coords})
            
            # Uzun dalları filtrele
            long_vessels_mask = np.zeros_like(mask, dtype=bool)
            for branch in branch_info:
                if branch['length'] > min_length:
                    for coord in branch['coords']:
                        y, x = int(coord[0]), int(coord[1])
                        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                            long_vessels_mask[y, x] = True
            
            # Orijinal stats formatını koru
            stats['branch-distance'] = [b['length'] for b in branch_info]
            stats['image-coord'] = [b['coords'] for b in branch_info]
            
        else:  # Eski skan versiyonları
            if 'image-coord' not in stats.columns and 'coordinates' in stats.columns:
                stats['image-coord'] = stats['coordinates']
            
            long_branches = stats[stats['branch-distance'] > min_length]
            long_vessels_mask = np.zeros_like(mask, dtype=bool)
            
            for _, branch in long_branches.iterrows():
                coords = branch['image-coord'] if 'image-coord' in branch else branch['coordinates']
                if isinstance(coords, (np.ndarray, list)):
                    for coord in coords:
                        y, x = int(coord[0]), int(coord[1])
                        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                            long_vessels_mask[y, x] = True
        
        return stats, long_vessels_mask
    except Exception as e:
        print(f"Skeleton analiz hatası: {str(e)}")
        return pd.DataFrame(), np.zeros_like(mask, dtype=bool)
# Kümeleme fonksiyonu (KMeans)
def cluster_vessels(stats, num_clusters=3):
    # Kümeleme için damar uzunlukları ve dallanma mesafelerini özellik olarak kullanacağız
    features = np.array([stats['branch-distance']])
    features = features.T  # Transpose ederek her satır bir özellik vektörü olacak şekilde düzenliyoruz
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    stats['Cluster'] = clusters
    return stats, kmeans

import os

def process_images(input_images, uploaded_files):
    results = []
    processed_count = 0
    for idx, img in tqdm(enumerate(input_images), total=len(input_images), desc="Damar Analizi"):
        try:
            img_array = np.array(img)
            if img_array is None:
                continue
            
            # Yüklenen dosyanın adından uzantısız kısmı al
            file_name = os.path.splitext(uploaded_files[idx])[0]  # Dosya ismini kullanıyoruz
            
            # Damar tespiti
            thin_mask, thick_mask, combined = enhanced_vessel_detection(img_array)
            
            # İskelet analizi
            thin_stats, _ = safe_skeleton_analysis(thin_mask)
            thick_stats, _ = safe_skeleton_analysis(thick_mask)
            
            # Metrik hesaplama
            if thin_stats is not None and thick_stats is not None:
                thin_length = thin_stats['branch-distance'].sum()
                thick_length = thick_stats['branch-distance'].sum()
                
                # Kalınlık haritası
                distance = distance_transform_edt(np.logical_or(thin_mask, thick_mask))
                thickness_map = distance * np.logical_or(thin_mask, thick_mask)
                
                # Kümeleme işlemi
                thin_stats_with_clusters, _ = cluster_vessels(thin_stats)
                thick_stats_with_clusters, _ = cluster_vessels(thick_stats)
                
                # Sonuçları kaydet
                results.append({
                    'Sample': file_name,  # Yüklenen dosya ismi
                    'Total_Vessel_Length': thin_length + thick_length,
                    'Thin_Vessel_Length': thin_length,
                    'Thick_Vessel_Length': thick_length,
                    'Avg_Thickness': np.mean(thickness_map[np.logical_or(thin_mask, thick_mask)]) * 2,
                    'Total_Branches': len(thin_stats) + len(thick_stats)
                })
                processed_count += 1
                
                # Görselleştirme
                fig, ax = plt.subplots(2, 3, figsize=(20, 12))
                ax[0,0].imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                ax[0,0].set_title(f'Orijinal Görüntü: {file_name}')
                
                combined_vessels = np.logical_or(thin_mask, thick_mask)
                red_vessels_image = img_array.copy()
                red_vessels_image[combined_vessels] = [0, 0, 255]
                
                ax[0,1].imshow(cv2.cvtColor(red_vessels_image, cv2.COLOR_BGR2RGB))
                ax[0,1].set_title(f'Kırmızı İşaretli Damarlar: {file_name}')
                
                im = ax[0,2].imshow(thickness_map, cmap='jet')
                plt.colorbar(im, ax=ax[0,2])
                ax[0,2].set_title('Kalınlık Haritası')
                
                ax[1,0].imshow(thin_mask, cmap='gray')
                ax[1,0].set_title(f'İnce Damarlar ({len(thin_stats)} dal)')
                
                ax[1,1].imshow(thick_mask, cmap='gray')
                ax[1,1].set_title(f'Kalın Damarlar ({len(thick_stats)} dal)')
                
                im2 = ax[1,2].imshow(np.log1p(thickness_map), cmap='jet')
                plt.colorbar(im2, ax=ax[1,2])
                ax[1,2].set_title('Logaritmik Kalınlık Haritası')
                
                for i in range(2):
                    for j in range(3):
                        ax[i, j].axis('off') 
                
                # PNG olarak kaydetme
                output_file = f'output_image_{idx}.png'
                plt.tight_layout()
                plt.savefig(output_file)
                
                # Görseli Streamlit'te gösterme
                st.image(output_file, caption=f'Analiz Görseli: {file_name}', use_container_width=True)
                
                # Görseli silme
                os.remove(output_file)
                
        except Exception as e:
            st.error(f"Hata: {str(e)}")
    
    if processed_count > 0:
        df = pd.DataFrame(results)
        return df
    else:
        st.warning("⚠️ Hiçbir görsel işlenemedi! Girdi klasörünü ve görselleri kontrol edin.")
        return None





# Streamlit GUI
def main():
    st.title("OVOBOARD analiz algoritması by Cyx-KA")
    
    # Görsel dosyalarını yükle
    uploaded_files = st.file_uploader("Görselleri Yükle", type=["jpg", "jpeg", "png", "tif", "tiff"], accept_multiple_files=True)
    
    if uploaded_files:
        filenames = [file.name for file in uploaded_files]
        images = [cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR) for file in uploaded_files]
        
        # Analiz butonu
        if st.button("Analiz Et"):
            with st.spinner("Görseller işleniyor..."):
                results = process_images(images, filenames)
            
            if results is not None:
                st.subheader("Sonuçlar")
                st.write(results)
                st.download_button("Sonuçları CSV olarak indir", data=results.to_csv(), file_name="results.csv", mime="text/csv")
    else:
        st.info("Görselleri yüklemek için 'Görselleri Yükle' butonuna tıklayın.")

if __name__ == "__main__":
    main()
