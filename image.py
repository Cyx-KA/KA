import streamlit as st
import zipfile
import os
import cv2
import numpy as np
import pandas as pd
from skimage.filters import frangi, meijering, threshold_multiotsu
from skimage.morphology import skeletonize, remove_small_objects, disk
from scipy.ndimage import distance_transform_edt
from skan import Skeleton, summarize
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import tempfile
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Geçici dosya dizinine ZIP dosyasını çıkarma fonksiyonu
def extract_zip(zip_file):
    try:
        # Geçici dizin oluşturuluyor
        temp_dir = './temp_extracted_files'
        
        # Geçici dizin yoksa oluşturuyoruz
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # ZIP dosyasını çıkarıyoruz
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        return temp_dir
    except Exception as e:
        print(f"ZIP çıkarma hatası: {e}")
        raise

# 2. Damar analiz fonksiyonları
def enhanced_vessel_detection(image):
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

def process_images(input_folder):
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

                fig, ax = plt.subplots(2, 3, figsize=(20, 12))

                ax[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax[0,0].set_title('Orijinal Görüntü')

                combined_vessels = np.logical_or(thin_mask, thick_mask)
                red_vessels_image = img.copy()
                red_vessels_image[combined_vessels] = [0, 0, 255]
                ax[0,1].imshow(cv2.cvtColor(red_vessels_image, cv2.COLOR_BGR2RGB))
                ax[0,1].set_title('Kırmızı İşaretli Damarlar')

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
                
                plt.tight_layout()
                plt.savefig(f'./temp_extracted_files/results/result_{filename}.png', dpi=150)
                plt.close()

        except Exception as e:
            print(f"Hata: {filename} - {str(e)}")

    if processed_count > 0:
        pd.DataFrame(results).to_csv('./temp_extracted_files/results.csv', index=False)
        print(f"\n✅ {processed_count} görsel başarıyla işlendi!")
    else:
        print("\n⚠️ Hiçbir görsel işlenemedi!")

# 3. Kümeleme işlemleri
def cluster_analysis():
    df = pd.read_csv("./temp_extracted_files/results.csv")

    features = ['Total_Vessel_Length', 'Thin_Vessel_Length', 'Thick_Vessel_Length', 'Avg_Thickness', 'Total_Branches']
    X = df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_pca)

    df['Cluster'] = clusters
    df['PCA1'] = X_pca[:,0]
    df['PCA2'] = X_pca[:,1]

    plt.figure(figsize=(10,7))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=100)

    for i, row in df.iterrows():
        plt.text(row['PCA1']+0.05, row['PCA2']+0.05, row['Image'], fontsize=9)

    plt.title("Kümeleme: Görsel bazlı damar verileri")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.legend(title='Küme')
    plt.tight_layout()
    plt.savefig('./temp_extracted_files/cluster_visualization.png', dpi=150)
    plt.show()

    df.to_csv("./temp_extracted_files/clustered_vessel_data.csv", index=False)

# 4. Streamlit UI
st.title("Damar ve Kümeleme Analizi")
st.write("Lütfen analiz etmek için bir ZIP dosyası yükleyin.")

uploaded_zip = st.file_uploader("ZIP Dosyasını Yükle", type=["zip"])

if uploaded_zip:
    with open('uploaded.zip', 'wb') as f:
        f.write(uploaded_zip.read())

    try:
        temp_dir = extract_zip('uploaded.zip')
        st.write(f"Görüntü sayısı: {len([f for f in os.listdir(temp_dir) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png'))])}")
        
        if st.button('Analizi Başlat'):
            with st.spinner('Analiz ediliyor...'):
                process_images(temp_dir)
                cluster_analysis()
                st.success("Analiz tamamlandı! Görselleştirmelere göz atabilirsiniz.")
                
    except Exception as e:
        st.error(f"Dosya çıkarma sırasında hata oluştu: {e}")
