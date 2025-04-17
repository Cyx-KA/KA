import os
import cv2
import numpy as np
import pandas as pd
from skimage.filters import frangi, meijering
from skimage.morphology import skeletonize, disk
from skimage.measure import regionprops
from skimage.segmentation import clear_border
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

# Damar tespiti fonksiyonu (Frangi ve Meijering filtreleri)
def enhanced_vessel_detection(img):
    # Frangi filtrasyonu ve damar tespiti işlemleri
    thin_mask = frangi(img)  # İnce damarları tespit et
    thick_mask = meijering(img)  # Kalın damarları tespit et
    combined = np.logical_or(thin_mask, thick_mask)  # İnce ve kalın damarları birleştir
    return thin_mask, thick_mask, combined

# İskelet analiz fonksiyonu
def safe_skeleton_analysis(mask):
    skeleton = skeletonize(mask)
    properties = regionprops(skeleton.astype(int))
    stats = {'branch-distance': [p.equivalent_diameter for p in properties]}  # Dal uzunluğu hesaplama
    return stats, skeleton

# Görsel işleme ve analiz fonksiyonu
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

            # Görüntüyü gri tonlara çevir
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1. Damar tespiti
            thin_mask, thick_mask, combined = enhanced_vessel_detection(gray_img)

            # 2. İskelet analizi
            thin_stats, _ = safe_skeleton_analysis(thin_mask)
            thick_stats, _ = safe_skeleton_analysis(thick_mask)

            # 3. Metrik hesaplama
            if thin_stats is not None and thick_stats is not None:
                thin_length = thin_stats['branch-distance'].sum() if 'branch-distance' in thin_stats else 0
                thick_length = thick_stats['branch-distance'].sum() if 'branch-distance' in thick_stats else 0

                # 4. Kalınlık haritası
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

                # 5. Görselleştirme
                fig, ax = plt.subplots(2, 3, figsize=(20, 12))
                ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax[0, 0].set_title('Orijinal Görüntü')

                combined_vessels = np.logical_or(thin_mask, thick_mask)
                red_vessels_image = img.copy()
                red_vessels_image[combined_vessels] = [0, 0, 255]

                ax[0, 1].imshow(cv2.cvtColor(red_vessels_image, cv2.COLOR_BGR2RGB))
                ax[0, 1].set_title('Kırmızı İşaretli Damarlar')

                im = ax[0, 2].imshow(thickness_map, cmap='jet')
                plt.colorbar(im, ax=ax[0, 2])
                ax[0, 2].set_title('Kalınlık Haritası')

                ax[1, 0].imshow(thin_mask, cmap='gray')
                ax[1, 0].set_title(f'İnce Damarlar ({len(thin_stats)} dal)')

                ax[1, 1].imshow(thick_mask, cmap='gray')
                ax[1, 1].set_title(f'Kalın Damarlar ({len(thick_stats)} dal)')

                im2 = ax[1, 2].imshow(np.log1p(thickness_map), cmap='jet')
                plt.colorbar(im2, ax=ax[1, 2])
                ax[1, 2].set_title('Logaritmik Kalınlık Haritası')

                for i in range(2):
                    for j in range(3):
                        ax[i, j].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, f'result_{filename}.png'), dpi=150)
                plt.close()

        except Exception as e:
            print(f"Hata: {filename} - {str(e)}")

    if processed_count > 0:
        result_df = pd.DataFrame(results)
        output_csv = os.path.join(output_folder, 'results.csv')
        result_df.to_csv(output_csv, index=False)
        print(f"{processed_count} görsel başarıyla işlendi ve sonuçlar {output_csv} dosyasına kaydedildi!")
    else:
        print("Hiçbir görsel işlenemedi!")

# Kümeleme analizi fonksiyonu
def cluster_analysis(input_file, output_folder):
    df = pd.read_csv(input_file)

    features = ['Total_Vessel_Length', 'Thin_Vessel_Length', 'Thick_Vessel_Length', 'Avg_Thickness', 'Total_Branches']
    X = df[features].copy()

    # Normalize et
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA ile 2 boyuta indir
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # K-Means ile kümeleme
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_pca)

    # Küme sonuçlarını dataframe'e ekle
    df['Cluster'] = clusters
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]

    # Kümeleme grafiği
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=100)
    for i, row in df.iterrows():
        plt.text(row['PCA1'] + 0.05, row['PCA2'] + 0.05, row['Image'], fontsize=9)

    plt.title("Kümeleme: Görsel bazlı damar verileri")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.legend(title='Küme')
    plt.tight_layout()
    cluster_image_path = os.path.join(output_folder, "cluster_visualization.png")
    plt.savefig(cluster_image_path, dpi=150)
    plt.show()

    # CSV dosyasına kaydet
    clustered_data_path = os.path.join(output_folder, "clustered_vessel_data.csv")
    df.to_csv(clustered_data_path, index=False)
    print(f"Kümeleme sonuçları {cluster_image_path} ve {clustered_data_path} dosyalarına kaydedildi.")
