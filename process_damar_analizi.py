import os
import cv2
import numpy as np
import pandas as pd
from skimage.filters import frangi, meijering, threshold_multiotsu
from skimage.morphology import skeletonize, disk
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import matplotlib.pyplot as plt

# Damar tespiti fonksiyonu (örnek bir işlem)
def enhanced_vessel_detection(img):
    # Frangi filtrasyonu ve damar tespiti işlemleri
    # Bu kısmı gerçek işleme kodunuza göre uyarlayabilirsiniz
    thin_mask = np.zeros_like(img)
    thick_mask = np.zeros_like(img)
    combined = np.zeros_like(img)
    return thin_mask, thick_mask, combined

# İskelet analiz fonksiyonu
def safe_skeleton_analysis(mask):
    skeleton = skeletonize(mask)
    stats = {'branch-distance': np.random.rand(10)}  # Bu, gerçek hesaplamaya göre değişir
    return stats, skeleton

# Görsel işleme fonksiyonu
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

            # 1. Damar tespiti
            thin_mask, thick_mask, combined = enhanced_vessel_detection(img)

            # 2. İskelet analizi
            thin_stats, _ = safe_skeleton_analysis(thin_mask)
            thick_stats, _ = safe_skeleton_analysis(thick_mask)

            # 3. Metrik hesaplama
            if thin_stats is not None and thick_stats is not None:
                thin_length = thin_stats['branch-distance'].sum()
                thick_length = thick_stats['branch-distance'].sum()

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
        pd.DataFrame(results).to_csv(os.path.join(output_folder, 'results.csv'), index=False)
        print(f"{processed_count} görsel başarıyla işlendi!")
    else:
        print("Hiçbir görsel işlenemedi!")

# Kümeleme analizi fonksiyonu
def cluster_analysis(input_file):
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
    plt.savefig("cluster_visualization.png", dpi=150)
    plt.show()

    df.to_csv("clustered_vessel_data.csv", index=False)
