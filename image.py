import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zipfile import ZipFile
import shutil

from skimage.filters import frangi, meijering, threshold_multiotsu
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# === Damar analiz fonksiyonu ===
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

# === Görselleri işleyip CSV ve görselleri üreten fonksiyon ===
def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    results = []

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
            continue

        img = cv2.imread(os.path.join(input_folder, filename))
        if img is None:
            continue

        thin_mask, thick_mask, combined = enhanced_vessel_detection(img)
        distance = np.zeros_like(thin_mask)
        thickness_map = distance * np.logical_or(thin_mask, thick_mask)

        thin_length = np.sum(thin_mask)
        thick_length = np.sum(thick_mask)
        avg_thickness = np.mean(thickness_map[np.logical_or(thin_mask, thick_mask)]) * 2
        total_branches = len(np.unique(thin_mask)) + len(np.unique(thick_mask))

        results.append({
            'Image': filename,
            'Uzunluk': thin_length + thick_length,
            'İnce': thin_length,
            'Kalın': thick_length,
            'Kalınlık': avg_thickness,
            'Dallanma': total_branches
        })

        fig, ax = plt.subplots(2, 3, figsize=(20, 12))
        ax[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0,0].set_title('Orijinal Görüntü')
        red_vessels_image = img.copy()
        red_vessels_image[np.logical_or(thin_mask, thick_mask)] = [0, 0, 255]
        ax[0,1].imshow(cv2.cvtColor(red_vessels_image, cv2.COLOR_BGR2RGB))
        ax[0,1].set_title('Kırmızı İşaretli Damarlar')
        ax[0,2].imshow(thickness_map, cmap='jet')
        ax[0,2].set_title('Kalınlık Haritası')
        ax[1,0].imshow(thin_mask, cmap='gray')
        ax[1,0].set_title(f'İnce Damarlar')
        ax[1,1].imshow(thick_mask, cmap='gray')
        ax[1,1].set_title(f'Kalın Damarlar')
        ax[1,2].imshow(np.log1p(thickness_map), cmap='jet')
        ax[1,2].set_title('Log Kalınlık Haritası')

        for i in range(2):
            for j in range(3):
                ax[i, j].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'result_{filename}.png'))
        plt.close()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_folder, 'results.csv'), index=False)
    return df, output_folder

# === Kümeleme fonksiyonu ===
def clustering_analysis(df, output_folder):
    features = ['Uzunluk', 'İnce', 'Kalın', 'Kalınlık', 'Dallanma']
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_pca)

    df['Küme'] = clusters
    df['PCA1'] = X_pca[:,0]
    df['PCA2'] = X_pca[:,1]

    plt.figure(figsize=(10,7))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Küme', palette='Set1', s=100)
    for i, row in df.iterrows():
        plt.text(row['PCA1']+0.05, row['PCA2']+0.05, row['Image'], fontsize=9)

    plt.title("Kümeleme: Görsel bazlı damar verileri")
    plt.savefig(os.path.join(output_folder, "cluster_visualization.png"), dpi=150)
    plt.close()

    df.to_csv(os.path.join(output_folder, "clustered_vessel_data.csv"), index=False)

# === Streamlit arayüzü ===
def main():
    st.set_page_config(page_title="Damar Analizi", layout="wide")
    st.title("🧬 Damar Görsel Analizi")
    uploaded = st.file_uploader("Bir zip dosyası yükleyin", type=["zip"])

    if uploaded:
        with st.spinner("Zip dosyası açılıyor..."):
            with open("uploaded.zip", "wb") as f:
                f.write(uploaded.getbuffer())
            shutil.unpack_archive("uploaded.zip", "uploaded_folder")

        st.success("📂 Dosyalar açıldı. Analiz başlatılıyor...")

        if st.button("🔍 Analiz Et"):
            with st.spinner("Görseller analiz ediliyor..."):
                df, output_folder = process_images("uploaded_folder", "output")
                clustering_analysis(df, output_folder)

            st.success("✅ Analiz tamamlandı!")

            st.subheader("📊 Sonuçlar")
            st.dataframe(df)

            st.subheader("📁 Analiz Görselleri")
            for file in os.listdir(output_folder):
                if file.endswith(".png") and file.startswith("result_"):
                    st.image(os.path.join(output_folder, file), caption=file, use_column_width=True)

            st.subheader("🔗 Kümeleme Görseli")
            st.image(os.path.join(output_folder, "cluster_visualization.png"), use_column_width=True)

            with open(os.path.join(output_folder, "results.csv"), "rb") as f:
                st.download_button("📥 Metrikler (results.csv)", f, file_name="results.csv")
            with open(os.path.join(output_folder, "clustered_vessel_data.csv"), "rb") as f:
                st.download_button("📥 Küme Sonuçları (clustered.csv)", f, file_name="clustered_vessel_data.csv")

if __name__ == "__main__":
    main()
