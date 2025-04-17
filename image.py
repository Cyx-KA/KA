import streamlit as st
import os, cv2, shutil
import pandas as pd
import numpy as np
from PIL import Image
from zipfile import ZipFile
from skimage.filters import frangi, meijering, threshold_multiotsu
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import seaborn as sns

# --- Damar tespiti ---
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

# --- Ana analiz fonksiyonu ---
def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    results = []
    image_paths = []

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None: continue
        thin_mask, thick_mask, combined = enhanced_vessel_detection(img)
        distance = np.zeros_like(thin_mask)
        thickness_map = distance * np.logical_or(thin_mask, thick_mask)
        thin_length = np.sum(thin_mask)
        thick_length = np.sum(thick_mask)
        avg_thickness = np.mean(thickness_map[np.logical_or(thin_mask, thick_mask)]) * 2
        total_branches = len(np.unique(thin_mask)) + len(np.unique(thick_mask))
        results.append({
            'Görsel': filename,
            'Uzunluk': thin_length + thick_length,
            'İnce': thin_length,
            'Kalın': thick_length,
            'Kalınlık': avg_thickness,
            'Dallanma': total_branches
        })
        fig, ax = plt.subplots(2, 3, figsize=(12, 8))
        ax[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0,0].set_title('Orijinal')
        combined_vessels = np.logical_or(thin_mask, thick_mask)
        red_vessels_image = img.copy(); red_vessels_image[combined_vessels] = [0, 0, 255]
        ax[0,1].imshow(cv2.cvtColor(red_vessels_image, cv2.COLOR_BGR2RGB)); ax[0,1].set_title('Damarlar')
        ax[0,2].imshow(thickness_map, cmap='jet'); ax[0,2].set_title('Kalınlık Haritası')
        ax[1,0].imshow(thin_mask, cmap='gray'); ax[1,0].set_title('İnce Damarlar')
        ax[1,1].imshow(thick_mask, cmap='gray'); ax[1,1].set_title('Kalın Damarlar')
        ax[1,2].imshow(np.log1p(thickness_map), cmap='jet'); ax[1,2].set_title('Log Kalınlık Haritası')
        for i in range(2): [a.axis('off') for a in ax[i]]
        plt.tight_layout()
        save_path = os.path.join(output_folder, f'result_{filename}.png')
        plt.savefig(save_path, dpi=150); plt.close()
        image_paths.append((filename, save_path))
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_folder, "results.csv"), index=False)
    return df, image_paths

# --- Kümeleme fonksiyonu ---
def cluster_images(results_csv_path, output_folder):
    df = pd.read_csv(results_csv_path)
    features = ['Uzunluk', 'İnce', 'Kalın', 'Kalınlık', 'Dallanma']
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    df['Küme'] = clusters
    df['PCA1'] = X_pca[:, 0]; df['PCA2'] = X_pca[:, 1]
    plt.figure(figsize=(10,7))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Küme', palette='Set1', s=100)
    for i, row in df.iterrows():
        plt.text(row['PCA1']+0.05, row['PCA2']+0.05, row['Görsel'], fontsize=9)
    plt.title("Kümeleme: Görsel Bazlı Damar Analizi")
    plt.tight_layout()
    cluster_path = os.path.join(output_folder, "cluster_visualization.png")
    plt.savefig(cluster_path, dpi=150); plt.close()
    df.to_csv(os.path.join(output_folder, "clustered_vessel_data.csv"), index=False)
    return cluster_path

# --- Zip dosyası oluştur ---
def zip_results(folder_path, output_zip):
    shutil.make_archive(output_zip, 'zip', folder_path)
    return output_zip + ".zip"

# --- Streamlit arayüzü ---
def main():
    st.title("🔬 Damar Analizi ve Kümeleme Sistemi")
    uploaded_zip = st.file_uploader("📁 Görselleri içeren ZIP dosyasını yükleyin", type=["zip"])
    
    if uploaded_zip:
        with open("temp.zip", "wb") as f:
            f.write(uploaded_zip.getbuffer())
        shutil.unpack_archive("temp.zip", "uploaded_folder")
        st.success("✅ Görseller yüklendi.")
        
        if st.button("📊 Analiz Et"):
            with st.spinner("🔍 Görseller analiz ediliyor..."):
                progress = st.progress(0)
                output_dir = "results"
                results_df, image_paths = process_images("uploaded_folder", output_dir)
                progress.progress(70)
                cluster_img = cluster_images(os.path.join(output_dir, "results.csv"), output_dir)
                zip_path = zip_results(output_dir, "final_results")
                progress.progress(100)

            st.subheader("📂 Analiz Sonuçları")
            st.dataframe(results_df)

            st.subheader("🖼️ Görsel Sonuçlar")
            for name, path in image_paths:
                with st.expander(name):
                    st.image(path, width=300)
                    with open(path, "rb") as img_file:
                        st.download_button("Büyüt", img_file, file_name=name + ".png")

            st.subheader("🧠 Kümeleme")
            st.image(cluster_img, use_column_width=True)

            st.download_button("📦 Tüm Sonuçları İndir (ZIP)", data=open(zip_path, "rb").read(), file_name="analiz_sonuclari.zip", mime="application/zip")

if __name__ == "__main__":
    main()
