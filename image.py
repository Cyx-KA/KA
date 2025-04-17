import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from skimage.filters import frangi, meijering, threshold_multiotsu
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from zipfile import ZipFile
import shutil
import seaborn as sns

# Gelişmiş damar tespiti
def enhanced_vessel_detection(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    l_channel = clahe.apply(lab[:,:,0])
    thin_vessels = frangi(l_channel, sigmas=range(1,4))
    thick_vessels = frangi(l_channel, sigmas=range(3,8))
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

# Görselleri işle
def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    results = []
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            continue
        img = cv2.imread(os.path.join(input_folder, filename))
        if img is None:
            continue
        thin_mask, thick_mask, combined = enhanced_vessel_detection(img)
        thickness_map = np.zeros_like(combined)
        thin_length = np.sum(thin_mask)
        thick_length = np.sum(thick_mask)
        avg_thickness = np.mean(thickness_map[np.logical_or(thin_mask, thick_mask)]) * 2
        total_branches = len(np.unique(thin_mask)) + len(np.unique(thick_mask))
        results.append({
            'Görsel': filename,
            'Uzunluk': int(thin_length + thick_length),
            'İnce': int(thin_length),
            'Kalın': int(thick_length),
            'Kalınlık': float(avg_thickness),
            'Dallanma': int(total_branches)
        })
        fig, ax = plt.subplots(2, 3, figsize=(20, 12))
        ax[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0,0].set_title('Orijinal Görüntü')
        combined_vessels = np.logical_or(thin_mask, thick_mask)
        red_img = img.copy()
        red_img[combined_vessels] = [0, 0, 255]
        ax[0,1].imshow(cv2.cvtColor(red_img, cv2.COLOR_BGR2RGB))
        ax[0,1].set_title('Kırmızı İşaretli Damarlar')
        ax[0,2].imshow(thickness_map, cmap='jet')
        ax[0,2].set_title('Kalınlık Haritası')
        ax[1,0].imshow(thin_mask, cmap='gray')
        ax[1,0].set_title('İnce Damarlar')
        ax[1,1].imshow(thick_mask, cmap='gray')
        ax[1,1].set_title('Kalın Damarlar')
        ax[1,2].imshow(np.log1p(thickness_map), cmap='jet')
        ax[1,2].set_title('Log Kalınlık Haritası')
        for i in range(2):
            for j in range(3):
                ax[i,j].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'result_{filename}.png'), dpi=150)
        plt.close()
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_folder, 'results.csv'), index=False)
    return df

# Kümeleme fonksiyonu
def cluster_analysis(output_folder):
    df = pd.read_csv(os.path.join(output_folder, 'results.csv'))
    X = df[['Uzunluk', 'İnce', 'Kalın', 'Kalınlık', 'Dallanma']]
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=2, random_state=42).fit(X_pca)
    df['Küme'] = kmeans.labels_
    df['PCA1'] = X_pca[:,0]
    df['PCA2'] = X_pca[:,1]
    plt.figure(figsize=(10,7))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Küme', palette='Set1', s=100)
    for i, row in df.iterrows():
        plt.text(row['PCA1']+0.05, row['PCA2']+0.05, row['Görsel'], fontsize=9)
    plt.title("Kümeleme: Görsel bazlı damar verileri")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "cluster.png"), dpi=150)
    df.to_csv(os.path.join(output_folder, "clustered_vessel_data.csv"), index=False)

# Streamlit arayüzü
def main():
    st.title("🧠 Damar Analizi ve Kümeleme Aracı")
    uploaded_file = st.file_uploader("📁 Lütfen görsellerinizi zip olarak yükleyin", type=["zip"])
    if uploaded_file:
        with open("uploaded.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())
        shutil.unpack_archive("uploaded.zip", "input_images")
        if st.button("🚀 Analiz Et"):
            with st.spinner("Görseller analiz ediliyor..."):
                df = process_images("input_images", "results")
                cluster_analysis("results")
            st.success("✅ Analiz tamamlandı!")
            st.subheader("📊 Sonuçlar")
            st.dataframe(df)

            st.subheader("📸 Analiz Görselleri")
            cols = st.columns(4)
            i = 0
            for file in sorted(os.listdir("results")):
                if file.startswith("result_") and file.endswith(".png"):
                    path = os.path.join("results", file)
                    with cols[i % 4]:
                        st.markdown(f'<a href="{path}" target="_blank"><img src="{path}" width="180"/></a>', unsafe_allow_html=True)
                    i += 1

            st.subheader("📌 Kümeleme Grafiği")
            st.image("results/cluster.png", use_column_width=True)

            shutil.make_archive("results", 'zip', "results")
            with open("results.zip", "rb") as f:
                st.download_button("📥 Sonuçları indir", f, file_name="damar_analizi_sonuclar.zip")

if __name__ == "__main__":
    main()
