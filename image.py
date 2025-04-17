import streamlit as st
import zipfile
import os
import tempfile
import pandas as pd
from PIL import Image
import numpy as np
from vessel_analysis import process_single_image
from cluster_analysis import perform_clustering
import matplotlib.pyplot as plt
import shutil

# Sayfa ayarları
st.set_page_config(layout="wide", page_title="Damar & Kümeleme Analizi")
st.title("🔬 Damar Analizi ve Kümeleme")

# 1. ZIP Yükleme Alanı
uploaded_zip = st.file_uploader("📤 Damar görüntülerini içeren ZIP dosyası yükle", type="zip")

if uploaded_zip:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # ZIP'i çıkart
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
        
        # 2. Damar Analizi
        st.header("📊 Damar Analizi Sonuçları")
        image_files = [f for f in os.listdir(tmp_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        all_results = []
        
        cols = st.columns(4)  # 4'lü grid
        
        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(tmp_dir, img_file)
            
            # Damar analizi yap
            result_img, csv_data = process_single_image(img_path)
            
            # Sonuçları kaydet
            all_results.append({
                "Image": img_file,
                **csv_data.to_dict(orient='records')[0]
            })
            
            # Görsel ve CSV butonu
            with cols[idx % 4]:
                with st.expander(f"🔍 {img_file}", expanded=False):
                    st.image(result_img, use_column_width=True)
                    st.download_button(
                        label="📥 CSV İndir",
                        data=csv_data.to_csv(index=False),
                        file_name=f"{os.path.splitext(img_file)[0]}_results.csv",
                        mime="text/csv",
                        key=f"csv_{idx}"
                    )
        
        # 3. Kümeleme Analizi
        if all_results:
            st.header("🧩 Kümeleme Analizi")
            df = pd.DataFrame(all_results)
            
            # Kümeleme yap
            clustered_df, fig = perform_clustering(df)
            
            # Görselleştirme
            st.pyplot(fig)
            
            # CSV İndirme Butonu
            st.download_button(
                label="💾 Tüm Kümeleme Sonuçlarını İndir",
                data=clustered_df.to_csv(index=False),
                file_name="clustered_results.csv",
                mime="text/csv"
            )
