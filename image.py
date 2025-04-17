import streamlit as st
import zipfile
import os
import tempfile
import pandas as pd
from vessel_analysis import process_single_image
from cluster_analysis import perform_clustering
import matplotlib.pyplot as plt
import shutil

# Streamlit cache problemi çözümü
if not hasattr(st, 'already_started'):
    st.already_started = True
    st.set_page_config(
        layout="wide",
        page_title="Damar Analizi",
        page_icon="🔬"
    )

st.title("🔬 Damar Analizi ve Kümeleme")

def safe_extract(uploaded_zip):
    """Güvenli ZIP çıkarma"""
    try:
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        return temp_dir
    except Exception as e:
        st.error(f"ZIP açma hatası: {str(e)}")
        return None

uploaded_zip = st.file_uploader("📤 ZIP Yükle", type="zip")

if uploaded_zip:
    temp_dir = safe_extract(uploaded_zip)
    if temp_dir:
        try:
            image_files = [f for f in os.listdir(temp_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.tif'))]
            
            if not image_files:
                st.warning("ZIP içinde uygun görsel bulunamadı!")
            else:
                results = []
                cols = st.columns(4)
                
                for idx, img_file in enumerate(image_files):
                    img_path = os.path.join(temp_dir, img_file)
                    result_img, csv_data = process_single_image(img_path)
                    
                    with cols[idx % 4]:
                        with st.expander(f"📷 {img_file}", expanded=False):
                            st.image(result_img, use_column_width=True)
                            st.download_button(
                                label="📊 CSV İndir",
                                data=csv_data.to_csv(index=False),
                                file_name=f"{os.path.splitext(img_file)[0]}.csv",
                                mime="text/csv"
                            )
                    results.append(csv_data.iloc[0].to_dict())
                
                if results:
                    df = pd.DataFrame(results)
                    clustered_df, fig = perform_clustering(df)
                    
                    st.header("🧩 Kümeleme Sonuçları")
                    st.pyplot(fig)
                    
                    st.download_button(
                        label="💾 Tüm Sonuçları İndir",
                        data=clustered_df.to_csv(index=False),
                        file_name="damar_kumeleme.csv",
                        mime="text/csv"
                    )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
