import streamlit as st
import zipfile
import os
import tempfile
import pandas as pd
from PIL import Image
from vessel_analysis import process_single_image
from cluster_analysis import perform_clustering
import matplotlib.pyplot as plt
import logging

# --- 1. HATA YÖNETİMİ VE LOGLAMA ---
logging.basicConfig(filename='app.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. SAYFA AYARLARI ---
st.set_page_config(
    layout="wide",
    page_title="Damar Analizi",
    page_icon="🔬"
)
st.title("🔬 Damar Analizi ve Kümeleme")

# --- 3. VERİ YÜKLEME ---
def extract_zip(uploaded_zip):
    """ZIP dosyasını çıkartır ve geçici dizin döndürür"""
    try:
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        return temp_dir
    except Exception as e:
        logger.error(f"ZIP çıkarma hatası: {str(e)}")
        st.error("ZIP dosyası işlenirken hata oluştu!")
        return None

# --- 4. ANA İŞLEM AKIŞI ---
uploaded_zip = st.file_uploader("📤 ZIP Yükle", type="zip")

if uploaded_zip:
    with st.spinner("Analiz yapılıyor..."):
        temp_dir = extract_zip(uploaded_zip)
        if temp_dir:
            try:
                # --- DAMAR ANALİZİ ---
                image_files = [f for f in os.listdir(temp_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.tif'))]
                
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
                
                # --- KÜMELEME ANALİZİ ---
                if results:
                    df = pd.DataFrame(results)
                    clustered_df, fig = perform_clustering(df)
                    
                    st.header("🧩 Kümeleme Sonuçları")
                    st.pyplot(fig)
                    
                    st.download_button(
                        label="💾 Tüm Sonuçları İndir",
                        data=clustered_df.to_csv(index=False),
                        file_name="kumeleme_sonuclari.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                logger.error(f"Uygulama hatası: {str(e)}")
                st.error("Analiz sırasında beklenmeyen bir hata oluştu!")
            finally:
                # Geçici dosyaları temizle
                shutil.rmtree(temp_dir, ignore_errors=True)
