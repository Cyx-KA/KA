import cv2
import numpy as np
import pandas as pd
from skimage.filters import frangi, meijering, threshold_multiotsu
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import distance_transform_edt
from skan import Skeleton, summarize
import matplotlib.pyplot as plt
import os
import streamlit as st
import tempfile
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from fpdf import FPDF
from PIL import Image
import io
from datetime import datetime
import base64
import math

# 1. GÖRÜNTÜ İŞLEME FONKSİYONLARI
def load_and_preprocess_image(file):
    try:
        img = Image.open(file)
        img_array = np.array(img)
        
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        return img_array
    except Exception as e:
        st.error(f"Görüntü yükleme hatası: {str(e)}")
        return None

def enhanced_vessel_detection(image, clip_limit=4.0, min_vessel_size=10):
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(16,16))
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
        
        # Min damar boyutu filtresi (1-100 aralığında)
        thin_mask = remove_small_objects(thin_mask, min_size=min_vessel_size)
        thick_mask = remove_small_objects(thick_mask, min_size=min_vessel_size)
        
        return thin_mask, thick_mask, combined
    except Exception as e:
        st.error(f"Damar tespit hatası: {str(e)}")
        return None, None, None

def safe_skeleton_analysis(mask, min_length=100):  # Min uzunluk 100-1000 aralığında
    try:
        skeleton = skeletonize(mask)
        skeleton_obj = Skeleton(skeleton)
        stats = summarize(skeleton_obj)
        
        if hasattr(skeleton_obj, 'coordinates'):
            coords = skeleton_obj.coordinates
            paths = skeleton_obj.paths_list()
            
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
            
            stats['branch-distance'] = [b['length'] for b in branch_info]
            stats['image-coord'] = [b['coords'] for b in branch_info]
        
        return stats, skeleton
    except Exception as e:
        st.error(f"Skeleton analiz hatası: {str(e)}")
        return pd.DataFrame(), np.zeros_like(mask, dtype=bool)

# 2. KÜMELEME VE GÖRSELLEŞTİRME
def perform_clustering(results, n_clusters=3):
    try:
        if len(results) < 2:
            results['Cluster'] = 0
            return results, None
        
        features = results[['Total_Vessel_Length', 'Avg_Thickness', 'Total_Branches']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=min(n_clusters, len(results)), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        results['Cluster'] = clusters
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        results['PCA1'] = X_pca[:,0]
        results['PCA2'] = X_pca[:,1]
        
        fig1 = px.scatter(
            results, x='PCA1', y='PCA2',
            color='Cluster', size='Total_Vessel_Length',
            hover_name='Sample',
            title='PCA Kümeleme Analizi'
        )
        
        fig2 = px.bar(
            results.melt(id_vars=['Sample', 'Cluster'], 
                        value_vars=['Thin_Vessel_Length', 'Thick_Vessel_Length'],
                        var_name='Tip',
                        value_name='Uzunluk'),
            x='Sample', y='Uzunluk', color='Tip',
            facet_col='Cluster',
            title='Damar Tipi Dağılımı'
        )
        
        # Yeni grafikler
        fig3 = px.scatter(
            results,
            x='Total_Vessel_Length',
            y='Total_Branches',
            color='Avg_Thickness',
            size='Avg_Thickness',
            hover_name='Sample',
            title='Toplam Damar Uzunluğu vs Dallanma Sayısı',
            color_continuous_scale='Viridis'
        )
        
        # Damar yoğunluğu hesaplama (normalize edilmiş)
        results['Vessel_Density'] = results['Total_Vessel_Length'] / (results['Total_Vessel_Length'].max() + 1e-6)
        results['Branching_Intensity'] = results['Total_Branches'] / (results['Total_Vessel_Length'] + 1e-6)
        
        fig4 = px.scatter(
            results,
            x='Vessel_Density',
            y='Branching_Intensity',
            color='Cluster',
            size='Avg_Thickness',
            hover_name='Sample',
            title='Damar Yoğunluğu vs Dallanma Yoğunluğu',
            trendline="lowess"
        )
        
        return results, [fig1, fig2, fig3, fig4]
    except Exception as e:
        st.error(f"Kümeleme hatası: {str(e)}")
        results['Cluster'] = 0
        return results, None

# 3. PDF RAPOR FONKSİYONLARI (Aynı bırakıldı)
# PDF oluşturma fonksiyonunu güncelliyoruz
def create_pdf_report(results, analysis_images, filenames, cluster_figs=None, dpi=300):
    try:
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 12)
                self.cell(0, 10, 'Damar Analiz Raporu', 0, 1, 'C')
            
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Sayfa {self.page_no()}', 0, 0, 'C')

        pdf = PDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Kapak sayfası
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 20, 'Damar Analiz Raporu', 0, 1, 'C')
        pdf.ln(10)
        pdf.set_font('Arial', '', 14)
        pdf.cell(0, 10, f'Rapor Tarihi: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')
        pdf.ln(20)
        
        # Sonuç tablosu
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Analiz Sonuçları', 0, 1)
        pdf.ln(10)
        
        # Tablo başlıkları
        headers = ['Örnek', 'Top.Uzun.', 'İnce Uzun.', 'Kalın Uzun.', 'Ort.Kalın.', 'Dal Say.']
        col_widths = [40, 25, 25, 25, 25, 20]
        
        pdf.set_font('Arial', 'B', 10)
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 10, header, 1, 0, 'C')
        pdf.ln()
        
        # Tablo içeriği
        pdf.set_font('Arial', '', 9)
        for _, row in results.iterrows():
            pdf.cell(col_widths[0], 10, str(row['Sample'])[:15], 1)
            pdf.cell(col_widths[1], 10, f"{row['Total_Vessel_Length']:.1f}", 1, 0, 'C')
            pdf.cell(col_widths[2], 10, f"{row['Thin_Vessel_Length']:.1f}", 1, 0, 'C')
            pdf.cell(col_widths[3], 10, f"{row['Thick_Vessel_Length']:.1f}", 1, 0, 'C')
            pdf.cell(col_widths[4], 10, f"{row['Avg_Thickness']:.2f}", 1, 0, 'C')
            pdf.cell(col_widths[5], 10, str(row['Total_Branches']), 1, 0, 'C')
            pdf.ln()
        
        # Analiz görselleri
        for img_bytes, filename in zip(analysis_images, filenames):
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, f'Analiz: {filename[:20]}', 0, 1)
            pdf.ln(5)
            
            # Geçici dosya işlemleri
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name
            
            try:
                # Görsel boyutlarını ayarla
                img = Image.open(tmp_path)
                img_width = 190
                img_height = img.height * img_width / img.width
                
                if img_height > 250:  # Maksimum yükseklik kontrolü
                    img_height = 250
                    img_width = img.width * img_height / img.height
                
                pdf.image(tmp_path, x=(210-img_width)/2, y=30, w=img_width)
            finally:
                os.unlink(tmp_path)
        
        # Kümeleme grafikleri
        if cluster_figs:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Kümeleme Analizi', 0, 1)
            pdf.ln(10)
            
            for fig in cluster_figs:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    fig.write_image(tmp.name, width=800, height=500, scale=2)
                    pdf.image(tmp.name, x=10, y=pdf.get_y(), w=190)
                    pdf.ln(100)
                    os.unlink(tmp.name)
        
        return pdf
    except Exception as e:
        st.error(f"PDF oluşturma hatası: {str(e)}")
        return None
        
# 4. ANA UYGULAMA
def main():
    st.set_page_config(
        layout="wide",
        page_title="OVOBOARD Damar Analiz Sistemi",
        page_icon="🔬"
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Analiz Parametreleri")
        
        params = {
            'min_length': st.slider("Min damar uzunluğu (piksel)", 100, 1000, 500),
            'min_vessel_size': st.slider("Min damar boyutu (piksel²)", 1, 100, 10),
            'num_clusters': st.slider("Küme sayısı", 2, 5, 3),
            'clip_limit': st.slider("CLAHE clip limit", 1.0, 8.0, 4.0),
            'dpi': st.slider("PDF Çözünürlük (DPI)", 100, 600, 300)
        }
        
        st.markdown("---")
        st.markdown("**🔧 Geliştirici:** Cyx-KA")
        st.markdown("**🔄 Versiyon:** 2.4")
    
    # Ana sayfa
    st.title("🔬 OVOBOARD Damar Analiz Sistemi")
    uploaded_files = st.file_uploader(
        "📁 Görselleri Yükle (JPG, PNG, TIFF)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        images = []
        valid_files = []
        
        with st.spinner("Görseller yükleniyor..."):
            for file in uploaded_files:
                img = load_and_preprocess_image(file)
                if img is not None:
                    images.append(img)
                    valid_files.append(file.name)
        
        if not images:
            st.error("Geçerli görsel bulunamadı")
            return
        
        if st.button("🚀 Analiz Başlat", type="primary"):
            with st.spinner("Analiz yapılıyor..."):
                results = []
                analysis_images = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, img_array in enumerate(images):
                    try:
                        status_text.text(f"İşleniyor: {valid_files[idx]} ({idx+1}/{len(images)})")
                        progress_bar.progress((idx+1)/len(images))
                        
                        file_name = os.path.splitext(valid_files[idx])[0]
                        
                        thin_mask, thick_mask, combined = enhanced_vessel_detection(
                            img_array, 
                            clip_limit=params['clip_limit'],
                            min_vessel_size=params['min_vessel_size']
                        )
                        if thin_mask is None:
                            continue
                        
                        thin_stats, _ = safe_skeleton_analysis(thin_mask, params['min_length'])
                        thick_stats, _ = safe_skeleton_analysis(thick_mask, params['min_length'])
                        
                        thin_length = thin_stats['branch-distance'].sum() if len(thin_stats) > 0 else 0
                        thick_length = thick_stats['branch-distance'].sum() if len(thick_stats) > 0 else 0
                        
                        distance = distance_transform_edt(np.logical_or(thin_mask, thick_mask))
                        
                        # Yeni kalınlık haritası (mavi-yeşil tonlarında)
                        thickness_map = np.where(
                            np.logical_or(thin_mask, thick_mask),
                            distance,
                            0
                        )
                        
                        avg_thickness = np.mean(distance[np.logical_or(thin_mask, thick_mask)]) * 2
                        
                        results.append({
                            'Sample': file_name,
                            'Total_Vessel_Length': thin_length + thick_length,
                            'Thin_Vessel_Length': thin_length,
                            'Thick_Vessel_Length': thick_length,
                            'Avg_Thickness': avg_thickness,
                            'Total_Branches': len(thin_stats) + len(thick_stats)
                        })
                        
                        # 6'LI FIGÜR FORMATI
                        fig, ax = plt.subplots(2, 3, figsize=(18, 12))
                        
                        # 1. Orijinal görüntü
                        ax[0,0].imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                        ax[0,0].set_title('Orijinal Görüntü')
                        ax[0,0].axis('off')
                        
                        # 2. Damar tespiti
                        combined_vessels = np.logical_or(thin_mask, thick_mask)
                        red_vessels = img_array.copy()
                        red_vessels[combined_vessels] = [0, 0, 255]
                        ax[0,1].imshow(cv2.cvtColor(red_vessels, cv2.COLOR_BGR2RGB))
                        ax[0,1].set_title('Tespit Edilen Damarlar')
                        ax[0,1].axis('off')
                        
                        # 3. Yeni kalınlık haritası (mavi-yeşil)
                        im1 = ax[0,2].imshow(thickness_map, cmap='hot')
                        plt.colorbar(im1, ax=ax[0,2], fraction=0.046, pad=0.04)
                        ax[0,2].set_title('Damar Kalınlık Haritası')
                        ax[0,2].axis('off')
                        
                        # 4. İnce damarlar
                        ax[1,0].imshow(thin_mask, cmap='gray')
                        ax[1,0].set_title(f'İnce Damarlar ({len(thin_stats)})')
                        ax[1,0].axis('off')
                        
                        # 5. Kalın damarlar
                        ax[1,1].imshow(thick_mask, cmap='gray')
                        ax[1,1].set_title(f'Kalın Damarlar ({len(thick_stats)})')
                        ax[1,1].axis('off')
                        
                        # 6. Dallanma yoğunluğu
                        branch_map = np.zeros_like(thin_mask, dtype=float)
                        if hasattr(Skeleton(skeletonize(np.logical_or(thin_mask, thick_mask))), 'coordinates'):
                            coords = Skeleton(skeletonize(np.logical_or(thin_mask, thick_mask))).coordinates
                            for coord in coords:
                                y, x = int(coord[0]), int(coord[1])
                                if 0 <= y < branch_map.shape[0] and 0 <= x < branch_map.shape[1]:
                                    branch_map[y, x] += 1
                        
                        im2 = ax[1,2].imshow(branch_map, cmap='hot')
                        plt.colorbar(im2, ax=ax[1,2], fraction=0.046, pad=0.04)
                        ax[1,2].set_title('Dallanma Yoğunluğu')
                        ax[1,2].axis('off')
                        
                        plt.tight_layout(pad=3.0)
                        
                        # Yüksek kalitede kaydet
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight', dpi=params['dpi'])
                        buf.seek(0)
                        analysis_images.append(buf.getvalue())
                        buf.close()
                        plt.close(fig)
                        
                    except Exception as e:
                        st.error(f"{valid_files[idx]} işlenirken hata: {str(e)}")
                        continue
                
                if not results:
                    st.error("Hiçbir görsel işlenemedi")
                    return
                
                results_df = pd.DataFrame(results)
                results_df, cluster_figs = perform_clustering(results_df, params['num_clusters'])
                
                st.session_state.results = results_df
                st.session_state.analysis_images = analysis_images
                st.session_state.valid_files = valid_files
                st.session_state.cluster_figs = cluster_figs
            
            progress_bar.empty()
            status_text.empty()
            
            # Sekmeler
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Sonuçlar", "🖼️ Görseller", "📊 Grafikler", "📄 PDF Rapor"])
            
            with tab1:
                st.dataframe(
                    st.session_state.results.style.format({
                        'Total_Vessel_Length': '{:.1f}',
                        'Thin_Vessel_Length': '{:.1f}',
                        'Thick_Vessel_Length': '{:.1f}',
                        'Avg_Thickness': '{:.2f}'
                    }),
                    use_container_width=True
                )
            
            with tab2:
                st.image(st.session_state.analysis_images, 
                        caption=st.session_state.valid_files, 
                        use_container_width=True)
            
            with tab3:
                if st.session_state.cluster_figs:
                    for fig in st.session_state.cluster_figs[:2]:  # İlk iki grafik
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Yeni düzenlenmiş grafikler
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(st.session_state.cluster_figs[2], use_container_width=True)
                    
                    with col2:
                        st.plotly_chart(st.session_state.cluster_figs[3], use_container_width=True)
            
            with tab4:
                if st.button("🖨️ PDF Rapor Oluştur", key="pdf_create"):
                    with st.spinner("PDF oluşturuluyor..."):
                        pdf = create_pdf_report(
                            st.session_state.results,
                            st.session_state.analysis_images,
                            st.session_state.valid_files,
                            st.session_state.cluster_figs[:2],  # Sadece ilk iki grafik PDF'de
                            params['dpi']
                        )
                        
                        if pdf:
                            pdf_bytes = pdf.output(dest='S').encode('latin1')
                            st.session_state.pdf_bytes = pdf_bytes
                            
                            st.download_button(
                                "📥 PDF İndir",
                                data=pdf_bytes,
                                file_name=f"damar_analiz_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf"
                            )
                            
                            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                            pdf_display = f"""
                            <iframe src="data:application/pdf;base64,{base64_pdf}" 
                                    width="100%" height="800px" 
                                    style="border:1px solid #eee;">
                            </iframe>
                            """
                            st.markdown(pdf_display, unsafe_allow_html=True)
                        else:
                            st.error("PDF oluşturulamadı")

if __name__ == "__main__":
    main()
