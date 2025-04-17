import streamlit as st
import os
from process_damar_analizi import process_images
from cluster_analysis import cluster_analysis

# Başlık
st.title("Damar Analizi ve Kümeleme")
st.markdown("Lütfen giriş klasörünü belirtin ve işlemi başlatın.")

# Klasör yolu için input
input_folder = st.text_input("Girdi Klasörü", "input_images")
output_folder = "final_vessel_analysis"

# Girdi klasörünü kontrol et ve damar analizini başlat
if st.button("Damar Analizini Başlat"):
    if os.path.exists(input_folder):
        st.write(f"Girdi klasörü: {input_folder} bulundu. Damar analizi başlatılıyor...")
        
        # Damar analizi fonksiyonunu çağır
        process_images(input_folder, output_folder)
        st.success("Damar analizi tamamlandı.")
        
        # Kümeleme için CSV dosyasının var olup olmadığını kontrol et
        if os.path.exists(os.path.join(output_folder, 'results.csv')):
            st.write("Kümeleme analizi başlatılabilir.")
            if st.button("Kümeleme Analizini Başlat"):
                cluster_analysis(os.path.join(output_folder, 'results.csv'))
                st.success("Kümeleme analizi tamamlandı.")
        else:
            st.error("Önce damar analizi tamamlanmalıdır.")
    else:
        st.error(f"Girdi klasörü '{input_folder}' bulunamadı. Lütfen geçerli bir yol girin.")
