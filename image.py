import streamlit as st
import os
from process_damar_analizi import process_images
from cluster_analysis import cluster_analysis

# Streamlit Arayüzü Başlangıcı
st.title("Damar Analizi ve Kümeleme")
st.markdown("Görüntülerin bulunduğu klasörü girin ve işlemi başlatın.")

# Kullanıcıdan klasör yolu alınır
input_folder = st.text_input("Girdi Klasörü:", "input_images")
output_folder = "final_vessel_analysis"

# Girdi klasörü mevcutsa işlem başlatılır
if st.button("Damar Analizini Başlat"):
    if os.path.exists(input_folder):
        st.write(f"Girdi klasörü: {input_folder} bulundu, işlem başlatılıyor...")
        
        # Damar Analizini Çalıştır
        process_images(input_folder, output_folder)
        st.success("Damar analizi tamamlandı.")
        
        # Kümeleme işlemi için CSV dosyasının var olup olmadığını kontrol et
        if os.path.exists(os.path.join(output_folder, 'results.csv')):
            st.write("Kümeleme işlemi başlatılabilir.")
            if st.button("Kümeleme Analizini Başlat"):
                cluster_analysis(os.path.join(output_folder, 'results.csv'))
                st.success("Kümeleme analizi tamamlandı.")
        else:
            st.error("Öncelikle damar analizini yapmanız gerekiyor.")
    
    else:
        st.error(f"Girdi klasörü '{input_folder}' bulunamadı. Lütfen geçerli bir yol girin.")
