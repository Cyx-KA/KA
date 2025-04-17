import streamlit as st
import os
from PIL import Image
from subprocess import call

# Başlık
st.title("Damar Analizi ve Kümeleme")

# Kullanıcıdan giriş klasörünü al
input_folder = st.text_input("Görüntülerin bulunduğu klasörü girin", "input_images")

# Çıkış klasörünü belirt
output_folder = "final_vessel_analysis"

# Görselleri listeleme
if os.path.exists(input_folder):
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png'))]
    st.write(f"Toplamda {len(files)} görsel mevcut.")
else:
    st.error("Girdi klasörü bulunamadı.")

# Görselleri küçük resim olarak göster
for file in files:
    img_path = os.path.join(input_folder, file)
    image = Image.open(img_path)
    st.image(image, caption=file, use_container_width=True)

# Analiz butonu
if st.button("Analizi Başlat"):
    # Damar analizi işlemini çağırma
    st.write("Damar analizi başlatılıyor...")
    call(["python", "vessel_analysis.py", input_folder, output_folder])
    
    # Kümeleme analizi işlemini çağırma
    st.write("Kümeleme analizi başlatılıyor...")
    call(["python", "cluster_analysis.py", os.path.join(output_folder, "results.csv")])

    st.success("Analiz tamamlandı!")
