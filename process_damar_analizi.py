import cv2
import numpy as np
import pandas as pd
import os

def process_images(img_path, output_folder):
    """Tek bir görüntüyü işleyip sonuçları kaydeder"""
    try:
        # Görüntüyü yükle
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Görüntü yüklenemedi: {img_path}")
        
        # Basit damar tespiti (örnek)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Sonuç görseli oluştur
        result_img = img.copy()
        result_img[binary == 255] = [0, 0, 255]  # Damarları kırmızı yap
        
        # CSV verisi
        csv_data = pd.DataFrame({
            "Image": [os.path.basename(img_path)],
            "Total_Pixels": [np.sum(binary == 255)],
            "Mean_Intensity": [np.mean(gray)]
        })
        
        # Görseli kaydet
        output_path = os.path.join(output_folder, f"processed_{os.path.basename(img_path)}")
        cv2.imwrite(output_path, result_img)
        
        return result_img, csv_data
        
    except Exception as e:
        print(f"Hata: {str(e)}")
        # Hata durumunda boş bir görsel ve veri döndür
        return np.zeros((100,100,3), pd.DataFrame({
            "Image": [os.path.basename(img_path)],
            "Total_Pixels": [0],
            "Mean_Intensity": [0]
        })
