import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def perform_clustering(df, n_clusters=3, random_state=42):
    """
    Damar analizi sonuçlarını kümeleme fonksiyonu
    
    Parametreler:
        df: Pandas DataFrame (damar metriklerini içeren)
        n_clusters: Oluşturulacak küme sayısı (varsayılan: 3)
        random_state: Tekrarlanabilirlik için seed değeri
    
    Döndürür:
        clustered_df: Küme bilgileri eklenmiş DataFrame
        fig: Kümeleme görseli (matplotlib Figure objesi)
    """
    try:
        # 1. Özellik seçimi
        features = ['Total_Vessel_Length', 'Thin_Vessel_Length', 
                   'Thick_Vessel_Length', 'Avg_Thickness', 'Total_Branches']
        X = df[features].copy()
        
        # 2. Veriyi ölçeklendirme
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 3. PCA ile boyut indirgeme
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # 4. K-Means kümeleme
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = kmeans.fit_predict(X_scaled)
        
        # 5. Silhouette skoru hesapla (küme kalitesi)
        silhouette_avg = silhouette_score(X_scaled, clusters)
        
        # 6. Sonuçları DataFrame'e ekle
        df['Cluster'] = clusters
        df['PCA1'] = X_pca[:, 0]
        df['PCA2'] = X_pca[:, 1]
        
        # 7. Görselleştirme
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            x='PCA1', y='PCA2',
            hue='Cluster',
            palette=sns.color_palette("hsv", n_clusters),
            data=df,
            s=100,
            alpha=0.8
        )
        
        # Küme merkezlerini işaretle
        centers = pca.transform(kmeans.cluster_centers_)
        plt.scatter(
            centers[:, 0], centers[:, 1],
            marker='X', s=200, c='black',
            label='Küme Merkezleri'
        )
        
        # Görsel detayları
        plt.title(f'Damar Kümeleme Sonuçları (Silhouette Skor: {silhouette_avg:.2f})')
        plt.xlabel('PCA Bileşen 1')
        plt.ylabel('PCA Bileşen 2')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        fig = plt.gcf()
        
        return df, fig
        
    except Exception as e:
        print(f"Kümeleme hatası: {str(e)}")
        return df, None


# TEST KODU (Bu kısmı silmeyin, modül import edildiğinde çalışmaz)
if __name__ == "__main__":
    # Test verisi oluştur
    test_data = {
        'Image': ['img1', 'img2', 'img3', 'img4', 'img5'],
        'Total_Vessel_Length': [3500, 4200, 1800, 2900, 3900],
        'Thin_Vessel_Length': [1800, 2200, 900, 1500, 2100],
        'Thick_Vessel_Length': [1700, 2000, 900, 1400, 1800],
        'Avg_Thickness': [2.1, 2.3, 1.8, 2.0, 2.2],
        'Total_Branches': [12, 15, 8, 10, 14]
    }
    test_df = pd.DataFrame(test_data)
    
    # Kümeleme yap
    clustered_df, fig = perform_clustering(test_df, n_clusters=2)
    
    # Sonuçları göster
    print(clustered_df)
    if fig:
        plt.show()
