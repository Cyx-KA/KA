import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def perform_clustering(df):
    """Basitleştirilmiş kümeleme fonksiyonu"""
    try:
        features = ['Total_Vessel_Length', 'Thin_Vessel_Length', 
                   'Thick_Vessel_Length', 'Avg_Thickness', 'Total_Branches']
        X = df[features]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        df['Cluster'] = clusters
        df['PCA1'] = X_pca[:,0]
        df['PCA2'] = X_pca[:,1]
        
        fig, ax = plt.subplots(figsize=(10,6))
        scatter = ax.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='viridis')
        
        for i, row in df.iterrows():
            ax.text(row['PCA1']+0.1, row['PCA2']+0.1, row['Image'], fontsize=8)
        
        plt.colorbar(scatter)
        ax.set_title("Damar Kümeleme Sonuçları")
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        
        return df, fig
        
    except Exception as e:
        print(f"Kümeleme hatası: {str(e)}")
        return df, None
