import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import logging

# --- 1. OPTİMAL KÜME SAYISI BULMA ---
def find_optimal_clusters(data, max_k=5):
    """Elbow Method ile optimal küme sayısını bulur"""
    inertias = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    return inertias

# --- 2. GELİŞMİŞ KÜMELEME ---
def perform_clustering(df, auto_cluster=True):
    """
    Gelişmiş kümeleme analizi
    
    Parametreler:
        df: Damar metriklerini içeren DataFrame
        auto_cluster: Otomatik küme sayısı belirleme
    """
    try:
        # --- VERİ HAZIRLAMA ---
        features = ['Total_Vessel_Length', 'Thin_Vessel_Length', 
                   'Thick_Vessel_Length', 'Avg_Thickness', 'Total_Branches']
        X = df[features]
        
        # --- STANDARDİZASYON ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # --- OTOMATİK KÜME BELİRLEME ---
        if auto_cluster:
            inertias = find_optimal_clusters(X_scaled)
            optimal_k = np.argmin(np.diff(inertias)) + 1  # Elbow point
            n_clusters = max(2, min(optimal_k, 5))  # 2-5 arasında sınırla
        else:
            n_clusters = 3
        
        # --- PCA ve KÜMELEME ---
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # --- SONUÇLARI DF'E EKLE ---
        df['Cluster'] = clusters
        df['PCA1'] = X_pca[:,0]
        df['PCA2'] = X_pca[:,1]
        
        # --- İNTERAKTİF GRAFİK ---
        fig = px.scatter(
            df, x='PCA1', y='PCA2', color='Cluster',
            hover_data=['Image'], 
            title=f'Damar Kümeleme (Optimal K={n_clusters})',
            width=800, height=600
        )
        
        return df, fig
        
    except Exception as e:
        logging.error(f"Kümeleme hatası: {str(e)}")
        return df, None
