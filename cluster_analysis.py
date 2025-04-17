import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def perform_clustering(df):
    """Kümeleme analizi yapar"""
    features = ['Total_Vessel_Length', 'Thin_Vessel_Length', 
               'Thick_Vessel_Length', 'Avg_Thickness', 'Total_Branches']
    
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    
    df['Cluster'] = clusters
    df['PCA1'] = X_pca[:,0
