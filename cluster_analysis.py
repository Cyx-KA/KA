import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def cluster_analysis(results_file):
    # CSV dosyasını oku
    df = pd.read_csv(results_file)

    # Kullanılacak metrikleri seç
    features = ['Total_Vessel_Length', 'Thin_Vessel_Length', 'Thick_Vessel_Length',
                'Avg_Thickness', 'Total_Branches']
    X = df[features].copy()

    # Normalize et
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA ile 2 boyuta indir
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # K-Means ile kümeleme
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_pca)

    # Küme sonuçlarını dataframe'e ekle
    df['Cluster'] = clusters
    df['PCA1'] = X_pca[:,0]
    df['PCA2'] = X_pca[:,1]

    # Görselleştir
    plt.figure(figsize=(10,7))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=100)

    for i, row in df.iterrows():
        plt.text(row['PCA1']+0.05, row['PCA2']+0.05, row['Image'], fontsize=9)

    plt.title("Kümeleme: Görsel bazlı damar verileri")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.legend(title='Küme')
    plt.tight_layout()
    plt.savefig("cluster_visualization.png", dpi=150)
    plt.show()

    # Yeni CSV olarak kaydet
    df.to_csv("clustered_vessel_data.csv", index=False)
