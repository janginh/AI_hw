import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df_clustering = pd.read_csv("C:/Users/user/Desktop/수업관련/3학년 2025/1학기/인공지능/과제_장인환/Clustering/df_file.csv")
df_clustering.info()

# 텍스트 벡터화
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df_clustering['Text'])

# K-means Clustering
k = 5  # 정치 = 0, 스포츠 = 1, 기술 = 2, 엔터테인먼트 = 3, 비즈니스 = 4
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
df_clustering['Cluster_KMeans'] = kmeans.labels_

# 차원 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='rainbow', alpha=0.6)
plt.title("K-means Clustering 결과 (2D PCA)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.show()
