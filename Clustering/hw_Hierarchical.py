import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

df_clustering = pd.read_csv("C:/Users/user/Desktop/수업관련/3학년 2025/1학기/인공지능/과제_장인환/Clustering/df_file.csv")
df_clustering.info()

# 벡터 생성 
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df_clustering['Text'])

# 거리 행렬 계산 (코사인 거리)
distance_matrix = cosine_distances(X)

# 계층적 클러스터링
linkage_matrix = linkage(distance_matrix, method='ward')

# 시각화, 일부분만 시각화
plt.figure(figsize=(15, 6))
dendrogram(linkage_matrix, truncate_mode='lastp', p=20, leaf_rotation=90., leaf_font_size=12.)
plt.title("Hierarchical Clustering")
plt.xlabel("text Clustering")
plt.ylabel("distance")
plt.show()
