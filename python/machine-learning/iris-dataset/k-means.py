from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sse = []
for k in range(1, 10):
 km = KMeans(n_clusters=k, init='k-means++', random_state=42)
 km.fit(X_scaled)
 sse.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 10), sse, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Soma dos Erros Quadráticos (SSE)')
plt.grid(True)
plt.savefig('elbow_method.pdf', format='pdf')
plt.show()


kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
labels = kmeans.fit_predict(X_scaled)

X_clustered = pd.DataFrame(X_scaled, columns=iris.feature_names)
X_clustered['Cluster'] = labels

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
X_clustered['PCA1'] = X_pca[:, 0]
X_clustered['PCA2'] = X_pca[:, 1]
plt.figure(figsize=(8, 5))
sns.scatterplot(data=X_clustered, x='PCA1', y='PCA2', hue='Cluster', palette='Set2')
plt.title('Clusters do K-Means (com PCA)')
plt.savefig('kmeans_clusters.pdf', format='pdf')
plt.show()


score = silhouette_score(X_scaled, labels)
print(f'Silhouette Score: {score:.2f}')

X_clustered['Target'] = iris.target
print(pd.crosstab(X_clustered['Cluster'], X_clustered['Target'], rownames=['Cluster'],
colnames=['Espécie real']))