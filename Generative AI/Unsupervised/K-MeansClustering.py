# Unsupervised learning: algorithms learn from unabled data

# K-means Clustering example: generates synthetic data with make_blobs
# applies K-Means to find clusters

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Generate sample data
# X contains the features, y_true contains the actual labels
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60,
                       random_state=0)

# 2. Initialize the KMeans model
# We specify the number of clusters (n_clusters) we want to find
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)

# 3. Fit the model to the data and predict the clusters
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# 4. Visualize the results
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.7,
            marker='*')
plt.title('K-Means Clustering Example')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
