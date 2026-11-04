# 1. Import libraries and create data[cite: 46]
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Create simple sample data with 4 groups[cite: 46]
X, y = make_blobs(n_samples=250, centers=4, random_state=10)
print("Dataset created successfully")
print("Number of samples:", len(X))

# 2. Display the dataset[cite: 46]
plt.scatter(X[:, 0], X[:, 1], c='green')
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.title("Dataset before Applying K-Means")
plt.show()

# 3. Find the number of clusters using the Elbow Method[cite: 46, 47]
wcss = []
for k in range(1, 9):
    model = KMeans(n_clusters=k, random_state=10, n_init=15)
    model.fit(X)
    wcss.append(model.inertia_)

# Plot the Elbow graph[cite: 47]
plt.plot(range(1, 9), wcss, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# 4. Apply K-Means with 4 clusters[cite: 47, 48]
model = KMeans(n_clusters=4, random_state=10, n_init=15)
labels = model.fit_predict(X)

# Display cluster centres[cite: 48]
print("Cluster Centres:")
print(model.cluster_centers_)

# 5. Visualize the clusters[cite: 48]
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    color="red",
    marker="D",
    s=180,
    label="Cluster Centres"
)
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.title("K-Means Cluster Visualization")
plt.legend()
plt.show()

# 6. Evaluate using Silhouette Score[cite: 49]
score = silhouette_score(X, labels)
print("Silhouette Score:", round(score, 4))
