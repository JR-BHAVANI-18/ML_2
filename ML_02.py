import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data from CSV file
file_path = 'ML_02.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Ensure the data is properly loaded
print(df.head())

# Prepare data for clustering (excluding 'CustomerID')
X = df.drop('CustomerID', axis=1)

# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# From the elbow graph, choose the optimal number of clusters
optimal_clusters = 3  # This should be chosen based on the Elbow graph

# Apply K-means to the dataset
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add the cluster results to the original DataFrame
df['Cluster'] = y_kmeans

# Print the DataFrame with clusters
print(df)

# Save the DataFrame with cluster assignments to a new CSV file
output_file_path = 'ML_02.csv'  # Replace with your desired output file path
df.to_csv(output_file_path, index=False)

print(f"Clustered data saved to {output_file_path}")
