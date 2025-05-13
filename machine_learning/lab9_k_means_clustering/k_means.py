import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Set random seed for reproducibility
np.random.seed(0)

# Generate synthetic data with 4 centers
# n_samples: total number of points
# centers: locations of the cluster centers
# cluster_std: standard deviation of the clusters
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

# Plot the original data points
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.3, ec='k', s=80)
plt.title('Original Data Points')
plt.xticks(())
plt.yticks(())

# Compare K=4 vs K=3 clustering
# Create a figure with two subplots side by side
plt.figure(figsize=(14, 6))

# K=4 clustering (original)
# -----------------------
# Initialize K-means model with 4 clusters
k_means_4 = KMeans(init="k-means++", n_clusters=4, n_init=12, random_state=0)

# Fit the model to the data
k_means_4.fit(X)
k_means_labels_4 = k_means_4.labels_
k_means_cluster_centers_4 = k_means_4.cluster_centers_

# Create the first subplot for K=4
ax1 = plt.subplot(1, 2, 1)

# Generate colors based on the number of clusters
colors_4 = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels_4))))

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)

    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]

    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)

    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()