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

# K=3 clustering
# Create a figure for the K-means plot
plt.figure(figsize=(10, 6))

# Initialize K-means model with 3 clusters
k_means_3 = KMeans(init="k-means++", n_clusters=3, n_init=12, random_state=0)

# Fit the model to the data
k_means_3.fit(X)
k_means_labels_3 = k_means_3.labels_
k_means_cluster_centers_3 = k_means_3.cluster_centers_

# Create the plot
ax1 = plt.subplot(1, 1, 1)

# Generate colors based on the number of clusters
colors_3 = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels_3))))

# For loop that plots the data points and centroids.
# k will range from 0-2, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len(set(k_means_labels_3))), colors_3):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels_3 == k)

    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers_3[k]

    # Plots the datapoints with color col.
    ax1.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)

    # Plots the centroids with specified color, but with a darker outline
    ax1.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax1.set_title('KMeans with 3 Clusters')

# Remove x-axis ticks
ax1.set_xticks(())

# Remove y-axis ticks
ax1.set_yticks(())

# Show the plot
plt.show()