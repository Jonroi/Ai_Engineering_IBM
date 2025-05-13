# K-Means Clustering Comparison: K=4 vs K=3
# ----------------------------------------------
# This script compares the results of K-means clustering using different numbers of clusters (K=4 vs K=3)
# to demonstrate how the choice of K affects the clustering pattern on the same dataset.

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
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.3, ec='k', s=80)
plt.title('Original Data Points')
plt.xticks(())
plt.yticks(())
plt.show()

# ---------------------------------------------------
# Original K=4 Clustering
# ---------------------------------------------------
# Initialize K-means model with 4 clusters
k_means_4 = KMeans(init="k-means++", n_clusters=4, n_init=12, random_state=0)

# Fit the model to the data
k_means_4.fit(X)
k_means_labels_4 = k_means_4.labels_
k_means_cluster_centers_4 = k_means_4.cluster_centers_

# Create a plot for K=4
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(1, 1, 1)

# Generate colors based on the number of clusters
colors_4 = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels_4))))

# Plot each cluster with a different color
for k, col in zip(range(len(k_means_cluster_centers_4)), colors_4):
    # Get points belonging to current cluster
    my_members = (k_means_labels_4 == k)
    
    # Get the center of current cluster
    cluster_center = k_means_cluster_centers_4[k]
    
    # Plot the data points of this cluster
    ax1.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.', ms=10)
    
    # Plot the centroid
    ax1.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)

# Set the title and remove ticks for cleaner visualization
ax1.set_title('K-means Clustering with K=4')
ax1.set_xticks(())
ax1.set_yticks(())
plt.show()

# ---------------------------------------------------
# New K=3 Clustering
# ---------------------------------------------------
# Initialize K-means model with 3 clusters
k_means_3 = KMeans(init="k-means++", n_clusters=3, n_init=12, random_state=0)

# Fit the model to the data
k_means_3.fit(X)
k_means_labels_3 = k_means_3.labels_
k_means_cluster_centers_3 = k_means_3.cluster_centers_

# Create a plot for K=3
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(1, 1, 1)

# Generate colors based on the number of clusters
colors_3 = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels_3))))

# Plot each cluster with a different color
for k, col in zip(range(len(k_means_cluster_centers_3)), colors_3):
    # Get points belonging to current cluster
    my_members = (k_means_labels_3 == k)
    
    # Get the center of current cluster
    cluster_center = k_means_cluster_centers_3[k]
    
    # Plot the data points of this cluster
    ax2.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.', ms=10)
    
    # Plot the centroid
    ax2.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)

# Set the title and remove ticks for cleaner visualization
ax2.set_title('K-means Clustering with K=3')
ax2.set_xticks(())
ax2.set_yticks(())
plt.show()

# ---------------------------------------------------
# Side-by-side comparison (K=4 vs K=3)
# ---------------------------------------------------
# Create a figure with two subplots side by side
plt.figure(figsize=(14, 6))

# K=4 subplot (left)
ax_comp1 = plt.subplot(1, 2, 1)
for k, col in zip(range(len(k_means_cluster_centers_4)), colors_4):
    my_members = (k_means_labels_4 == k)
    cluster_center = k_means_cluster_centers_4[k]
    ax_comp1.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.', ms=10)
    ax_comp1.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)
ax_comp1.set_title('K-means with K=4')
ax_comp1.set_xticks(())
ax_comp1.set_yticks(())

# K=3 subplot (right)
ax_comp2 = plt.subplot(1, 2, 2)
for k, col in zip(range(len(k_means_cluster_centers_3)), colors_3):
    my_members = (k_means_labels_3 == k)
    cluster_center = k_means_cluster_centers_3[k]
    ax_comp2.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.', ms=10)
    ax_comp2.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=10)
ax_comp2.set_title('K-means with K=3')
ax_comp2.set_xticks(())
ax_comp2.set_yticks(())

plt.suptitle('Comparison of Different K Values in K-means Clustering', fontsize=16)
plt.tight_layout()
plt.show()

# Print cluster centers for K=4
print("K=4 Cluster Centers:")
for i, center in enumerate(k_means_cluster_centers_4):
    print(f"Cluster {i+1}: {center}")

# Print cluster centers for K=3
print("\nK=3 Cluster Centers:")
for i, center in enumerate(k_means_cluster_centers_3):
    print(f"Cluster {i+1}: {center}")

# Print metrics to quantify the difference
print("\nMetrics:")
print(f"K=4 Inertia (Sum of squared distances): {k_means_4.inertia_:.2f}")
print(f"K=3 Inertia (Sum of squared distances): {k_means_3.inertia_:.2f}")
