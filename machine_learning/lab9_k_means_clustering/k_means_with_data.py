import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Load customer data
cust_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")

# Display data info to see missing values
print("Dataset info:")
print(f"Shape: {cust_df.shape}")
print(f"Missing values per column:\n{cust_df.isna().sum()}")

# Drop non-numeric column
cust_df = cust_df.drop('Address', axis=1)

# Extract features (all columns except CustomerID)
X = cust_df.values[:,1:] 

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Standardize the data
X_scaled = StandardScaler().fit_transform(X_imputed)

clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X_scaled)  # Using the standardized data
labels = k_means.labels_

# First print column names to verify the structure
print("\nColumn names in original dataframe:")
for idx, col_name in enumerate(cust_df.columns[1:]):  # Skip CustomerID
    print(f"Index {idx}: {col_name}")

# Display 2D scatter plot of Age vs Income with Education (Edu) as point size
plt.figure(figsize=(10, 6))
# Age is at index 0, Edu at index 1, Income at index 3 in the data
area = np.pi * (X_scaled[:, 1] + 2.5)**2  # Edu (index 1) for point size with offset
plt.scatter(X_scaled[:, 0], X_scaled[:, 3], s=area, c=labels.astype(float), cmap='tab10', ec='k', alpha=0.6)
plt.colorbar(label='Cluster')
plt.xlabel('Age (standardized)', fontsize=14)
plt.ylabel('Income (standardized)', fontsize=14)
plt.title('Customer Segmentation with K-Means (k=3)', fontsize=16)
plt.grid(alpha=0.3)
plt.show()

# Create interactive 3D scatter plot
# Create a DataFrame with scaled values for plotting
df_scaled = pd.DataFrame(X_scaled, columns=cust_df.columns[1:])
df_scaled['Cluster'] = labels

# Print column names to help with debugging
print("\nAvailable columns for plotting:", df_scaled.columns.tolist())

# Create interactive 3D scatter plot using correct column names
fig = px.scatter_3d(
    df_scaled, 
    x='Edu',  # Correct column name for Education
    y='Age', 
    z='Income', 
    opacity=0.7, 
    color='Cluster'
)

fig.update_traces(marker=dict(size=5, line=dict(width=.25)), showlegend=False)
fig.update_layout(
    coloraxis_showscale=True,  # Show color scale
    width=1000, 
    height=800,
    title="3D Customer Segmentation by Age, Education, and Income",
    scene=dict(
        xaxis_title='Education Level',
        yaxis_title='Age',
        zaxis_title='Income'
    )
)

# Save plot as HTML file for better viewing
html_file = "c:/Users/jonir/Documents/GitHub/Ai_Engineering_IBM/machine_learning/lab9_k_means_clustering/customer_segmentation_3d.html"
fig.write_html(html_file)
print(f"\nInteractive 3D plot saved to: {html_file}")

# Show the plot
fig.show()

