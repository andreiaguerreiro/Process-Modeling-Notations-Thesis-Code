#pip install --upgrade scikit-learn threadpoolctl --user

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance
from IPython.display import display
import warnings


# loat the date etc into df
file_path = <Substitute with file path>
df = pd.read_excel(file_path)

# Use the multiple-choice question: "Based on your experience, please select the 3 process modeling notations you use most frequently"
notation_col = "Process Notations you use the most"

# Drop rows with NaN in this column
notation_data = df[[notation_col]].dropna()

# Split the notations (comma-separated) into lists
notation_data["Notations_List"] = notation_data[notation_col].apply(lambda x: [i.strip() for i in x.split(",")])

# One-hot encode the notations
mlb = MultiLabelBinarizer()
notation_encoded = mlb.fit_transform(notation_data["Notations_List"])

# Perform PCA
pca = PCA()
X_pca_full = pca.fit_transform(notation_encoded)

# Scree plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_.cumsum(), marker='o')
plt.title("Cumulative Explained Variance by Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
plt.legend()
plt.show()

# 🔍 9. Determine Optimal Number of Clusters
# Elbow Method
inertia = []
K_range = range(2, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(notation_encoded)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method - Optimal k')
plt.grid()
plt.show()

# Cluster using KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(notation_encoded)

# Reduce dimensions for visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(notation_encoded)

# Create a DataFrame with PCA component loadings
feature_names = mlb.classes_
pca_components_df = pd.DataFrame(
    pca.components_.T,
    columns=["PCA1", "PCA2"],
    index=feature_names
).round(3)

# Display the PCA components
display(pca_components_df)

# Create a DataFrame for visualization
cluster_df = pd.DataFrame({
    "PCA1": reduced[:, 0],
    "PCA2": reduced[:, 1],
    "Cluster": clusters
})

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=cluster_df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=100)
plt.title("Clusters of Respondents by Modeling Notation Usage", fontsize=16)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

# Visualizing the clusters sizes and distances in 2D
visualizer = InterclusterDistance(kmeans)
visualizer.fit(cluster_df)
visualizer.show()

# Add the cluster labels to the original notation_data
notation_data_with_clusters = notation_data.copy()
notation_data_with_clusters["Cluster"] = clusters

# Also add the original roles and experience levels for profiling
notation_data_with_clusters["Role"] = df.loc[notation_data_with_clusters.index, 
    "What best describes your current role in relation to process modeling? "]
notation_data_with_clusters["Experience"] = df.loc[notation_data_with_clusters.index, 
    "Experience (Years)"]

# Flatten the notation usage per cluster
notation_data_with_clusters["Notations"] = notation_data_with_clusters["Notations_List"].apply(lambda lst: ", ".join(lst))

# Explode the list to count individual notation frequency per cluster
exploded = notation_data_with_clusters.explode("Notations_List")

# Group by cluster and notation
notation_freq_by_cluster = exploded.groupby(["Cluster", "Notations_List"]).size().unstack(fill_value=0)

# Group by cluster and role
role_freq_by_cluster = notation_data_with_clusters.groupby(["Cluster", "Role"]).size().unstack(fill_value=0)

# Group by cluster and experience
experience_by_cluster = notation_data_with_clusters.groupby(["Cluster", "Experience"]).size().unstack(fill_value=0)

# Display results
display(notation_freq_by_cluster)
display(role_freq_by_cluster)
display(experience_by_cluster)
