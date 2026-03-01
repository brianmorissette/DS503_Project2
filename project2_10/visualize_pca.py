import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def parse_hadoop_output(filepath):
    """
    Parses the specific tab and comma-separated output from KMeansOutput2.java.
    Expected format per line: ClusterID \t w,x,y,z \t CentroidW,CentroidX,CentroidY,CentroidZ
    """
    data = []
    if not os.path.exists(filepath):
        print(f"Error: Could not find {filepath}. Please ensure the file is in the same directory.")
        return None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    cluster_id = int(parts[0])
                    # The 2nd item is the "w,x,y,z" string
                    point_coords = [float(val) for val in parts[1].split(',')]
                    
                    # Combine cluster ID and the 4 coordinates into a single row
                    data.append([cluster_id] + point_coords)
                except ValueError:
                    # Skip any lines that fail to parse (like accidental headers)
                    continue
                    
    # Create a DataFrame for easy manipulation
    df = pd.DataFrame(data, columns=['ClusterID', 'w', 'x', 'y', 'z'])
    return df

def create_pca_scatterplot(filepath, output_image_path, title):
    print(f"Processing {filepath}...")
    df = parse_hadoop_output(filepath)
    
    if df is None or df.empty:
        print(f"Skipping {filepath} due to lack of data.")
        return

    # 1. Extract the 4D features and the Cluster labels
    features = ['w', 'x', 'y', 'z']
    X = df[features].values
    y = df['ClusterID'].values

    # 2. Apply PCA to reduce 4D to 2D
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)
    
    # Calculate how much variance the 2 components explain
    variance_ratio = pca.explained_variance_ratio_ * 100
    
    # 3. Create a DataFrame for plotting
    pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pc_df['ClusterID'] = y

    # 4. Plot the data
    plt.figure(figsize=(10, 8))
    
    # Get unique clusters and sort them so the legend is in order
    unique_clusters = sorted(pc_df['ClusterID'].unique())
    
    for cluster in unique_clusters:
        cluster_data = pc_df[pc_df['ClusterID'] == cluster]
        plt.scatter(cluster_data['PC1'], cluster_data['PC2'], 
                    label=f'Cluster {cluster}', 
                    alpha=0.6,    # Slight transparency to see overlapping points
                    s=30)         # Size of the dots
        
    # Formatting the plot
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({variance_ratio[0]:.1f}% variance explained)')
    plt.ylabel(f'Principal Component 2 ({variance_ratio[1]:.1f}% variance explained)')
    
    # Move legend outside the plot if there are many clusters
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 5. Save the plot as a high-res PNG for the report
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    print(f"Success! Saved visualization to {output_image_path}\n")
    plt.close()

if __name__ == "__main__":
    # 1. Visualize the Synthetic Random Dataset
    create_pca_scatterplot(
        filepath='vis_random.csv', 
        output_image_path='pca_vis_random.png', 
        title='PCA of Synthetic 4D Dataset (K-Means Clustering)'
    )
    
    # 2. Visualize the Real-world Kaggle Dataset
    create_pca_scatterplot(
        filepath='vis_kaggle.csv', 
        output_image_path='pca_vis_kaggle.png', 
        title='PCA of Credit Card Default Dataset (K-Means Clustering)'
    )