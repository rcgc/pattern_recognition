import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

class SimpleRandom:
    """A simple linear congruential generator (LCG) for pseudo-random numbers."""
    def __init__(self, seed=42):
        self.modulus = 2**31
        self.multiplier = 1103515245
        self.increment = 12345
        self.state = seed

    def random(self):
        """Generates a pseudo-random float in the range [0,1)."""
        self.state = (self.multiplier * self.state + self.increment) % self.modulus
        return self.state / self.modulus

    def randint(self, a, b):
        """Generates a pseudo-random integer in the range [a, b]."""
        return a + int(self.random() * (b - a + 1))

def load_iris_data(filename):
    """Loads the Iris dataset from a CSV file and returns feature data and labels."""
    data, labels = [], []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        for row in reader:
            data.append([float(value) for value in row[1:-1]])  # Feature values
            labels.append(row[-1])  # Class labels
    return data, labels

def euclidean_distance(p1, p2):
    """Computes the Euclidean distance between two points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def initialize_centroids(data, k, seed=42):
    """Selects k random points from the dataset as initial centroids using SimpleRandom."""
    rand_gen = SimpleRandom(seed)
    indices = []
    while len(indices) < k:
        idx = rand_gen.randint(0, len(data) - 1)
        if idx not in indices:
            indices.append(idx)
    return [data[i] for i in indices]

def assign_clusters(data, centroids):
    """Assigns each data point to the nearest centroid."""
    cluster_labels = []
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)
        cluster_labels.append(cluster_index)
    return clusters, cluster_labels

def update_centroids(clusters):
    """Computes new centroids as the mean of points in each cluster."""
    new_centroids = []
    for cluster in clusters:
        if cluster:
            new_centroids.append([sum(dim) / len(cluster) for dim in zip(*cluster)])
        else:
            new_centroids.append(cluster)  # Avoid empty clusters
    return new_centroids

def k_means(filename, k, max_iters=100, seed=42):
    """Performs K-Means clustering on the Iris dataset."""
    data, labels = load_iris_data(filename)
    centroids = initialize_centroids(data, k, seed)

    for i in range(max_iters):
        clusters, cluster_labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(clusters)

        if new_centroids == centroids:
            print("Convergence reached in iteration:", i+1)
            break  # Convergence reached

        centroids = new_centroids

    return centroids, clusters, cluster_labels, labels

def map_clusters_to_labels(cluster_labels, true_labels):
    """Maps K-Means cluster indices to actual Iris class labels using majority voting."""
    unique_labels = list(set(true_labels))
    label_map = {}

    for cluster_idx in set(cluster_labels):
        true_labels_in_cluster = [true_labels[i] for i in range(len(cluster_labels)) if cluster_labels[i] == cluster_idx]
        
        if true_labels_in_cluster:
            most_common_label = max(set(true_labels_in_cluster), key=true_labels_in_cluster.count)
            label_map[cluster_idx] = most_common_label

    predicted_labels = [label_map[cluster] for cluster in cluster_labels]
    return predicted_labels, unique_labels

def plot_confusion_matrix(true_labels, predicted_labels, classes):
    """Plots a confusion matrix in a separate figure."""
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show(block=False)  # Show non-blocking figure

def plot_clusters(clusters, centroids):
    """Plots clusters in a separate figure with their respective centroids."""
    plt.figure(figsize=(8, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, cluster in enumerate(clusters):
        cluster_points = list(zip(*cluster))
        plt.scatter(cluster_points[0], cluster_points[1], color=colors[i % len(colors)], label=f'Cluster {i+1}', alpha=0.6)

    centroid_points = list(zip(*centroids))
    plt.scatter(centroid_points[0], centroid_points[1], color='black', marker='x', s=100, label='Centroids')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('K-Means Clustering with Centroids')
    plt.show(block=False)  # Show non-blocking figure

# Run the K-Means algorithm
final_centroids, final_clusters, cluster_labels, true_labels = k_means("Iris.csv", k=3, seed=42)

# Display Centroids in console
print("Final centroids:", final_centroids)

# Display both plots simultaneously
plot_clusters(final_clusters, final_centroids)  # Show clusters
plot_confusion_matrix(true_labels, *map_clusters_to_labels(cluster_labels, true_labels))  # Show confusion matrix

# Keep figures open
plt.pause(10)  # Keep plots open for 10 seconds before user closes them manually
input("Press Enter to exit...")
