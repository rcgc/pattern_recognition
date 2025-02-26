import csv
import math
import matplotlib.pyplot as plt

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
    """Loads the Iris dataset from a CSV file, assuming the last column is the label."""
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        for row in reader:
            data.append([float(value) for value in row[:-1]])  # Ignore label
    return data

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
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)
    return clusters

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
    data = load_iris_data(filename)
    centroids = initialize_centroids(data, k, seed)

    for i in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(clusters)

        if new_centroids == centroids:
            print("Convergence reached in iteration:", i+1)
            break  # Convergence reached

        centroids = new_centroids

    return centroids, clusters

def plot_clusters(clusters, centroids):
    """Plots clusters using Matplotlib with distinct colors and black 'X' markers for centroids."""
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Default Matplotlib colors

    for i, cluster in enumerate(clusters):
        cluster_points = list(zip(*cluster))  # Transpose to get x and y coordinates
        plt.scatter(cluster_points[0], cluster_points[1], color=colors[i % len(colors)], label=f'Cluster {i+1}')

    centroid_points = list(zip(*centroids))  # Transpose centroids
    plt.scatter(centroid_points[0], centroid_points[1], color='black', marker='x', s=100, label='Centroids')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('K-Means Clustering of Iris Dataset')
    plt.show()

# Run the K-Means algorithm
final_centroids, final_clusters = k_means("Iris.csv", k=3, seed=42)
print("Final centroids:", final_centroids)

# Plot the clusters
plot_clusters(final_clusters, final_centroids)
