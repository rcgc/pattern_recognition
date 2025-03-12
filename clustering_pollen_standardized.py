import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Load Pollen dataset
df_original = pd.read_csv('pollen.csv', header=None)

# Separate features and labels
df_classless = df_original.drop(33, axis=1)  # Remove class column
true_labels = df_original[33].tolist()  # Extract true labels
data = df_classless.values.tolist()  # Convert feature dataframe to list of lists

# Standardize features (mean = 0, std = 1)
def standardize_data(data):
    mean_vals = data.mean()
    std_vals = data.std()
    return (data - mean_vals) / std_vals

df_classless = df_classless.apply(standardize_data, axis=0)
data = df_classless.values.tolist()

# K-Means functions
class SimpleRandom:
    def __init__(self, seed=42):
        self.modulus = 2**31
        self.multiplier = 1103515245
        self.increment = 12345
        self.state = seed

    def random(self):
        self.state = (self.multiplier * self.state + self.increment) % self.modulus
        return self.state / self.modulus

    def randint(self, a, b):
        return a + int(self.random() * (b - a + 1))

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def initialize_centroids(data, k, seed=42):
    rand_gen = SimpleRandom(seed)
    indices = set()
    while len(indices) < k:
        indices.add(rand_gen.randint(0, len(data) - 1))
    return [data[i] for i in indices]

def assign_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    cluster_labels = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)
        cluster_labels.append(cluster_index)
    return clusters, cluster_labels

def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if cluster:
            new_centroids.append([sum(dim) / len(cluster) for dim in zip(*cluster)])
        else:
            new_centroids.append(cluster)  # Avoid empty clusters
    return new_centroids

def k_means(data, k, max_iters=100, seed=42):
    centroids = initialize_centroids(data, k, seed)
    for i in range(max_iters):
        clusters, cluster_labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(clusters)
        if new_centroids == centroids:
            print("Convergence reached at iteration:", i+1)
            break
        centroids = new_centroids
    return centroids, clusters, cluster_labels

def map_clusters_to_labels(cluster_labels, true_labels):
    unique_labels = list(set(true_labels))
    label_map = {}
    for cluster_idx in set(cluster_labels):
        true_labels_in_cluster = [true_labels[i] for i in range(len(cluster_labels)) if cluster_labels[i] == cluster_idx]
        if true_labels_in_cluster:
            most_common_label = max(set(true_labels_in_cluster), key=true_labels_in_cluster.count)
            label_map[cluster_idx] = most_common_label
    predicted_labels = [label_map[cluster] for cluster in cluster_labels]
    
    # Map the class labels to a continuous range of integers starting from 0
    label_to_int = {label: idx for idx, label in enumerate(sorted(set(true_labels)))}
    predicted_labels_int = [label_to_int[label] for label in predicted_labels]
    true_labels_int = [label_to_int[label] for label in true_labels]
    
    return predicted_labels_int, true_labels_int, label_to_int

def calculate_confusion_matrix(true_labels, predicted_labels, num_classes):
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(true_labels, predicted_labels):
        cm[t][p] += 1
    return cm

def plot_confusion_matrix(true_labels, predicted_labels, classes):
    num_classes = len(classes)
    cm = calculate_confusion_matrix(true_labels, predicted_labels, num_classes)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show(block=False)
    return cm

# Manual Metric Functions

def calculate_accuracy(true_labels, predicted_labels):
    correct = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
    total = len(true_labels)
    return correct / total

def calculate_precision(true_labels, predicted_labels, label):
    tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p == label)
    fp = sum(1 for t, p in zip(true_labels, predicted_labels) if t != label and p == label)
    return tp / (tp + fp) if tp + fp > 0 else 0

def calculate_recall(true_labels, predicted_labels, label):
    tp = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p == label)
    fn = sum(1 for t, p in zip(true_labels, predicted_labels) if t == label and p != label)
    return tp / (tp + fn) if tp + fn > 0 else 0

def calculate_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

def calculate_mcc(true_labels, predicted_labels, num_classes):
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(true_labels, predicted_labels):
        confusion_matrix[t][p] += 1
    
    mcc = 0
    for i in range(num_classes):
        for j in range(num_classes):
            tp = confusion_matrix[i][i]
            tn = sum(confusion_matrix[x][y] for x in range(num_classes) for y in range(num_classes) if x != i and y != j)
            fp = sum(confusion_matrix[i][y] for y in range(num_classes) if y != i)
            fn = sum(confusion_matrix[x][j] for x in range(num_classes) if x != j)
            numerator = tp * tn - fp * fn
            denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = numerator / denominator if denominator > 0 else 0
    return mcc

def print_metrics_manually(true_labels, predicted_labels, num_classes):
    accuracy = calculate_accuracy(true_labels, predicted_labels)
    
    print("Derived Metrics (manual):")
    print(f"Accuracy: {accuracy:.4f}")
    
    for label in range(num_classes):
        precision = calculate_precision(true_labels, predicted_labels, label)
        recall = calculate_recall(true_labels, predicted_labels, label)
        f1 = calculate_f1(precision, recall)
        print(f"Class {label+1} Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    
    mcc = calculate_mcc(true_labels, predicted_labels, num_classes)
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")

# Run K-Means on standardized pollen.csv
k = 12  # Adjust clusters as needed
final_centroids, final_clusters, cluster_labels = k_means(data, k)

# Map clusters to labels and plot confusion matrix
predicted_labels, true_labels_int, label_to_int = map_clusters_to_labels(cluster_labels, true_labels)
unique_classes = sorted(label_to_int.keys())
cm = plot_confusion_matrix(true_labels_int, predicted_labels, unique_classes)

# Print derived metrics manually
num_classes = len(unique_classes)
print_metrics_manually(true_labels_int, predicted_labels, num_classes)

# Keep figures open
plt.pause(10)  # Keep plots open for 10 seconds before user closes them manually
input("Press Enter to exit...")
