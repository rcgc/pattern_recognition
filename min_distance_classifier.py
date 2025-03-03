import math

# Load the dataset from a CSV file
def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        next(file)  # Skip header
        for line in file:
            parts = line.strip().split(',')
            features = list(map(float, parts[1:-1]))  # Convert features to float
            label = parts[-1]  # Class label
            dataset.append(features + [label])
    return dataset

# Improved Shuffle (Fisher-Yates without random)
def shuffle_dataset(dataset):
    n = len(dataset)
    for i in range(n - 1, 0, -1):
        j = (i * 7) % (i + 1)  # More dynamic swapping to mix data better
        dataset[i], dataset[j] = dataset[j], dataset[i]

# Split dataset into training and testing sets while maintaining class balance
def split_dataset(dataset, N):
    training_set, testing_set = [], []
    
    class_counts = {}
    for row in dataset:
        label = row[-1]
        class_counts[label] = class_counts.get(label, 0) + 1

    class_training_counts = {label: 0 for label in class_counts}

    for row in dataset:
        label = row[-1]
        if len(training_set) < N and class_training_counts[label] < (N // len(class_counts)):
            training_set.append(row)
            class_training_counts[label] += 1
        else:
            testing_set.append(row)

    return training_set, testing_set

# Compute centroids for each class
def find_centroids(dataset):
    class_data = {}
    
    for row in dataset:
        features, label = row[:-1], row[-1]
        if label not in class_data:
            class_data[label] = []
        class_data[label].append(features)

    centroids = {}
    for label, points in class_data.items():
        num_points = len(points)
        num_features = len(points[0])
        centroid = [sum(points[j][i] for j in range(num_points)) / num_points for i in range(num_features)]
        centroids[label] = centroid

    return centroids

# Calculate Euclidean distance
def euclidean_distance(v1, v2):
    return math.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(len(v1))))

# Predict the class of a given feature vector
def predict(centroids, vector):
    closest_class = None
    min_distance = float('inf')

    for label, centroid in centroids.items():
        distance = euclidean_distance(vector, centroid)
        if distance < min_distance:
            min_distance = distance
            closest_class = label

    return closest_class

# Calculate Accuracy
def calculate_accuracy(dataset, centroids):
    correct_predictions = sum(1 for row in dataset if predict(centroids, row[:-1]) == row[-1])
    return correct_predictions / len(dataset) if dataset else 0

# Calculate Precision for each class
def calculate_precision(dataset, centroids):
    class_counts = {}
    true_positives = {}
    false_positives = {}

    for row in dataset:
        label = row[-1]
        class_counts[label] = class_counts.get(label, 0) + 1
        true_positives[label] = 0
        false_positives[label] = 0

    for row in dataset:
        actual = row[-1]
        predicted = predict(centroids, row[:-1])

        if predicted == actual:
            true_positives[actual] += 1
        else:
            false_positives[predicted] = false_positives.get(predicted, 0) + 1

    precision_per_class = {}
    for label in class_counts.keys():
        tp = true_positives[label]
        fp = false_positives.get(label, 0)
        precision_per_class[label] = tp / (tp + fp) if (tp + fp) > 0 else 0

    avg_precision = sum(precision_per_class.values()) / len(precision_per_class) if precision_per_class else 0
    return avg_precision

# Main Execution
filename = "iris.csv"  # Change this to the actual file path
N = 50  # Number of training instances

# Load and shuffle dataset
iris_data = load_dataset(filename)
shuffle_dataset(iris_data)

# Split dataset into training and testing sets
training_data, testing_data = split_dataset(iris_data, N)

# Compute centroids using the training data
centroids = find_centroids(training_data)
print("Centroids:")
#print(centroids)
print(f"Iris-virginica: {centroids['Iris-virginica']}")
print(f"Iris-setosa: {centroids['Iris-setosa']}")
print(f"Iris-versicolor: {centroids['Iris-versicolor']}\n")


# Compute and display Accuracy & Precision
train_accuracy = calculate_accuracy(training_data, centroids)
test_accuracy = calculate_accuracy(testing_data, centroids)
train_precision = calculate_precision(training_data, centroids)
test_precision = calculate_precision(testing_data, centroids)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")
print(f"Training Precision: {train_precision:.2f}")
print(f"Testing Precision: {test_precision:.2f}\n")

# Test prediction
test_vector = [5.1, 3.5, 1.4, 0.2]  # Example input
print(f"Test vector: {test_vector}")
predicted_class = predict(centroids, test_vector)
print(f"Predicted class: {predicted_class}")