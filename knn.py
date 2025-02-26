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

# Shuffle dataset
def shuffle_dataset(dataset):
    n = len(dataset)
    for i in range(n - 1, 0, -1):
        j = (i * 7) % (i + 1)
        dataset[i], dataset[j] = dataset[j], dataset[i]

# Split dataset
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

# Calculate Euclidean distance
def euclidean_distance(v1, v2):
    if len(v1) != len(v2):
        raise ValueError(f"Feature length mismatch: {len(v1)} vs {len(v2)}")
    return math.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(len(v1))))

# K-Nearest Neighbors Prediction
def knn_predict(training_set, test_vector, k=3):
    distances = [(row, euclidean_distance(row[:-1], test_vector)) for row in training_set]
    distances.sort(key=lambda x: x[1])
    
    label_counts = {}
    for i in range(k):
        label = distances[i][0][-1]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    max_count = -1
    predicted_label = None
    for label, count in label_counts.items():
        if count > max_count:
            max_count = count
            predicted_label = label
    
    return predicted_label

# Compute accuracy
def calculate_accuracy(dataset, training_set, k=3):
    correct = sum(1 for row in dataset if knn_predict(training_set, row[:-1], k) == row[-1])
    return correct / len(dataset) if dataset else 0

# Compute confusion matrix
def confusion_matrix(dataset, training_set, k=3):
    unique_labels = sorted(set(row[-1] for row in dataset))
    matrix = {label: {l: 0 for l in unique_labels} for label in unique_labels}
    
    for row in dataset:
        actual = row[-1]
        predicted = knn_predict(training_set, row[:-1], k)
        matrix[actual][predicted] += 1
    
    return matrix

# Main Execution
filename = "iris.csv"  # Change this to the actual file path
N = 105  # Number of training instances

# Load and shuffle dataset
iris_data = load_dataset(filename)
shuffle_dataset(iris_data)

# Split dataset into training and testing sets
training_data, testing_data = split_dataset(iris_data, N)

# Validate feature lengths
expected_feature_length = len(training_data[0][:-1])
for i, row in enumerate(training_data):
    if len(row[:-1]) != expected_feature_length:
        raise ValueError(f"Inconsistent feature length in dataset at row {i}: Expected {expected_feature_length}, but got {len(row[:-1])}")

# Compute accuracy
train_accuracy = calculate_accuracy(training_data, training_data, k=3)
test_accuracy = calculate_accuracy(testing_data, training_data, k=3)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")

# Compute and print confusion matrix
test_conf_matrix = confusion_matrix(testing_data, training_data, k=3)
print("Confusion Matrix:")
for actual, predictions in test_conf_matrix.items():
    print(f"{actual}: {predictions}")

# Test prediction
test_vector = [5.1, 3.5, 1.4, 0.2]  # Example input
# print(f"Expected feature length: {expected_feature_length}")
# print(f"Actual test vector length: {len(test_vector)}")

if len(test_vector) != expected_feature_length:
    raise ValueError(f"Test vector feature length mismatch: Expected {expected_feature_length}, but got {len(test_vector)}")

predicted_class = knn_predict(training_data, test_vector, k=3)
print(f"Test vector: {test_vector}")
print(f"Predicted class: {predicted_class}")
