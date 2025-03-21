import math
import numpy as np
import matplotlib.pyplot as plt

# Load dataset from CSV file
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

# Generate pseudo-random numbers manually
def random_numbers(seed, low, high, size):
    state = seed
    results = []
    for _ in range(size):
        state = (state * 1664525 + 1013904223) % (2**32)
        results.append(low + (state % (high - low)))
    return results

# Shuffle dataset manually
def shuffle_dataset(dataset):
    indices = random_numbers(seed=42, low=0, high=len(dataset), size=len(dataset))
    shuffled_dataset = [dataset[i] for i in indices]
    dataset.clear()
    dataset.extend(shuffled_dataset)

# Split dataset into training and testing sets
def split_dataset(dataset, N):
    training_set, testing_set = dataset[:N], dataset[N:]
    return training_set, testing_set

# Separate data by class
def separate_by_class(dataset):
    separated = {}
    for row in dataset:
        label = row[-1]
        if label not in separated:
            separated[label] = []
        separated[label].append(row[:-1])
    return separated

# Calculate mean and standard deviation for each feature
def summarize_dataset(dataset):
    return [(np.mean(attribute), np.std(attribute) + 1e-6) for attribute in zip(*dataset)]

# Summarize data by class
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for label, instances in separated.items():
        summaries[label] = summarize_dataset(instances)
    return summaries

# Calculate Gaussian probability density function
def gaussian_probability(x, mean, stdev):
    exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# Calculate class probabilities
def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for label, class_summaries in summaries.items():
        probabilities[label] = 1
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            probabilities[label] *= gaussian_probability(input_vector[i], mean, stdev)
    return probabilities

# Predict class label
def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    return max(probabilities, key=probabilities.get)

# Compute confusion matrix
def confusion_matrix(dataset, summaries):
    unique_labels = sorted(set(row[-1] for row in dataset))
    matrix = {label: {l: 0 for l in unique_labels} for label in unique_labels}
    
    for row in dataset:
        actual = row[-1]
        predicted = predict(summaries, row[:-1])
        matrix[actual][predicted] += 1
    
    return matrix, unique_labels

# Compute evaluation metrics
def compute_metrics(conf_matrix):
    classes = sorted(conf_matrix.keys())
    tp, fp, fn, tn = {}, {}, {}, {}
    precision, recall, f_score, accuracy, mcc = {}, {}, {}, 0, 0
    
    total_samples = sum(sum(row.values()) for row in conf_matrix.values())
    
    for cls in classes:
        tp[cls] = conf_matrix[cls][cls]
        fp[cls] = sum(conf_matrix[row][cls] for row in classes if row != cls)
        fn[cls] = sum(conf_matrix[cls].values()) - tp[cls]
        tn[cls] = total_samples - (tp[cls] + fp[cls] + fn[cls])
        
        precision[cls] = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
        recall[cls] = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
        f_score[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls]) if (precision[cls] + recall[cls]) > 0 else 0
        
    accuracy = sum(tp.values()) / total_samples if total_samples > 0 else 0
    mcc_numerator = sum(tp[cls] * tn[cls] - fp[cls] * fn[cls] for cls in classes)
    mcc_denominator = math.sqrt(
        sum((tp[cls] + fp[cls]) * (tp[cls] + fn[cls]) * (tn[cls] + fp[cls]) * (tn[cls] + fn[cls]) for cls in classes)
    )
    
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0
    
    return accuracy, precision, recall, f_score, mcc

# Plot confusion matrix
def plot_confusion_matrix(conf_matrix, labels):
    matrix = np.array([[conf_matrix[actual][pred] for pred in labels] for actual in labels])
    
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap=plt.cm.Blues)
    plt.colorbar(cax)
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='black')
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Main Execution
filename = "iris.csv"  # Change this to the actual file path
N = 105  # Number of training instances

# Load and shuffle dataset
iris_data = load_dataset(filename)
shuffle_dataset(iris_data)

# Split dataset into training and testing sets
training_data, testing_data = split_dataset(iris_data, N)

# Train Naïve Bayes classifier
summaries = summarize_by_class(training_data)

# Compute and print confusion matrix
test_conf_matrix, labels = confusion_matrix(testing_data, summaries)

# Compute and display performance metrics
accuracy, precision, recall, f_score, mcc = compute_metrics(test_conf_matrix)
print(f"\nAccuracy: {accuracy:.2f}")
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f_score)
print(f"Matthews Correlation Coefficient: {mcc:.2f}")

# Plot confusion matrix
plot_confusion_matrix(test_conf_matrix, labels)
