import pandas as pd
import math
import matplotlib.pyplot as plt

# Proportion of the dataset to use for training
training_percentage = 0.8

# Class labels used in the Iris dataset
class_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Initial seed for the Linear Congruential Generator (LCG)
seed = 42


def generate_random():
    """Custom pseudo-random number generator using LCG."""
    global seed
    a, c, m = 1664525, 1013904223, 2**32
    seed = (a * seed + c) % m
    return seed / m


def load_and_prepare_data(filename):
    """
    Load the Iris dataset, normalize features, shuffle,
    and split into training and testing sets.
    """
    df = pd.read_csv(filename)
    df = df.drop(columns=['Id'])  # Drop unused ID column
    df['class'] = df['Species']
    df = df.drop(columns=['Species'])  # Replace 'Species' with 'class'

    # Map class names to integer labels
    class_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    df['class'] = df['class'].map(class_map)

    # Normalize features to [0, 1] range
    for col in df.columns[:-1]:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)

    # Shuffle the dataset manually
    indices = list(df.index)
    for i in range(len(indices) - 1, 0, -1):
        j = int(generate_random() * (i + 1))
        indices[i], indices[j] = indices[j], indices[i]
    df = df.loc[indices].reset_index(drop=True)

    # Split into training and test sets
    split_idx = int(training_percentage * len(df))
    X_train = df.iloc[:split_idx, :-1].values.tolist()
    y_train_raw = df.iloc[:split_idx, -1].tolist()
    X_test = df.iloc[split_idx:, :-1].values.tolist()
    y_test_raw = df.iloc[split_idx:, -1].tolist()

    # One-hot encode training labels
    y_train = [[1 if i == label else 0 for i in range(3)] for label in y_train_raw]
    return X_train, y_train, X_test, y_test_raw


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + math.exp(-x))


def dsigmoid(y):
    """Derivative of the sigmoid function."""
    return y * (1 - y)


def softmax(x):
    """Softmax activation for output layer."""
    exp_x = [math.exp(i) for i in x]
    sum_exp = sum(exp_x)
    return [i / sum_exp for i in exp_x]


class DNN:
    """Simple fully connected feedforward neural network."""

    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        """Initialize weights and biases."""
        self.lr = lr
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases with small random values
        self.w1 = [[generate_random() - 0.5 for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [[generate_random() - 0.5 for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [0.0] * output_size

    def forward(self, x):
        """Forward pass through the network."""
        self.z1 = [sum(x[i] * self.w1[i][j] for i in range(self.input_size)) + self.b1[j]
                   for j in range(self.hidden_size)]
        self.a1 = [sigmoid(z) for z in self.z1]

        self.z2 = [sum(self.a1[i] * self.w2[i][j] for i in range(self.hidden_size)) + self.b2[j]
                   for j in range(self.output_size)]
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, x, y):
        """Backward pass using gradient descent."""
        # Output layer error
        dL_dz2 = [self.a2[i] - y[i] for i in range(self.output_size)]
        dL_dw2 = [[dL_dz2[j] * self.a1[i] for j in range(self.output_size)] for i in range(self.hidden_size)]
        dL_db2 = dL_dz2

        # Hidden layer error
        dL_dz1 = [dsigmoid(self.a1[i]) * sum(self.w2[i][j] * dL_dz2[j] for j in range(self.output_size))
                  for i in range(self.hidden_size)]
        dL_dw1 = [[dL_dz1[j] * x[i] for j in range(self.hidden_size)] for i in range(self.input_size)]
        dL_db1 = dL_dz1

        # Gradient descent updates
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.w1[i][j] -= self.lr * dL_dw1[i][j]
        for j in range(self.hidden_size):
            self.b1[j] -= self.lr * dL_db1[j]
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                self.w2[i][j] -= self.lr * dL_dw2[i][j]
        for j in range(self.output_size):
            self.b2[j] -= self.lr * dL_db2[j]

    def train(self, X, y, epochs=200):
        """
        Train the neural network using the given dataset.
        Returns loss and accuracy history.
        """
        loss_history = []
        accuracy_history = []

        for epoch in range(epochs):
            total_loss = 0
            correct = 0

            for xi, yi in zip(X, y):
                output = self.forward(xi)
                self.backward(xi, yi)
                total_loss += sum((output[i] - yi[i]) ** 2 for i in range(len(yi))) / len(yi)
                predicted = output.index(max(output))
                actual = yi.index(1)
                if predicted == actual:
                    correct += 1

            avg_loss = total_loss / len(X)
            accuracy = correct / len(X)
            loss_history.append(avg_loss)
            accuracy_history.append(accuracy)

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

        return loss_history, accuracy_history

    def predict(self, X):
        """Predict output probabilities for input features."""
        return [self.forward(x) for x in X]


def compute_confusion_matrix(y_true, y_pred):
    """Generate a confusion matrix from true and predicted labels."""
    matrix = {actual: {pred: 0 for pred in class_labels} for actual in class_labels}
    for a, p in zip(y_true, y_pred):
        matrix[class_labels[a]][class_labels[p]] += 1
    return matrix


def calculate_metrics(conf_matrix):
    """Calculate precision, recall, specificity, F1 score, and accuracy."""
    TP, FP, FN, TN = [], [], [], []
    total = sum([sum(conf_matrix[row].values()) for row in class_labels])

    for label in class_labels:
        tp = conf_matrix[label][label]
        fp = sum(conf_matrix[row][label] for row in class_labels if row != label)
        fn = sum(conf_matrix[label][col] for col in class_labels if col != label)
        tn = total - (tp + fp + fn)
        TP.append(tp)
        FP.append(fp)
        FN.append(fn)
        TN.append(tn)

    precision = [TP[i] / (TP[i] + FP[i]) if TP[i] + FP[i] else 0 for i in range(3)]
    recall = [TP[i] / (TP[i] + FN[i]) if TP[i] + FN[i] else 0 for i in range(3)]
    specificity = [TN[i] / (TN[i] + FP[i]) if TN[i] + FP[i] else 0 for i in range(3)]
    f1 = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) if precision[i] + recall[i] else 0 for i in range(3)]
    accuracy = sum(TP) / total

    return precision, recall, specificity, f1, accuracy


def plot_confusion_matrix(conf_matrix):
    """Display the confusion matrix as a heatmap."""
    matrix = [[conf_matrix[row][col] for col in class_labels] for row in class_labels]
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap='Blues')
    plt.colorbar(cax)

    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i][j]), va='center', ha='center')

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def plot_training_curves(loss_history, accuracy_history):
    """Plot training loss and accuracy over epochs."""
    epochs = list(range(1, len(loss_history) + 1))

    plt.figure(figsize=(12, 5))

    # Plot loss over epochs
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history, label='Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)

    # Plot accuracy over epochs
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy_history, label='Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Main script
if __name__ == "__main__":
    X_train, y_train, X_test, y_test_raw = load_and_prepare_data('iris.csv')

    dnn = DNN(input_size=4, hidden_size=8, output_size=3, lr=0.1)
    loss_history, accuracy_history = dnn.train(X_train, y_train, epochs=200)

    predictions = dnn.predict(X_test)
    y_pred = [max(range(3), key=lambda i: p[i]) for p in predictions]

    conf_matrix = compute_confusion_matrix(y_test_raw, y_pred)
    precision, recall, specificity, f1, accuracy = calculate_metrics(conf_matrix)

    print(f"\nAccuracy: {accuracy:.2f}")
    for i, label in enumerate(class_labels):
        print(f"{label}: Precision={precision[i]:.2f}, Recall={recall[i]:.2f}, "
              f"Specificity={specificity[i]:.2f}, F1={f1[i]:.2f}")

    plot_confusion_matrix(conf_matrix)
    plot_training_curves(loss_history, accuracy_history)
