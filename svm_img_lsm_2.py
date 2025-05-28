# Uses HOG simplified
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

DATASET_DIR = 'dataset_LSM'
CSV_FILE = 'mexican_sign_language_dataset.csv'
VALID_LABELS = [chr(c) for c in range(ord('a'), ord('z') + 1) if c not in map(ord, ['j', 'k', 'q', 'x', 'z'])]

def compute_gradients(image):
    gx = np.zeros_like(image, dtype=float)
    gy = np.zeros_like(image, dtype=float)

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    padded = np.pad(image, ((1, 1), (1, 1)), mode='edge')

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+3, j:j+3]
            gx[i, j] = np.sum(region * sobel_x)
            gy[i, j] = np.sum(region * sobel_y)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
    return magnitude, orientation

def extract_hog(image, cell_size=8, bin_size=9):
    """
    Compute simplified HOG features without block normalization.

    Parameters:
        image (np.ndarray): Grayscale image (2D array)
        cell_size (int): Size of the cell (cell_size x cell_size)
        bin_size (int): Number of orientation bins

    Returns:
        np.ndarray: HOG feature vector (flattened cell histograms)
    """
    magnitude, orientation = compute_gradients(image)
    h, w = image.shape
    n_cells_x = w // cell_size
    n_cells_y = h // cell_size
    histogram = np.zeros((n_cells_y, n_cells_x, bin_size))

    angle_unit = 180 / bin_size

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            cell_mag = magnitude[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            cell_ori = orientation[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]

            for y in range(cell_size):
                for x in range(cell_size):
                    bin_idx = int(cell_ori[y, x] // angle_unit) % bin_size
                    histogram[i, j, bin_idx] += cell_mag[y, x]

    return histogram.flatten()  # Result: 8*8*9 = 576 features


def extract_features(image_path):
    image = Image.open(image_path).convert('L').resize((64, 64))
    image_np = np.array(image, dtype=float)
    return extract_hog(image_np)

# Step 1: Load or build CSV
if os.path.exists(CSV_FILE):
    print("CSV dataset found. Loading...")
    df = pd.read_csv(CSV_FILE)
else:
    print("CSV not found. Processing images to extract features...")
    data = []
    labels = []

    for label in VALID_LABELS:
        folder = os.path.join(DATASET_DIR, label)
        count = 0
        for i in range(1, 301):  # 1 to 300
            filename = f"{label}_({i}).jpg"
            filepath = os.path.join(folder, filename)
            if os.path.exists(filepath):
                features = extract_features(filepath)
                data.append(features)
                labels.append(label)
                count += 1
            else:
                print(f"Missing file: {filepath}")
        print(f"Subfolder '{label}' processed ({count} images).")

    df = pd.DataFrame(data)
    df['label'] = labels
    df.to_csv(CSV_FILE, index=False)
    print("CSV created:", CSV_FILE)

# Step 2: Split data and train
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Step 3: Evaluation
print("\n--- Performance Metrics ---")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall   :", recall_score(y_test, y_pred, average='macro'))
print("F1 Score :", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 4: Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(y_test, y_pred, sorted(VALID_LABELS))
