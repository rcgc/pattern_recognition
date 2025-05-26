# eigensigns.py (updated with dual-space PCA logic)

"""
Eigensigns Classification Script

This script performs:
1. Load images from "dataset_LSM"
2. Split into train/test (80/20)
3. Apply PCA to retain 80% variance
4. Train SVM classifier on PCA-reduced data
5. Display confusion matrix and metrics
6. Visualize top eigenvectors ("eigensigns")
7. Export PCA model and classifier using joblib
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import joblib

# Constants
DATASET_DIR = 'dataset_LSM'
EXPECTED_IMAGE_SIZE = (200, 200)
EXPORT_PCA_PATH = 'pca_model.joblib'
EXPORT_CLASSIFIER_PATH = 'svm_classifier.joblib'


def load_images_from_folders(base_dir):
    images = []
    labels = []
    label_names = sorted(os.listdir(base_dir))

    for label in label_names:
        folder = os.path.join(base_dir, label)
        if not os.path.isdir(folder):
            continue
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(folder, filename)
                image = Image.open(path).convert('L')
                if image.size != EXPECTED_IMAGE_SIZE:
                    raise ValueError(f"Image {path} is not {EXPECTED_IMAGE_SIZE}. Found size: {image.size}")
                images.append(np.array(image).flatten())
                labels.append(label)

    return np.array(images), np.array(labels)


def dual_pca(X, variance_threshold=0.80):
    print("Centering data and computing dual covariance matrix...")
    X_centered = X - np.mean(X, axis=0)
    cov_dual = X_centered @ X_centered.T / X.shape[0]  # Shape: (n_samples, n_samples)

    print("Performing eigen decomposition...")
    eigenvalues, eigenvectors = np.linalg.eigh(cov_dual)  # Ensure sorted ascending
    idx = np.argsort(eigenvalues)[::-1]  # Descending order
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    k = np.searchsorted(cumulative_variance, variance_threshold) + 1

    print(f"Retaining {k}/{X.shape[0]} components to preserve {cumulative_variance[k - 1] * 100:.2f}% of variance")

    # Project back to feature space
    U = (X_centered.T @ eigenvectors[:, :k]) / np.sqrt(eigenvalues[:k])
    U = U.T  # Shape: (k, n_features)

    X_transformed = X_centered @ U.T  # Project samples to reduced space
    return X_transformed, U, eigenvalues[:k], explained_variance_ratio[:k]


def display_eigensigns(components, variances, image_shape, explained_variance_percent, num_vectors=16):
    grid_size = int(np.ceil(np.sqrt(num_vectors)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle(f'Top {num_vectors} EigenSigns')

    for i, ax in enumerate(axes.flat):
        if i < len(components):
            ax.imshow(components[i].reshape(image_shape), cmap='gray')
            ax.set_title(f'#{i + 1} ({variances[i]:.2f}%)', fontsize=9)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def main():
    print("Loading dataset...")
    X, y = load_images_from_folders(DATASET_DIR)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Splitting dataset (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("Applying dual PCA...")
    X_train_pca, components, eigenvalues, explained_ratios = dual_pca(X_train, variance_threshold=0.80)
    explained_variance_percent = np.sum(explained_ratios) * 100
    X_test_centered = X_test - np.mean(X_train, axis=0)
    X_test_pca = X_test_centered @ components.T

    print(f"Reduced to {components.shape[0]} components to retain {explained_variance_percent:.2f}% of variance.")

    print("Visualizing eigensigns...")
    display_eigensigns(components, explained_ratios * 100, EXPECTED_IMAGE_SIZE, explained_variance_percent, num_vectors=16)

    print("Training classifier...")
    clf = SVC(kernel='rbf', class_weight='balanced')
    clf.fit(X_train_pca, y_train)

    print("Evaluating model...")
    y_pred = clf.predict(X_test_pca)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    print("\nClassification Report:")
    print(report)

    print("Exporting models...")
    joblib.dump({'components': components, 'mean': np.mean(X_train, axis=0)}, EXPORT_PCA_PATH)
    joblib.dump(clf, EXPORT_CLASSIFIER_PATH)
    print("Models saved.")


if __name__ == '__main__':
    main()
