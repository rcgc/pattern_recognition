"""
MNIST Rotation + PCA De-rotation (3 Sample Grid)

1. Loads mnist_test.csv
2. Picks 3 random images
3. Rotates each image by a random angle
4. Uses PCA to estimate the rotation and undo it
5. Shows all steps and PCA axis visualization with white grids on images
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.ndimage import rotate
import random

# --------------------------
# Load MNIST Test Data
# --------------------------
data = pd.read_csv("mnist_test.csv")
X = data.drop("label", axis=1).values
y = data["label"].values

# --------------------------
# Pick 3 Random Samples
# --------------------------
indices = random.sample(range(len(X)), 3)
samples = X[indices]
labels = y[indices]

# --------------------------
# Function: Rotate and Recover Image with PCA
# --------------------------
def process_image(image_flat):
    original_img = image_flat.reshape(28, 28)
    angle = random.uniform(-45, 45)
    rotated_img = rotate(original_img, angle, reshape=True)

    coords = np.column_stack(np.nonzero(rotated_img > 30))
    if len(coords) < 2:
        return original_img, rotated_img, rotated_img, angle, 0, None, None, None

    coords_centered = coords - coords.mean(axis=0)
    pca = PCA(n_components=2)
    pca.fit(coords_centered)

    pc1 = pca.components_[0]
    pca_angle = np.arctan2(pc1[1], pc1[0]) * 180 / np.pi

    # 🔧 Fix: Normalize PCA angle to [-90, 90] to avoid upside-down recovery
    if pca_angle > 90:
        pca_angle -= 180
    elif pca_angle < -90:
        pca_angle += 180

    recovered_img = rotate(rotated_img, -pca_angle, reshape=True)
    return original_img, rotated_img, recovered_img, angle, pca_angle, coords, pca, coords.mean(axis=0)

# --------------------------
# Function: Draw PCA Axes on Image
# --------------------------
def draw_pca_arrows(ax, center, pca, colors=["red", "blue"], scale=20):
    for vec, color in zip(pca.components_, colors):
        ax.arrow(center[1], center[0], vec[1]*scale, vec[0]*scale,
                 color=color, head_width=2, head_length=2)

# --------------------------
# Function: Display Results in Grid (3 Samples) with White Grid Lines
# --------------------------
def show_grid(samples, labels):
    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle("MNIST (3 Samples): Original, Rotated (PCA Arrows), PCA-Recovered", fontsize=18)

    def imshow_with_grid(ax, img, title):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)

        # Set ticks at pixel boundaries
        grid_step = 4
        ax.set_xticks(np.arange(-0.5, img.shape[1], grid_step))
        ax.set_yticks(np.arange(-0.5, img.shape[0], grid_step))

        # Show grid with white lines
        ax.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.7)

        # Hide tick labels but keep ticks and grid visible
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Hide axis spines for cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)

    for row, (img_flat, label) in enumerate(zip(samples, labels)):
        img_orig, img_rot, img_rec, true_angle, rec_angle, coords, pca, center = process_image(img_flat)

        # Original
        imshow_with_grid(axs[row, 0], img_orig, f"Label: {label}\nOriginal Res ({img_orig.shape})")

        # Rotated
        imshow_with_grid(axs[row, 1], img_rot, f"Rotated ({true_angle:.2f}°)\n New Res {img_rot.shape}")
        if coords is not None and pca is not None and center is not None:
            draw_pca_arrows(axs[row, 1], center, pca)

        # Recovered
        imshow_with_grid(axs[row, 2], img_rec, f"Recovered (~{-rec_angle:.2f}°)\n New Res {img_rec.shape}")
        if coords is not None and pca is not None and center is not None:
            theta = -np.deg2rad(rec_angle)
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            rotated_components = rot_matrix @ pca.components_.T
            pca_rotated = PCA(n_components=2)
            pca_rotated.components_ = rotated_components.T
            draw_pca_arrows(axs[row, 2], center, pca_rotated)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    show_grid(samples, labels)
