import pandas as pd
import os
import matplotlib.pyplot as plt
from utils import mnist_image, mnist_padded_image, fft_image, mirror_image

def generate_images(df, labels, folder_name):
    """Generates and saves FFT spectrum images from a given dataset, sorted by labels."""
    print(f"Starting: {folder_name} folder")

    # Create directories for each label (0-9)
    for label in range(10):
        os.makedirs(f"{folder_name}/{label}", exist_ok=True)

    # Iterate through dataset rows
    for i, (row, label) in enumerate(zip(df.iterrows(), labels)):
        row_image = row[1]  # Extract actual pixel values
        padded_image = mnist_padded_image(row_image)  # Convert to padded 64x64 image
        half_image_fft = fft_image(padded_image[:32, :])  # Apply FFT to top 32x64 part
        mirrored_image = mirror_image(half_image_fft)  # Mirror the FFT spectrum

        # Save the image in the corresponding label folder
        plt.imsave(f"{folder_name}/{label}/{i}.png", mirrored_image, cmap='gray')

    print(f"Finished: {folder_name} folder")

def generate_train_images(df_train, df_train_labels):
    """Generates processed images for the training dataset."""
    generate_images(df_train, df_train_labels, "training")

# Load train dataset
df_train = pd.read_csv("mnist_train.csv")
df_train_labels = df_train['label']
df_train = df_train.drop('label', axis=1)

# Load test dataset
df_test = pd.read_csv("mnist_test.csv")
df_test_labels = df_test['label']
df_test = df_test.drop('label', axis=1)

# Combine datasets
df_combined = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
df_combined_labels = pd.concat([df_train_labels, df_test_labels], axis=0).reset_index(drop=True)

# Generate images
generate_train_images(df_combined, df_combined_labels)
