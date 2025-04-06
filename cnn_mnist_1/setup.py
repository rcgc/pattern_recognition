import pandas as pd
import os
import matplotlib.pyplot as plt
from utils import mnist_image, mnist_padded_image, fft_image

def generate_images(df, labels, folder_name):
    """Generates and saves FFT spectrum images from a given dataset, sorted by labels."""
    print(f"Starting: {folder_name} folder")
    
    for label in range(10):
        os.makedirs(f"{folder_name}/{label}", exist_ok=True)
    
    for i, (row, label) in enumerate(zip(df.iterrows(), labels)):
        image = mnist_image(row[1])
        padded_image = mnist_padded_image(row[1])
        fft_spectrum = fft_image(padded_image)
        
        plt.imsave(f"{folder_name}/{label}/{i}.png", fft_spectrum, cmap='gray')
    
    print(f"Finished: {folder_name} folder")

def generate_train_images(df_train, df_train_labels):
    """Generates FFT spectrum images for training dataset with correct labels."""
    generate_images(df_train, df_train_labels, "training")

def generate_test_images(df_test, df_test_labels):
    """Generates FFT spectrum images for testing dataset with correct labels."""
    generate_images(df_test, df_test_labels, "testing")

# Load train dataset
df_train = pd.read_csv("mnist_train.csv")
df_train_labels = df_train['label']
df_train = df_train.drop('label', axis=1)

# Load test dataset
df_test = pd.read_csv("mnist_test.csv")
df_test_labels = df_test['label']
df_test = df_test.drop('label', axis=1)

# Generate images
generate_train_images(df_train, df_train_labels)
generate_test_images(df_test, df_test_labels)
