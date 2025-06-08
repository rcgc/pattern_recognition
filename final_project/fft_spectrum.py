import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mnist_image(row):
    """Converts a row of MNIST dataset into a 28x28 image array."""
    return np.array(row, dtype=np.uint8).reshape(28, 28)

def mnist_padded_image(row):
    """Converts a row of MNIST dataset into a 64x64 padded image array with the original centered."""
    image = mnist_image(row)
    padded_image = np.zeros((64, 64), dtype=np.uint8)
    start = (64 - 28) // 2
    padded_image[start:start+28, start:start+28] = image
    return padded_image

def fft_image(image):
    """Computes the absolute value of the Fast Fourier Transform of an image and returns it."""
    fft_result = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_result)
    magnitude_spectrum = np.abs(fft_shifted)
    return np.log1p(magnitude_spectrum)

def display_images(images, titles):
    """Displays multiple images in a single plot."""
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.show()

def process_and_display_row(row):
    """Processes and displays a single row as original, padded, and FFT images."""
    row_as_image = mnist_image(row)
    row_as_image_padded = mnist_padded_image(row)
    row_as_image_fft = fft_image(row_as_image_padded)

    display_images(
        [row_as_image, row_as_image_padded, row_as_image_fft],
        ["Original 28x28", "Padded 64x64", "FFT Magnitude Spectrum"]
    )

# Load train dataset
df_train = pd.read_csv("mnist_train.csv")
df_train_labels = df_train['label']
df_train = df_train.drop('label', axis=1)

sample = df_train.iloc[0]
process_and_display_row(sample)