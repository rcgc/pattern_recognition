import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

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

def half_image(row):
    return row[:32, :]

def fft_image(image):
    """Computes the absolute value of the Fast Fourier Transform of an image and returns it."""
    fft_result = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_result)
    magnitude_spectrum = np.abs(fft_shifted)
    return np.log1p(magnitude_spectrum)

def mirror_image(image):
    """Mirrors the given 32x64 image to create a 64x64 image."""
    mirrored = np.vstack([image, np.flipud(image)])  # Stack original and its vertical flip
    return mirrored

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

    # Keep only the top 32 rows
    half_row_as_image_padded = half_image(row_as_image_padded)
    half_row_as_image_fft = fft_image(half_row_as_image_padded)
    mirrored_row_as_image_fft = mirror_image(half_row_as_image_fft)

    display_images(
        [row_as_image, half_row_as_image_padded, mirrored_row_as_image_fft],
        ["Original 28x28", "Padded 32x64", "FFT Magnitude Spectrum (Mirrored)"]
    )

def build_model():
    """Builds and returns the CNN model."""
    model = models.Sequential([
        layers.Input(shape=(64, 64, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model