import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def mnist_image(row):
    """Converts a row of MNIST dataset into a 28x28 image array."""
    return np.array(row, dtype=np.float32).reshape(28, 28)

def mnist_padded_image(row, size=400):
    """Pads the MNIST image to a larger square canvas, centered."""
    image = mnist_image(row)
    padded = np.zeros((size, size), dtype=np.float32)
    start = (size - 28) // 2
    padded[start:start+28, start:start+28] = image
    return padded

import numpy as np

def fresnel_diffraction_angular_spectrum(u0, wavelength=0.01, z=40, dx=10):
    """
    Simulates Fresnel diffraction using the Angular Spectrum of Plane Waves method.
    
    u0: 2D numpy array (input wave amplitude, grayscale image)
    wavelength: wavelength in same units as dx (e.g., mm)
    z: propagation distance (same unit as dx)
    dx: pixel pitch (scalar, same unit as wavelength)
    """
    N, M = u0.shape
    k = 2 * np.pi / wavelength

    # Frequency coordinates
    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(M, d=dx)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')

    # Compute the transfer function (angular spectrum propagator)
    H = np.exp(1j * 2 * np.pi * z * np.sqrt((1 / wavelength)**2 - FX**2 - FY**2))

    # Forward FFT of the input field
    U0 = np.fft.fft2(u0)
    U0_shifted = np.fft.fftshift(U0)

    # Apply the transfer function
    U1 = U0_shifted * H

    # Inverse FFT to get propagated field
    u1 = np.fft.ifft2(U1)

    return np.abs(u1), 20 * np.log(np.abs(u1) + 1e-10)  # Linear and log-scaled intensity


def display_images(images, titles):
    """Displays multiple images side by side."""
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def process_and_display_row(row):
    image = mnist_image(row)
    padded = mnist_padded_image(row, size=400)
    fft_spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(padded))))

    # Fresnel simulation parameters
    wavelength = 633e-9   # red light ~633nm
    distance = 0.1        # 10 cm
    pixel_pitch = 10e-6   # 10 micrometers per pixel

    amplitude, log_intensity = fresnel_diffraction_angular_spectrum(padded, wavelength, distance, pixel_pitch)

    display_images(
        [image, padded, fft_spectrum, amplitude],
        ["Original 28x28", "Padded 400x400", "FFT Spectrum", "Fresnel Diffraction"]
    )

def save_fresnel_mnist_digit_instances(df_data, df_labels, digit, count=20):
    """
    Saves 'count' Fresnel-diffraction-processed images of a specified MNIST digit
    into a folder named after the digit in the current directory.

    Parameters:
    - df_data: DataFrame with pixel data (without label)
    - df_labels: Series with digit labels
    - digit: Digit class to filter (e.g., 1)
    - count: Number of images to save (default: 10)
    """
    digit_folder = os.path.join(os.getcwd(), str(digit))
    os.makedirs(digit_folder, exist_ok=True)

    # Fresnel parameters
    wavelength = 633e-9   # 633 nm
    distance = 0.01        # 102 cm
    pixel_pitch = 10e-6   # 10 micrometers

    indices = df_labels[df_labels == digit].index[:count]
    for i, idx in enumerate(indices):
        padded = mnist_padded_image(df_data.iloc[idx], size=400)
        amplitude, log_intensity = fresnel_diffraction_angular_spectrum(padded, wavelength, distance, pixel_pitch)
        distance += 0.01
        file_path = os.path.join(digit_folder, f'{i+1}.png')
        plt.imsave(file_path, amplitude, cmap='gray')
        print(f"Saved Fresnel-processed image: {file_path}")

# Load MNIST data
df_train = pd.read_csv("mnist_train.csv")
df_train_labels = df_train['label']
df_train = df_train.drop('label', axis=1)

digit = 9

# Select first image with the desired digit
sample_idx = df_train_labels[df_train_labels == digit].index[0]
sample = df_train.iloc[sample_idx]

process_and_display_row(sample)
save_fresnel_mnist_digit_instances(df_train, df_train_labels, digit=digit)