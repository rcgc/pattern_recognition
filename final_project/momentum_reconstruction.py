import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
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
    Saves 'count' Fresnel-diffraction-processed images of a specified digit
    and one padded MNIST image into a folder named after the digit.

    Parameters:
    - df_data: DataFrame with pixel data (without label)
    - df_labels: Series with digit labels
    - digit: Digit class to filter (e.g., 1)
    - count: Number of images to save (default: 10)
    """
    digit_folder = os.path.join(os.getcwd(), str(digit))
    os.makedirs(digit_folder, exist_ok=True)

    # Fresnel parameters
    wavelength = 633e-9     # 633 nm
    distance = 0.001        # starting at 1 mm
    pixel_pitch = 10e-6     # 10 micrometers
    increment = 0.01       # Optional: increment distance 1 cm

    indices = df_labels[df_labels == digit].index[:count]

    # Save padded image only once (from the first sample)
    if len(indices) > 0:
        first_idx = indices[0]
        padded_once = mnist_padded_image(df_data.iloc[first_idx], size=400)
        padded_filename = os.path.join(digit_folder, f'padded.png')
        plt.imsave(padded_filename, padded_once, cmap='gray')
        print(f"Saved padded image: {padded_filename}")

    for i, idx in enumerate(indices):
        padded = mnist_padded_image(df_data.iloc[idx], size=400)

        # Apply Fresnel diffraction
        amplitude, log_intensity = fresnel_diffraction_angular_spectrum(padded, wavelength, distance, pixel_pitch)

        # Save Fresnel image
        fresnel_filename = os.path.join(digit_folder, f'fresnel_{i+1}.png')
        plt.imsave(fresnel_filename, amplitude, cmap='gray')
        print(f"Saved Fresnel-processed image: {fresnel_filename}")

        distance += increment

def reconstruct_padded_from_fresnel_images(digit_folder, wavelength=633e-9, distance=0.01, pixel_pitch=10e-6):
    """
    Attempts to reconstruct the original padded MNIST image from the 10 Fresnel images
    using inverse angular spectrum propagation and compares it to the saved padded.png.
    """
    fresnel_images = []
    
    # Load all fresnel_*.png images
    for i in range(1, 11):
        path = os.path.join(digit_folder, f'fresnel_{i}.png')
        if os.path.exists(path):
            image = imread(path)
            if image.ndim == 3:  # if RGB, convert to grayscale
                image = image[..., 0]
            fresnel_images.append(image)
    
    if not fresnel_images:
        print("No Fresnel images found.")
        return
    
    # Average the 10 amplitude images (incoherent averaging)
    avg_fresnel = np.mean(fresnel_images, axis=0)

    # Perform inverse angular spectrum propagation
    N, M = avg_fresnel.shape
    fx = np.fft.fftfreq(N, d=pixel_pitch)
    fy = np.fft.fftfreq(M, d=pixel_pitch)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    k = 2 * np.pi / wavelength

    # Transfer function for inverse propagation (conjugate of forward)
    H_inv = np.exp(-1j * 2 * np.pi * distance * np.sqrt((1 / wavelength)**2 - FX**2 - FY**2))

    # Since we only have amplitude, create complex field with zero phase
    U_measured = np.fft.fftshift(np.fft.fft2(avg_fresnel))
    U_back = U_measured * H_inv
    reconstructed = np.abs(np.fft.ifft2(np.fft.ifftshift(U_back)))

    # Load original padded image
    padded_path = os.path.join(digit_folder, 'padded.png')
    if not os.path.exists(padded_path):
        print("Original padded image not found.")
        return
    original_padded = imread(padded_path)
    if original_padded.ndim == 3:  # if RGB, convert to grayscale
        original_padded = original_padded[..., 0]

    # Display comparison
    display_images(
        [original_padded, avg_fresnel, reconstructed],
        ["Original Padded", "Avg. Fresnel Amplitude", "Reconstructed"]
    )

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
reconstruct_padded_from_fresnel_images(str(digit))
