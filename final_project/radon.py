import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import radon

def mnist_image(row):
    """Converts a row of MNIST dataset into a 28x28 image array."""
    return np.array(row, dtype=np.float32).reshape(28, 28)

def mnist_padded_image(row, size=128):
    """Pads the MNIST image to a larger square canvas, centered."""
    image = mnist_image(row)
    padded = np.zeros((size, size), dtype=np.float32)
    start = (size - 28) // 2
    padded[start:start+28, start:start+28] = image
    return padded

def fresnel_diffraction(u0, wavelength=633e-9, z=0.1, dx=10e-6):
    """Simulates Fresnel diffraction using FFT-based approach."""
    N, M = u0.shape
    k = 2 * np.pi / wavelength

    # Coordinate grid
    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(M, d=dx)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')

    # Quadratic phase factor (transfer function in frequency domain)
    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))

    # Apply Fresnel diffraction formula
    U1 = np.fft.fft2(u0)
    U2 = H * U1
    u2 = np.fft.ifft2(U2)
    
    return np.abs(u2)**2  # Intensity

def compute_radon_transform(image, theta=None):
    """
    Computes the Radon transform (sinogram) of an image.
    """
    if theta is None:
        theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    return radon(image, theta=theta, circle=False)

def display_images(images, titles):
    """Displays multiple images side by side."""
    fig, axes = plt.subplots(1, len(images), figsize=(18, 5))
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def process_and_display_row(row):
    image = mnist_image(row)
    padded = mnist_padded_image(row, size=128)
    fft_spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(padded))))

    # Fresnel simulation parameters
    wavelength = 633e-9   # 633nm red laser
    distance = 0.1        # 10 cm
    pixel_pitch = 10e-6   # 10 microns

    fresnel_intensity = fresnel_diffraction(padded, wavelength, distance, pixel_pitch)

    # Compute Radon transform (sinogram)
    sinogram = compute_radon_transform(padded)

    display_images(
        [image, padded, fft_spectrum, fresnel_intensity, sinogram],
        ["Original 28x28", "Padded 128x128", "FFT Spectrum", "Fresnel Diffraction", "Radon Transform (Sinogram)"]
    )

# Load MNIST data
df_train = pd.read_csv("mnist_train.csv")
df_train_labels = df_train['label']
df_train = df_train.drop('label', axis=1)

# Select one sample
sample = df_train.iloc[3]
process_and_display_row(sample)
