import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load and convert image to grayscale
img = Image.open('butterfly.jpg').convert('L')  # 'L' mode = grayscale
fly = np.array(img, dtype=float)

plt.figure()
plt.imshow(fly, cmap='gray')
plt.axis('off')
plt.axis('equal')

# Center the data by subtracting the mean of each row
m, n = fly.shape
mn = np.mean(fly, axis=1, keepdims=True)
X = fly - mn

# Covariance matrix
Z = (1 / np.sqrt(n - 1)) * X.T
covZ = Z.T @ Z

# SVD
U, S, Vt = np.linalg.svd(covZ)
V = Vt.T

# Variances (eigenvalues squared)
variances = S**2
plt.figure()
plt.bar(np.arange(len(variances)), variances, color='b')
plt.xlim([0, 20])
plt.xlabel('eigenvector number')
plt.ylabel('eigenvalue')

tot = np.sum(variances)
cumvar = np.cumsum(variances) / tot
print(np.column_stack((np.arange(1, len(cumvar) + 1), cumvar)))

# Compression using top PCs
PCs = 40
VV = V[:, :PCs]
Y = VV.T @ X
compression_ratio = m / (2 * PCs + 1)
print(f"Compression ratio: {compression_ratio:.2f}")

XX = VV @ Y
XX = XX + mn

plt.figure()
plt.imshow(XX, cmap='gray')
plt.axis('off')
plt.axis('equal')

# Multiple reconstructions with varying PCs
pcs_list = [2, 6, 10, 14, 20, 30, 40, 60, 90, 120, 150, 180]
plt.figure(figsize=(12, 10))
for z, PCs in enumerate(pcs_list, 1):
    VV = V[:, :PCs]
    Y = VV.T @ X
    XX = VV @ Y
    XX = XX + mn

    plt.subplot(4, 3, z)
    plt.imshow(XX, cmap='gray')
    plt.axis('off')
    plt.axis('equal')
    comp_ratio = round(10 * m / (2 * PCs + 1)) / 10
    plt.title(f'{comp_ratio}:1 compression\n{PCs} principal components')

plt.tight_layout()
plt.show()
