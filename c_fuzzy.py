import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib import cm

def random_numbers(seed, low, high, size):
    state = seed
    results = []
    for _ in range(size):
        state = (state * 1664525 + 1013904223) % (2**32)
        results.append(low + (state % (high - low)))
    return results

def generate_colors(k):
    """Genera una lista de colores usando la paleta por defecto de Matplotlib."""
    colors = [cm.tab10(i)[:3] for i in range(k)]
    return colors

def clip(value, min_value, max_value):
    """Asegura que el valor esté dentro del rango [min_value, max_value]."""
    return max(min_value, min(max_value, value))

def euclidean_distance(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return sum((p1[i] - p2[i]) ** 2 for i in range(len(p1)))

def c_fuzzy_clustering(pixels, k, m=2, max_iter=100, tol=1e-5, seed=42):
    """Implementación del clustering difuso de tipo c-fuzzy."""
    # Inicializar los centroides aleatoriamente
    indices = random_numbers(seed, 0, len(pixels), k)
    centroids = [pixels[i] for i in indices]
    n = len(pixels)

    # Inicializar la matriz de pertenencia
    U = [[0] * k for _ in range(n)]
    for i in range(n):
        random_values = random_numbers(seed + i, 0, 100, k)
        total = sum(random_values)
        U[i] = [value / total for value in random_values]

    for _ in range(max_iter):
        # Actualizar los centroides
        centroids = []
        for j in range(k):
            num = [0] * len(pixels[0])
            den = 0
            for i in range(n):
                weight = U[i][j] ** m
                num = [num[x] + weight * pixels[i][x] for x in range(len(pixels[i]))]
                den += weight
            centroids.append([clip(num[x] / den, 0, 255) for x in range(len(num))])

        # Actualizar la matriz de pertenencia
        new_U = [[0] * k for _ in range(n)]
        for i in range(n):
            for j in range(k):
                new_U[i][j] = 1 / sum((euclidean_distance(pixels[i], centroids[j]) / euclidean_distance(pixels[i], centroids[c])) ** (2 / (m - 1))
                                      for c in range(k))

        # Verificar la convergencia
        if all(abs(new_U[i][j] - U[i][j]) < tol for i in range(n) for j in range(k)):
            break
        
        U = new_U

    return centroids, U

def cluster_brain_mri(image_path, output_path, k=4, m=2, seed=42):
    # Cargar la imagen
    image = imread(image_path)

    # Manejar imágenes en escala de grises
    if len(image.shape) == 2:  # Imagen en escala de grises
        h, w = image.shape
        image = image.reshape(h, w, 1)  # Agregar dimensión de canal
    else:
        h, w, c = image.shape

    # Convertir la imagen en una lista de píxeles
    pixels = [tuple(clip(int(image[i, j][b] * 255), 0, 255) for b in range(image.shape[2])) for i in range(h) for j in range(w)]

    # Ejecutar clustering difuso c-fuzzy
    centroids, U = c_fuzzy_clustering(pixels, k, m, seed=seed)

    # Crear imagen segmentada utilizando los colores
    colors = generate_colors(k)
    clustered_image = [[(0, 0, 0) for _ in range(w)] for _ in range(h)]
    for i in range(h):
        for j in range(w):
            pixel_index = i * w + j
            for c in range(k):
                r, g, b = colors[c]
                membership = U[pixel_index][c]
                clustered_image[i][j] = (clip(clustered_image[i][j][0] + r * membership, 0, 255),
                                          clip(clustered_image[i][j][1] + g * membership, 0, 255),
                                          clip(clustered_image[i][j][2] + b * membership, 0, 255))

    # Guardar la imagen procesada
    plt.imsave(output_path, clustered_image, cmap=None)  # Usar None para no aplicar cmap

    # Mostrar la imagen original y la segmentada
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap='gray' if image.shape[2] == 1 else None)
    ax[0].set_title("Imagen Original")
    ax[0].axis("off")

    ax[1].imshow(clustered_image)
    ax[1].set_title("Imagen Clustering C-Fuzzy")
    ax[1].axis("off")

    plt.show()

if __name__ == "__main__":
    cluster_brain_mri("brain_mri.jpg", "output_fuzzy.jpg", k=4, m=2, seed=42)
