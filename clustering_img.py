import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib import cm  # Importar el módulo de colormaps

def random_numbers(seed, low, high, size):
    state = seed
    results = []
    for _ in range(size):
        state = (state * 1664525 + 1013904223) % (2**32)
        results.append(low + (state % (high - low)))
    return results

def generate_colors(k):
    """Genera una lista de colores usando la paleta por defecto de Matplotlib."""
    colors = [cm.tab10(i)[:3] for i in range(k)]  # Usar la colormap 'tab10', que tiene 10 colores distintos
    return colors

def clip(value, min_value, max_value):
    """Asegura que el valor esté dentro del rango [min_value, max_value]."""
    return max(min_value, min(max_value, value))

def cluster_brain_mri(image_path, output_path, k=4, seed=42):
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
    
    # Inicializar centroides usando la función random_numbers
    indices = random_numbers(seed, 0, len(pixels), k)
    centroids = [pixels[i] for i in indices]
    clusters = [0] * len(pixels)

    # Generar colores para cada cluster usando la paleta de Matplotlib
    colors = generate_colors(k)
    
    # Función para calcular distancia euclidiana
    def euclidean_distance(p1, p2):
        return sum((p1[i] - p2[i]) ** 2 for i in range(len(p1)))
    
    # K-Means simple sin librerías
    for _ in range(10):  # Número de iteraciones fijas
        new_centroids = [(0,) * len(centroids[0])] * k
        counts = [0] * k
        
        for i, pixel in enumerate(pixels):
            distances = [euclidean_distance(pixel, centroid) for centroid in centroids]
            cluster = distances.index(min(distances))
            clusters[i] = cluster
            new_centroids[cluster] = tuple(new_centroids[cluster][j] + pixel[j] for j in range(len(pixel)))
            counts[cluster] += 1
        
        for i in range(k):
            if counts[i] > 0:
                centroids[i] = tuple(int(new_centroids[i][j] / counts[i]) for j in range(len(new_centroids[i])))
                centroids[i] = tuple(clip(v, 0, 255) for v in centroids[i])  # Asegurar rango [0, 255]
    
    # Crear imagen segmentada utilizando los colores
    clustered_image = [[colors[clusters[i * w + j]] for j in range(w)] for i in range(h)]
    
    # Guardar la imagen procesada
    plt.imsave(output_path, clustered_image, cmap=None)  # Usar None para no aplicar cmap
    
    # Mostrar la imagen original y la segmentada
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap='gray' if image.shape[2] == 1 else None)
    ax[0].set_title("Imagen Original")
    ax[0].axis("off")
    
    ax[1].imshow(clustered_image)
    ax[1].set_title("Imagen Clustering K-Means")
    ax[1].axis("off")
    
    plt.show()

if __name__ == "__main__":
    cluster_brain_mri("brain_mri.jpg", "output.jpg", k=3, seed=42)
