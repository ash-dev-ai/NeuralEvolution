import numpy as np
from scipy.ndimage import sobel
from sklearn.cluster import KMeans

class FitnessUtils:
    @staticmethod
    def calculate_symmetry(canvas):
        """Calculate horizontal symmetry score (0-1)"""
        try:
            if canvas is None or canvas.ndim != 2:
                return 0.0
                
            h, w = canvas.shape
            half = w // 2
            left = canvas[:, :half]
            right = np.flip(canvas[:, half + w%2:], axis=1)
            return 1 - np.mean(np.abs(left - right))
        except Exception as e:
            print(f"Symmetry calculation error: {e}")
            return 0.0

    @staticmethod
    def calculate_contrast(canvas):
        """Calculate contrast with minimum safety threshold"""
        try:
            return max(np.std(canvas), 0.1)  # Prevent complete flatness
        except Exception as e:
            print(f"Contrast calculation error: {e}")
            return 0.1

    @staticmethod
    def detect_edge_density(canvas, threshold=0.2):
        """Calculate proportion of significant edges using Sobel operator"""
        try:
            dx = sobel(canvas, axis=0)
            dy = sobel(canvas, axis=1)
            edge_strength = np.hypot(dx, dy)
            return np.mean(edge_strength > threshold)
        except Exception as e:
            print(f"Edge detection error: {e}")
            return 0.0

    @staticmethod 
    def calculate_color_coherence(canvas, n_clusters=3):
        """Measure color organization using k-means clustering (0-1)"""
        try:
            pixels = (canvas * 255).reshape(-1, 1).astype(np.uint8)
            kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(pixels)
            cluster_sizes = np.bincount(kmeans.labels_)
            return 1 - (np.max(cluster_sizes) / len(pixels))  # 1 = balanced
        except Exception as e:
            print(f"Color clustering error: {e}")
            return 0.0

    @staticmethod
    def calculate_active_area(canvas, threshold=0.1):
        """Measure proportion of non-blank canvas area (0-1)"""
        try:
            return np.mean(canvas > threshold)
        except Exception as e:
            print(f"Active area calculation error: {e}")
            return 0.0