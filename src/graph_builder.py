import cv2
import numpy as np
from scipy.spatial import Delaunay, Voronoi

def detect_nuclei_centroids(image_path):
    """
    Detects the centroids of nuclei in a histopathology image.
    This is a placeholder implementation. A more sophisticated method should be used for real applications.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))

    return np.array(centroids)

def build_graph(centroids, method='delaunay'):
    """
    Builds a graph from the given centroids using either Delaunay triangulation or Voronoi tessellation.
    """
    if method == 'delaunay':
        tri = Delaunay(centroids)
        return tri.simplices
    elif method == 'voronoi':
        vor = Voronoi(centroids)
        return vor.ridge_points
    else:
        raise ValueError("Invalid graph building method. Choose either 'delaunay' or 'voronoi'.")
