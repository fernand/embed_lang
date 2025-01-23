import cv2
import numpy as np
import voyageai

import api_config

N = 16
MIN = -5
MAX = 5

def to_lab(colors):
    # Normalize the first channel to [0, 100]
    colors[:, :, 0] = (colors[:, :, 0] - MIN) / (MAX - MIN) * 100
    # Normalize the second and third channels to [-127, 128]
    colors[:, :, 1] = (colors[:, :, 1] - MIN) / (MAX - MIN) * (128 + 127) - 127
    colors[:, :, 2] = (colors[:, :, 2] - MIN) / (MAX - MIN) * (128 + 127) - 127
    return colors

# Normalize to [-1 , 1]
def normalize_points(points):
    return 2 * (points - MIN) / (MAX - MIN) - 1

# Assumes [-1 , 1] points
def scale_points_to_image(points, width, height):
    points_scaled = np.empty_like(points)
    points_scaled[:, 0] = (points[:, 0] + 1) * (width / 2.0)
    points_scaled[:, 1] = (points[:, 1] + 1) * (height / 2.0)
    return points_scaled

def voronoi_diagram(width, height, points, colors):
    """
    Generate a Voronoi diagram.

    Args:
        width: Width of the rasterized image.
        height: Height of the rasterized image.
        points: List of (x, y) tuples representing point coordinates.
        colors: List of RGB tuples for the colors corresponding to each point.

    Returns:
        A NumPy array representing the Voronoi diagram.
    """
    y, x = np.indices((height, width))
    x = x[..., np.newaxis]
    y = y[..., np.newaxis]

    points = np.array(points)
    colors = np.array(colors)

    distances = np.sqrt((x - points[:, 0])**2 + (y - points[:, 1])**2)
    nearest_point = np.argmin(distances, axis=-1)

    return colors[nearest_point]

if __name__ == '__main__':
    vo = voyageai.Client(api_key=api_config.VOYAGE_API_KEY)
    # emb = vo.embed(['The dog chased the ball.', 'The cat chased the ball.'], model='voyage-large-2-instruct').embeddings
    # np.save('emb.npy', emb)
    # W = np.random.randn(1024, 5 * 16)
    # np.save('W.npy', W)

    # emb = np.load('emb.npy') # shape [2, 1024]
    emb = np.array(vo.embed(['bicycle', 'He likes to ride his bicycle.', 'She likes to ride her bicycle.'], model='voyage-large-2-instruct').embeddings)
    W = np.load('W.npy') # shape [1024, (3 + 2) * N]
    proj = emb @ W
    points = proj[:, :2*N].reshape(len(emb), N, 2)
    colors = proj[:, 2*N:].reshape(len(emb), N, 3)

    print('Min/Max points', round(np.min(points), 1), round(np.max(points), 1))
    print('Min/Max colors', round(np.min(colors), 1), round(np.max(colors), 1))
    points = normalize_points(points)
    colors = to_lab(colors)

    for i in range(len(emb)):
        scaled_points = scale_points_to_image(points[i], 256, 256)
        diag = voronoi_diagram(256, 256, scaled_points, colors[i]).astype(np.float32)
        rgb_image = cv2.cvtColor(diag, cv2.COLOR_LAB2RGB)
        rgb_image = (rgb_image * 255).astype(np.uint8)
        cv2.imwrite(f'{i}.png', rgb_image)
