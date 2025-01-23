import cv2
import numpy as np
import voyageai
from PIL import Image, ImageDraw, ImageFont

import api_config

"""
    W = np.random.randn(1024, 5 * 16).astype(np.float32)
"""

MODEL = 'voyage-lite-02-instruct'
N = 16
MIN = -3
MAX = 3

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

def create_image_grid_with_captions(rgb_images, captions, cols=4,
                                    padding=10, font_path=None, font_size=16):
    """
    Lay out the `rgb_images` (list of PIL Images or np arrays)
    in a grid with `cols` columns, writing each corresponding
    string in `captions` below it. Return a new PIL Image.

    :param rgb_images: list of either PIL Images or np arrays with shape [H, W, 3]
    :param captions:   list of string captions
    :param cols:       number of columns in the grid
    :param padding:    margin (pixels) between cells and around edges
    :param font_path:  path to a TTF font (optional). If None, will use default PIL font.
    :param font_size:  size of the text font
    """
    assert len(rgb_images) == len(captions), "images and captions must match in length"

    # Convert np arrays to PIL Images if needed, and measure dimension
    pil_images = []
    for im in rgb_images:
        if isinstance(im, np.ndarray):
            pil_images.append(Image.fromarray(im))  # np.uint8 -> PIL
        else:
            pil_images.append(im)

    w, h = pil_images[0].size

    # Choose a font
    if font_path is not None:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    # We need to measure text height to offset images from captions
    dummy_draw = ImageDraw.Draw(pil_images[0])
    text_heights = []
    for cap in captions:
        # you can measure the bounding box if you want to be more precise
        bbox = dummy_draw.textbbox((0,0), cap, font=font)
        text_height = bbox[3] - bbox[1]
        text_heights.append(text_height)
    # Or just pick a single text height to use for all if you prefer
    max_text_height = max(text_heights)

    # Calculate grid size
    rows = (len(pil_images) + cols - 1) // cols

    # Overall canvas size
    out_w = padding + cols*(w + padding)
    out_h = padding + rows*(h + max_text_height + padding)

    # Create new white canvas
    out_img = Image.new("RGB", (out_w, out_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(out_img)

    x = y = padding
    for i, (img, cap) in enumerate(zip(pil_images, captions)):
        # Paste image
        out_img.paste(img, (x, y))

        # Write caption below the image
        text_x = x
        text_y = y + h  # just below the image
        draw.text((text_x, text_y), cap, fill=(0, 0, 0), font=font)

        # Move to next column
        x += w + padding
        # If we filled this row, reset and move to next row
        if (i + 1) % cols == 0:
            x = padding
            y += (h + max_text_height + padding)

    return out_img

if __name__ == '__main__':
    captions = [
        'animal',
        'animals',
        'bicycle',
        'bicycles',
        'He likes to ride his bicycle.',
        'She likes to ride her bicycle.'
    ]

    vo = voyageai.Client(api_key=api_config.VOYAGE_API_KEY)
    emb = np.array(vo.embed(captions, model=MODEL).embeddings, dtype=np.float32)

    W = np.load('W_1024.npy').astype(np.float32)
    proj = emb @ W
    points = proj[:, :2*N].reshape(len(emb), N, 2)
    colors = proj[:, 2*N:].reshape(len(emb), N, 3)

    print('Min/Max points', round(np.min(points), 1), round(np.max(points), 1))
    print('Min/Max colors', round(np.min(colors), 1), round(np.max(colors), 1))
    points = normalize_points(points)
    colors = to_lab(colors)

    rgb_diagrams = []
    for i in range(len(emb)):
        scaled_points = scale_points_to_image(points[i], 256, 256)
        diagram = voronoi_diagram(256, 256, scaled_points, colors[i])
        rgb_image = cv2.cvtColor(diagram, cv2.COLOR_LAB2RGB)
        rgb_image = (rgb_image * 255).astype(np.uint8)
        rgb_diagrams.append(rgb_image)

    output = create_image_grid_with_captions(rgb_diagrams, captions, cols=2)
    output.save('diagrams.png')