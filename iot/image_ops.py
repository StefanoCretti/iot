"""Placeholder"""

from typing import Iterable

from matplotlib.colors import ListedColormap
from matplotlib import colormaps as cm
import numpy as np
import scipy as sp
from skimage import feature, segmentation, morphology


def min_max_scale(img: np.ndarray) -> np.ndarray:
    """Perform min-max scaling on the input image."""

    img_min = img.min()
    img_max = img.max()

    if img_min == img_max:
        return img

    return (img - img_min) / (img_max - img_min)


def to_pillow_range(img: np.ndarray) -> np.ndarray:
    """Convert the image to the range [0, 255] and cast to uint8."""

    img = min_max_scale(img) * 255
    return img.astype("uint8")


def mono_to_color(
    img: np.ndarray,
    palette: str | ListedColormap = "viridis",
) -> np.ndarray:
    """Convert a monochromatic image to a colored one using a palette."""

    cmap = cm.get_cmap(palette)
    norm = min_max_scale(img)
    return cmap(norm)[..., :3]


def dist_watershed(
    img: np.ndarray,
    footer_size: int,
    min_distance: int = 1,
) -> np.ndarray:
    """Watershed segmentation using distance map to define seeds."""

    # Find maxima in the distance map (centers of the areas to segment)
    dist: np.ndarray = sp.ndimage.distance_transform_edt(img)
    footer = morphology.disk(footer_size)
    coords = feature.peak_local_max(dist, min_distance, footprint=footer, labels=img)

    # Use the maxima to create a seed map for the watershed
    seeds_map = np.zeros(dist.shape, dtype=bool)
    seeds_map[tuple(coords.T)] = True
    seeds_map, _ = sp.ndimage.label(seeds_map)

    return segmentation.watershed(dist * -1, seeds_map, mask=img)


def blob_boundaries(
    blob_mask: np.ndarray,
    tolerance: float,
) -> tuple[int, int, int, int]:
    """Return the boundaries of the blob with a given tolerance (%)."""

    non_zero = blob_mask.nonzero()
    x_min, x_max = min(non_zero[0]), max(non_zero[0])
    y_min, y_max = min(non_zero[1]), max(non_zero[1])

    x_diff = x_max - x_min
    y_diff = y_max - y_min
    shift = int(max(x_diff, y_diff) * tolerance)

    x_min = max(0, x_min - shift)
    x_max = min(blob_mask.shape[0], x_max + shift)
    y_min = max(0, y_min - shift)
    y_max = min(blob_mask.shape[1], y_max + shift)

    return (x_min, x_max, y_min, y_max)


def zoom_in(img: np.ndarray, coords: tuple[int, int, int, int]) -> np.ndarray:
    """Return a zoomed-in image based on the provided coords."""

    x_min, x_max, y_min, y_max = coords
    return img[x_min:x_max, y_min:y_max]


def zoom_out(
    img: np.ndarray,
    coords: tuple[int, int, int, int],
    full_mask: np.ndarray,
) -> np.ndarray:
    """Return a zoomed-out image based on the provided coords."""

    x_min, x_max, y_min, y_max = coords
    zoomed_out = np.zeros(full_mask.shape)
    zoomed_out[x_min:x_max, y_min:y_max] = img
    return zoomed_out


def iterate_blobs(full_mask: np.ndarray) -> Iterable[np.ndarray]:
    """Iterate over the blobs returning their binary masks one at a time."""

    for label in range(1, full_mask.max() + 1):
        single_mask = (full_mask == label).reshape(full_mask.shape)
        single_mask[single_mask > 0] = 1

        if single_mask.sum() == 0:
            continue

        yield single_mask


def get_outline(full_mask: np.ndarray) -> np.ndarray:
    """Return a version of the cell mask with only the outline."""

    outline = np.zeros(full_mask.shape, dtype=bool)
    outline[segmentation.find_boundaries(full_mask)] = True
    return outline


def add_outline(raw: np.ndarray, mask: np.ndarray, color: str) -> np.ndarray:
    """Add an outline to the raw image based on the mask and color."""

    if len(raw.shape) == 2:
        raw = mono_to_color(raw)

    # NOTE: Using a mock palette which is either black or the desired color
    # Black is required since it is zero in all channels, axis 2 sums to zero
    outline = get_outline(mask).astype("uint") * 255
    palette = ListedColormap(["black", color])
    outline = mono_to_color(outline, palette)

    overlap = outline.sum(axis=2) > 0
    overlay = raw.copy()
    for ch in range(3):
        overlay[overlap, ch] = outline[overlap, ch]

    return overlay
