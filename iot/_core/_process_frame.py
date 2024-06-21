"""Placeholder"""

import numpy as np
import scipy as sp
from skimage import filters, segmentation, morphology, measure, feature, exposure

from iot._utils import image_ops as imops

# from iot.base_classes import LabeledImage
# from iot.image_logger import ImageLogger
from iot._core.nucleus import Nucleus
from iot._core.frame import Frame


def mad_discard(nuclei: list[Nucleus], attribute: str, num_mads: int):
    """Discard nuclei with a statistic deviating too much from the median."""

    attr_list = [getattr(nucleus, attribute) for nucleus in nuclei]

    median = np.median(attr_list)
    deviations = np.abs(attr_list - median)
    mad = np.median(deviations)

    for i in reversed(range(len(attr_list))):
        if not median - num_mads * mad <= attr_list[i] <= median + num_mads * mad:
            nuclei.pop(i)


def _refine_obj_mask(mask: np.ndarray, raw: np.ndarray, opts: dict) -> np.ndarray:
    """Process an individual object mask after rough segmentation."""

    refined_mask = np.zeros(mask.shape, dtype=int)

    for num, blob in enumerate(imops.get_blobs(mask)):

        # # img_log = ImageLogger("debug/single_nuclei")
        raw_img = raw.copy()

        # Restrict the raw signal to the object's boundaries
        bounds = imops.blob_boundaries(blob, opts["cell_blob_margin"])
        raw_img = imops.mono_to_color(imops.zoom_in(raw_img, bounds))
        # # img_log.add_image(LabeledImage("Raw Signal", raw_img))

        # Restrict the mask to the object's boundaries
        blob = imops.zoom_in(blob, bounds)
        # img_log.add_image(LabeledImage("Raw Mask", blob))

        # Fill holes in the putative nuclei
        blob = sp.ndimage.binary_fill_holes(blob)
        # img_log.add_image(LabeledImage("Filled Holes", blob))

        # Create footer which is a fraction of the average between x and y sizes
        f_size = int(np.mean(blob.shape) * opts["cell_foot_frac"])
        footer = morphology.disk(f_size)

        # Create opened and closed masks
        opened_mask = morphology.binary_opening(blob, footer)
        closed_mask = morphology.binary_closing(blob, footer)

        # Discard cell if the difference between masks is too big
        if opened_mask.sum() > 0:
            if closed_mask.sum() / opened_mask.sum() > opts["cell_mask_fold_change"]:
                opened_mask = np.zeros_like(opened_mask)
                closed_mask = np.zeros_like(closed_mask)

        # Compute the blobs for fixing the opened mask
        # TODO: Try erode, compute centroids, use them as seeds on non-eroded mask
        blobs = opened_mask ^ closed_mask
        # blobs = morphology.binary_opening(blobs, morphology.disk(2))
        blobs = imops.dist_watershed(
            blobs,
            opts["cell_water_foot_size"],
            opts["cell_water_min_dist"],
        )
        blobs = measure.label(blobs)
        # img_log.add_image(LabeledImage("Blobs", blobs))

        # Try patching the opened mask with the blobs, keeping only those
        # which produce a minimal shift in the convex hull wrt their size
        with_blobs = opened_mask.copy()
        if with_blobs.sum() == 0:
            init_hull = 0
        else:
            init_hull = morphology.convex_hull_image(opened_mask).sum()

        for diff_blob in imops.get_blobs(blobs):

            # Add the blob to the mask
            blob_mask = opened_mask.copy()
            blob_mask[diff_blob > 0] = 1

            # Get convex hull shift
            blob_size = diff_blob.sum()
            blob_hull = morphology.convex_hull_image(blob_mask).sum()
            diff_hull = blob_hull - init_hull

            if diff_hull < blob_size * opts["cell_hull_shift_frac"]:
                with_blobs[diff_blob > 0] = 1

        # img_log.add_image(LabeledImage("With Blobs", with_blobs))

        # Smoothen after adding the blobs
        with_blobs = morphology.binary_closing(with_blobs, footer)
        # img_log.add_image(LabeledImage("Closed Blobs", with_blobs))

        # Plot the segmented nucleus on the raw image
        overlay = imops.add_outline(raw_img, with_blobs, "orange")
        # img_log.add_image(LabeledImage("Overlay", overlay))

        # Add the mask to the refined masks
        mask = imops.zoom_out(with_blobs, bounds, mask)
        refined_mask[mask > 0] = num + 1

        # if opts["debug"]:
        # img_log.plot_images("group")
        # img_log.clear_images()

    # Relabel the cells
    refined_mask = measure.label(refined_mask)

    clean_frame = Frame(raw, refined_mask)
    nuclei = clean_frame.get_nuclei()

    # TODO: Does not work great if cells have similar size?
    mad_discard(nuclei, "size", opts["cell_size_num_mads"])

    clean_frame = Frame.from_nuclei(raw, nuclei)

    return clean_frame.get_image("labels")


def process_frame(frame: np.ndarray, gamma: float, opts: dict) -> np.ndarray:
    """Segment the cell nuclei in the frame and create a mask."""

    print(f"Processing a frame with gamma {gamma}")

    mask: np.ndarray = frame.copy()

    mask = exposure.adjust_gamma(mask, gamma)

    mask = mask > filters.threshold_otsu(mask)

    mask = segmentation.clear_border(mask)

    mask = morphology.remove_small_objects(mask, min_size=opts["full_small_obj"])

    # Find maxima in the distance map (centers of the areas to segment)

    seed_mask = sp.ndimage.distance_transform_edt(sp.ndimage.binary_fill_holes(mask))

    footer = morphology.disk(opts["full_water_foot_size"])
    # Calculate the average diameter of objects in the image

    coords = feature.peak_local_max(
        seed_mask,
        min_distance=opts["full_water_min_dist"],
        footprint=footer,
        labels=mask,
    )

    # Use the maxima to create a seed map for the watershed
    seeds_map = np.zeros(seed_mask.shape, dtype=bool)
    for i in range(coords.shape[0]):
        x, y = coords[i]
        seeds_map[x, y] = True
    seeds_map, _ = sp.ndimage.label(seeds_map)

    mask = segmentation.watershed(seed_mask * -1, seeds_map, mask=mask)

    # import matplotlib.pyplot as plt

    # plt.imshow(mask)
    # plt.show()

    mask = _refine_obj_mask(mask, frame, opts)

    return mask.astype(int)
