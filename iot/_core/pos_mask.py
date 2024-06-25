"""Placeholder"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from skimage import measure, morphology

from .._typing import Bbox


def masks_from_frame(full_mask: np.ndarray) -> Iterable["PosMask"]:
    """Placeholder"""

    labels = measure.label(full_mask)
    region = measure.regionprops(labels)

    for props in region:
        bbox = props.bbox

        mask = full_mask[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        mask = (mask == props.label).astype(bool)
        yield PosMask(mask, bbox)


def joint_bbox(masks: Iterable["PosMask"], margin: float = 0.0) -> Bbox:
    """Return the joint bounding box of the masks with a given margin (%)."""

    bboxes = [m.bbox for m in masks]

    x_min = min(b[0] for b in bboxes)
    x_max = max(b[2] for b in bboxes)
    y_min = min(b[1] for b in bboxes)
    y_max = max(b[3] for b in bboxes)

    x_diff = x_max - x_min
    y_diff = y_max - y_min
    shift = int(max(x_diff, y_diff) * margin)

    return (x_min - shift, y_min - shift, x_max + shift, y_max + shift)


class PosMask:
    """Cropped mask of a single object in an image.

    Class used to focus on an individual object in the image at a time.
    Rather than creating multiple, full-size masks, this mask allows to
    work on a crop by saving the bounding box of the object.
    """

    def __init__(self, mask: np.ndarray, bbox: Bbox):
        self._bbox = bbox
        self._mask = mask

    @property
    def mask(self) -> np.ndarray:
        """Cropped mask as a numpy array."""
        return self._mask

    @property
    def bbox(self) -> Bbox:
        """Bounding box of the object in the full image."""
        return self._bbox

    @property
    def centroid(self) -> np.ndarray:
        """Centroid coordinates of the object with respect to the full image."""
        centroid = measure.centroid(self._mask)
        return np.array((centroid[0] + self._bbox[0], centroid[1] + self._bbox[1]))

    @property
    def area(self) -> int:
        """Area of the object in squared pixels."""
        return self._mask.sum()

    def bbox_inside(self, bbox: Bbox) -> bool:
        """Return whether the object is fully inside the given bounding box."""
        return all(
            (
                bbox[0] <= self._bbox[0],
                bbox[1] <= self._bbox[1],
                bbox[2] >= self._bbox[2],
                bbox[3] >= self._bbox[3],
            )
        )

    def bbox_project(self, bbox: Bbox) -> np.ndarray:
        """Return the mask projected into the given bounding box."""

        if not self.bbox_inside(bbox):
            raise ValueError("Object is not fully inside the given bounding box.")

        x_min, y_min, x_max, y_max = bbox
        mask = np.zeros((x_max - x_min, y_max - y_min), dtype=bool)
        mask[
            self._bbox[0] - x_min : self._bbox[2] - x_min,
            self._bbox[1] - y_min : self._bbox[3] - y_min,
        ] = self._mask

        return mask

    def __and__(self, other: "PosMask") -> "PosMask":
        """Return intersection of the two masks."""

        bbox = joint_bbox([self, other])
        mask = self.bbox_project(bbox) * other.bbox_project(bbox)
        return PosMask(mask, bbox)

    def __or__(self, other: "PosMask") -> "PosMask":
        """Return union of the two masks."""

        bbox = joint_bbox([self, other])
        mask = self.bbox_project(bbox) + other.bbox_project(bbox)
        return PosMask(mask, bbox)

    def best_match(
        self,
        others: Iterable["PosMask"],
        max_dist: int,
        diff_frac: float,
    ) -> "PosMask" | None:
        """Find the best match for the object (if any) among the other objects."""

        dists = [np.linalg.norm(self.centroid - new.centroid) for new in others]
        min_dist = np.min(dists)

        if min_dist > max_dist:
            return None

        new = others[np.argmin(dists)]

        diff_mask = (self | new).mask ^ (self & new).mask

        # If you move an object, the difference between the two masks will have
        # a width which is at most distance moved, therefore use a disk with
        # diamater comparable to the shift to remove expected mask difference.
        footer = morphology.disk(np.ceil(min_dist / 2))
        diff_mask = morphology.erosion(diff_mask, footprint=footer)

        # TODO: could add to footer a term proportional to the change in
        # equivalent diameter of the object, to account for shape change,
        # though it should hopefully be small enough to not affect the result.

        if diff_mask.sum() > self.area * diff_frac:
            return None

        return new
