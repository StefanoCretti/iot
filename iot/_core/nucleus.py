"""Placeholder."""

from typing import Iterable

import numpy as np

from .._utils import image_ops as imops

# TODO: Remove these tmp imports for testing purposes
import matplotlib.pyplot as plt
from skimage import morphology


class Nucleus:
    """Placeholder"""

    def __init__(
        self,
        mask: np.ndarray,
        raw: np.ndarray | None = None,
        tolerance: float = 0,
    ):

        # Centroid needs to be computed before zooming in
        info = imops.get_nuclei_info(mask, raw, ["centroid"])
        self._centroid = np.array((info["centroid-0"], info["centroid-1"]))

        self._bounds = imops.blob_boundaries(mask, tolerance)
        self._mask = imops.zoom_in(mask, self._bounds)
        self._raw = imops.zoom_in(raw, self._bounds) if raw is not None else None
        self._full_size = mask.shape
        self._size: int = self._mask.sum()

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Placeholder"""
        return self._bounds

    @property
    def size(self) -> int:
        """Placeholder"""
        return self._size

    @property
    def centroid(self) -> np.ndarray:
        """Placeholder"""
        return self._centroid

    @property
    def mean_intensity(self) -> float:
        """Placeholder"""
        return np.mean(self._raw[self._mask])

    def get_mask(self, zoomed: bool = True) -> np.ndarray:
        """Placeholder"""

        if zoomed:
            return self._mask
        return imops.zoom_out(self._mask, self._bounds, np.zeros(self._full_size))

    # def does_overlap(self, other: "Nucleus") -> bool:
    #     """Return whether cell bounding boxes have any intersection."""

    #     x_min_s, x_max_s, y_min_s, y_max_s = self._bounds
    #     x_min_o, x_max_o, y_min_o, y_max_o = other.bounds

    #     x_overlap = (x_min_o <= x_max_s <= x_max_o) or (x_min_o <= x_min_s <= x_max_o)
    #     y_overlap = (y_min_o <= y_max_s <= y_max_o) or (y_min_o <= y_min_s <= y_max_o)

    #     return x_overlap and y_overlap

    def __and__(self, other: "Nucleus") -> np.ndarray:
        """Return amount of pixels shared by the two nuclei."""

        overlap = self.get_mask(zoomed=False) * other.get_mask(zoomed=False)
        return overlap

    def __or__(self, other: "Nucleus") -> np.ndarray:
        """Placeholder"""
        inters = self.get_mask(zoomed=False) + other.get_mask(zoomed=False)
        inters[inters > 1] = 1
        return inters

    # def shift_centroid(self, new_centroid: np.ndarray) -> "Nucleus":
    #     """Placeholder"""

    #     old_centroid = self.centroid
    #     shift = new_centroid - old_centroid

    #     new_bounds = np.array(self._bounds)
    #     new_bounds[:2] = new_bounds[:2] + shift[0]
    #     new_bounds[2:] = new_bounds[2:] + shift[1]

    #     # TODO: Check for collision with the image borders

    #     new_mask = np.zeros(self._full_size, dtype=bool)
    #     new_mask[new_bounds] = self._mask
    #     return Nucleus(new_mask)


class NucleiSeq:
    """Placeholder"""

    def __init__(self, nucleus: "Nucleus", gamma: float):
        self._gammas: list[float] = [gamma]
        self._nuclei: list["Nucleus"] = [nucleus]

    @property
    def nuclei(self) -> Iterable["Nucleus"]:
        """Placeholder"""
        return self._nuclei

    @property
    def gammas(self) -> Iterable[float]:
        """Placeholder"""
        return self._gammas

    @property
    def last_gamma(self) -> float:
        """Placeholder"""
        return self._gammas[-1]

    @property
    def last_nucleus(self) -> "Nucleus":
        """Placeholder"""
        return self._nuclei[-1]

    def find_match(
        self,
        nuclei: list["Nucleus"],
        max_dist: int,
        diff_frac: float,
    ) -> "Nucleus":
        """Placeholder"""

        old = self.last_nucleus

        dists = [np.linalg.norm(old.centroid - new.centroid) for new in nuclei]

        min_dist = np.min(dists)

        if min_dist > max_dist:
            print("Returned None due to distance")
            return False

        new = nuclei[np.argmin(dists)]
        diff = (old | new) - (old & new)

        # If you move an object, the difference between the two masks will have
        # a width which is at most distance moved, therefore use a disk with
        # diamater comparable to the shift to remove expected mask difference.
        footer = morphology.disk(np.ceil(min_dist / 2))
        diff = morphology.erosion(diff, footprint=footer)

        # TODO: could add to footer a term proportional to the change in
        # equivalent diameter of the object, to account for shape change,
        # though it should hopefully be small enough to not affect the result.

        # print(old.size)
        # print(abs(old.size - new.size))
        if diff.sum() > old.size * diff_frac:
            return False

        return new

        # TODO: save the best one found and use it if no better is found
        # TODO: Use more than one frame as reference

        # score = (old & new) / (old | new)
        # if score > min_score:
        # score = abs(old.size - new.size) / (old.size + new.size)
        # print(score)
        # if score < min_score:
        #     self._gammas.append(gamma)
        #     self._nuclei.append(new)
        #     return True

        # return False

    def add_nucleus(self, nucleus: Nucleus, gamma: float):
        """Placeholder"""

        self._nuclei.append(nucleus)
        self._gammas.append(gamma)
