"""Placeholder."""

from typing import TYPE_CHECKING

import pandas as pd


from .._utils import image_ops as imops
from .pos_mask import PosMask

if TYPE_CHECKING:
    from .fov import Fov

# TODO: Remove these tmp imports for testing purposes
import matplotlib.pyplot as plt
from skimage import morphology
from .._utils import image_ops as imops


class Nucleus:
    """Placeholder"""

    def __init__(
        self,
        mask: "PosMask",
        fov: "Fov",
        label: int = 1,
    ):
        self._masks: list["PosMask"] = [mask]
        self._label: int = label
        self._fov: "Fov" = fov

    @property
    def centroid(self) -> list[tuple[int, int]]:
        """Return sequence of centroid positions for the nucleus overtime."""
        return [m.centroid for m in self._masks]

    @property
    def area(self) -> list[int]:
        """Return sequence of areas for the nucleus overtime."""
        return [m.area for m in self._masks]

    @property
    def label(self) -> int:
        """Return the label of the nucleus."""
        return self._label

    def _skimage_prop(self, prop: str | list[str]) -> pd.DataFrame:
        """Return a skimage property of the nucleus overtime."""
        raws = self._fov.get_frames("raw")
        masks = [m.bbox_project((0, 0, *self._fov.shape)) for m in self._masks]
        masks = [m.astype(int) for m in masks]  # TODO: Decide bool or int for masks
        props = [imops.skimage_props(m, r, prop) for m, r in zip(masks, raws)]
        return pd.concat(props, ignore_index=True)

    # @property
    # def mean_intensity(self) -> list[float]:
    #     """Return the mean intensity of the nucleus overtime."""
    #     return self._skimage_prop(["mean_intensity"])

    # @property
    # def max_intensity(self) -> list[float]:
    #     """Return the max intensity of the nucleus overtime."""
    #     return self._skimage_prop(["max_intensity"])

    # @property
    # def min_intensity(self) -> list[float]:
    #     """Return the min intensity of the nucleus overtime."""
    #     return self._skimage_prop(["min_intensity"])

    @property
    def masks(self) -> tuple["PosMask"]:
        """Return the masks of the nucleus."""
        return tuple(self._masks)

    @property
    def info(self) -> pd.DataFrame:
        """Return information about the nucleus in a DataFrame."""

        info = self._skimage_prop(["mean_intensity", "max_intensity", "min_intensity"])
        info["area"] = self.area
        info["id"] = [self.label] * len(self._masks)
        return info

    def add_mask(self, mask: "PosMask"):
        """Add a new mask (new time point) to the nucleus."""
        self._masks.append(mask)
