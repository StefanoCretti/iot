"""Placeholder."""

from typing import Iterable

import numpy as np
import pandas as pd
import PIL

from .._utils import image_ops as imops
from .nucleus import Nucleus


class Frame:
    """Individual frame of a time-lapse microscopy sequence."""

    def __init__(self, raw: np.ndarray, mask: np.ndarray) -> None:
        self._raw_frame: np.ndarray = raw
        self._nuclei_mask: np.ndarray = mask

    @classmethod
    def from_nuclei(cls, raw: np.ndarray, nuclei: Iterable[Nucleus]):
        """Factory method to create a Frame starting from Nuclei instances."""

        # TODO: Doc note that overlapping masks override each other

        mask = np.zeros_like(raw)
        for ind, nucleus in enumerate(nuclei):
            mask[nucleus.get_mask(zoomed=False) == 1] = ind + 1

        return cls(raw, mask)

    @property
    def nuclei_info(self) -> pd.DataFrame:
        """Information on the nuclei in frame as a pd.DataFrame."""

        props = [
            "label",
            "area",
            "centroid",
            "intensity_max",
            "intensity_mean",
            "intensity_min",
        ]

        return imops.skimage_props(self._nuclei_mask, self._raw_frame, props)

    @property
    def stn_ratio(self) -> float:
        """Return the signal-to-noise ratio of the frame.

        The signal-to-noise ratio is computed as the ratio between the mean
        pixel intensity within the mask (signal) and the mean pixel intensity
        outside the mask (noise).
        """

        mask = self.get_image("mask").astype(bool)
        mean_signal = self._raw_frame[mask].mean()
        mean_noise = self._raw_frame[~mask].mean()

        return mean_signal / mean_noise

    def get_image(self, modality: str) -> np.ndarray:
        """Fetch the image according to the specified modality."""

        match modality:
            case "raw":
                img = self._raw_frame.copy()
            case "mask":
                img = self._nuclei_mask.copy()
                img[img > 0] = 1
            case "outlined":
                img = imops.add_outline(self._raw_frame, self._nuclei_mask, "orange")
            case "labels":
                img = self._nuclei_mask.copy()
            case _:
                raise ValueError("Invalid modality")

        return img

    def get_nuclei(self) -> list[Nucleus]:
        """Placeholder"""

        # TODO: This way of doing it might cause label swap, probably better to
        # store the nuclei as a list of Nucleus instances and add label propery.
        return [Nucleus(n, self._raw_frame) for n in imops.get_blobs(self._nuclei_mask)]
