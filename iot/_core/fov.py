"""Placeholder"""

import os
import pathlib
from typing import Iterable

import numpy as np
import pandas as pd
import PIL

from .._utils import image_ops as imops
from .._utils.io import TiffIterable
from ..plotting import plot_frame

from .pos_mask import masks_from_frame

# from .frame import Frame
from .nucleus import Nucleus
from ._process_frame import process_frame  # TODO: move
from .pos_mask import PosMask
from tqdm.auto import tqdm


class Fov:
    """Single field of view (FOV), possibly with multiple frames.

    This object can be used to process microscopy data stored in a tiff file.
    In the case of a time-lapse, different time points are different frames
    stored in the same file. Information can be then fetched in different
    formats (image or tabular).

    Parameters
    ----------
    tif_path : str
        Path to the .tiff file in string form.
    """

    def __init__(self, tif_path: str, opts: dict, pbar_pos: int = 0):

        self._path: str = tif_path
        self._pbar_pos: int = pbar_pos
        self._nuclei: list["Nucleus"] = self._get_nuclei(opts)

    @property
    def name(self) -> str:
        """Placeholder"""
        return pathlib.Path(self._path).name

    @property
    def condition(self) -> str | None:
        """Placeholder"""
        return self._condition

    @condition.setter
    def condition(self, value: str | None) -> None:
        """Placeholder"""
        self._condition = value

    @property
    def path(self) -> str:
        """Placeholder"""
        return self._path

    @property
    def num_frames(self) -> int:
        """Placeholder"""
        return len(TiffIterable(self._path))

    @property
    def shape(self) -> tuple[int, int]:
        """Placeholder"""
        return next(TiffIterable(self._path)).shape

    @property
    def nuclei_info(self) -> pd.DataFrame:
        """Placeholder"""

        parts = [nucleus.info for nucleus in self._nuclei]
        parts = [p.assign(frame=list(range(self.num_frames))) for p in parts]
        infos = pd.concat(parts, ignore_index=True)
        infos["file"] = self._path
        infos["condition"] = self.condition
        # infos["id"] = list(range(len(infos)))

        return infos

    @property
    def stn_ratios(self) -> pd.DataFrame:
        """Placeholder"""

        raw_frames = self.get_frames("raw")
        mask_frames = self.get_frames("mask")

        stns = []
        for raw, mask in zip(raw_frames, mask_frames):
            mean_signal = raw[mask].mean()
            mean_noise = raw[~mask].mean()
            stns.append(mean_signal / mean_noise)

        stn_dataf = pd.DataFrame()
        stn_dataf["stn_ratio"] = stns
        stn_dataf["frame"] = [f + 1 for f in range(self.num_frames)]
        stn_dataf["file"] = self._path

        return stn_dataf

    def _get_nuclei(self, opts: dict) -> list["Nucleus"]:
        """Placeholder"""

        tiff_frames = TiffIterable(self._path)
        first_frame = process_frame(next(tiff_frames), opts["gamma_init"], opts)
        nuclei = [Nucleus(pm, self, i + 1) for i, pm in enumerate(first_frame)]

        for raw_frame in tqdm(
            tiff_frames,
            total=self.num_frames,
            initial=1,
            desc=self.path,
            position=self._pbar_pos,
            leave=False,
        ):

            frames_cache: dict[float, list["PosMask"]] = {}

            for nucleus in nuclei:

                gamma = opts["gamma_init"]
                ref = nucleus.masks[-1]

                while True:

                    # Get the frame processed with the correct parameters
                    # Compute it if missing, else fetch from the cache
                    if gamma not in frames_cache:
                        frames_cache[gamma] = process_frame(raw_frame, gamma, opts)
                        masks = frames_cache.get(gamma)

                        success = ref.best_match(
                            masks,
                            opts["cell_movement_thr"],
                            opts["cell_match_frac"],
                        )

                    if success:
                        nucleus.add_mask(success)
                        break

                    gamma += opts["gamma_step"]

                    if gamma > opts["gamma_max"]:
                        nucleus.add_mask(ref)
                        break

        return nuclei

    def save_gif(self, path: str, modality: str, frame_duration: int) -> None:
        """Placeholder"""

        frames = self.get_frames(modality)
        frames = np.stack([imops.to_pillow_range(a) for a in frames])
        images = [PIL.Image.fromarray(a_frame) for a_frame in frames]
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=frame_duration,
        )

    def save_frames(self, folder: str, modality: str) -> None:
        """Placeholder"""

        os.makedirs(folder, exist_ok=True)
        for ind, frame in enumerate(self.get_frames(modality)):
            plot_frame(frame, f"{folder}/frame_{ind}.png")

    #  @property
    # def nuclei_info(self) -> pd.DataFrame:
    #     """Information on the nuclei in frame as a pd.DataFrame."""

    #     props = [
    #         "label",
    #         "area",
    #         "centroid",
    #         "intensity_max",
    #         "intensity_mean",
    #         "intensity_min",
    #     ]

    #     return imops.skimage_props(self._nuclei_mask, self._raw_frame, props)

    # @property
    # def stn_ratio(self) -> float:
    #     """Return the signal-to-noise ratio of the frame.

    #     The signal-to-noise ratio is computed as the ratio between the mean
    #     pixel intensity within the mask (signal) and the mean pixel intensity
    #     outside the mask (noise).
    #     """

    #     mask = self.get_image("mask").astype(bool)
    #     mean_signal = self._raw_frame[mask].mean()
    #     mean_noise = self._raw_frame[~mask].mean()

    #     return mean_signal / mean_noise

    def get_frames(self, modality: str) -> Iterable[np.ndarray]:
        """Fetch the image according to the specified modality."""

        match modality:
            case "raw":
                frames = tuple(TiffIterable(self._path))

            case "labels":
                frames = []
                bounds = [0, 0, *self.shape]

                for i in range(self.num_frames):

                    mask = np.zeros(self.shape)
                    for nucleus in self._nuclei:
                        part = nucleus.masks[i].bbox_project(bounds)
                        mask[part > 0] = nucleus.label

                    frames.append(mask)

            case "mask":
                frames = self.get_frames("labels")
                frames = [f > 0 for f in frames]

            case "outlined":
                color = "orange"
                masks = self.get_frames("labels")
                raws = self.get_frames("raw")
                frames = [imops.add_outline(r, m, color) for r, m in zip(raws, masks)]

            case _:
                raise ValueError("Invalid modality")

        return frames
