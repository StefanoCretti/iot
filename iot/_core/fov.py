"""Module for the Field of View (FOV) class.

Implements the FOV class, used to handle a single field of view from a microscopy
experiment (one .tiff file with multiple frames, one for each time point).
The class provides methods to extract information from the FOV as a whole.
"""

import os
import pathlib
from typing import Iterable

import numpy as np
import pandas as pd
import PIL
import tqdm

from .._utils import image_ops as imops
from .._utils.io import TiffIterable
from ..plotting import plot_frame
from ._process_frame import process_frame
from .nucleus import Nucleus
from .pos_mask import PosMask


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
    opts : dict
        Dictionary with the options for the analysis.
    pbar_pos : int, optional
        Position of the progress bar in the terminal. Mostly used when loading
         multiple Fov objects using multiprocessing. Default is 0.
    """

    def __init__(self, tif_path: str, opts: dict, pbar_pos: int = 0):

        self._path: str = tif_path
        self._pbar_pos: int = pbar_pos
        self._nuclei: list["Nucleus"] = self._get_nuclei(opts)

    @property
    def name(self) -> str:
        """Name of the FOV, extracted from the path."""
        return pathlib.Path(self._path).name

    @property
    def condition(self) -> str | None:
        """Condition of the FOV (by default subfolder name, if any)."""
        return self._condition

    @condition.setter
    def condition(self, value: str | None) -> None:
        """Overwrite the condition of the FOV.

        Parameters
        ----------
        value : str
            New condition for the FOV.
        """
        self._condition = value

    @property
    def path(self) -> str:
        """Full path to the FOV file."""
        return self._path

    @property
    def num_frames(self) -> int:
        """Number of frames in the FOV."""
        return len(TiffIterable(self._path))

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the FOV frames."""
        return next(TiffIterable(self._path)).shape

    @property
    def nuclei_info(self) -> pd.DataFrame:
        """Tabular information about all nuclei in the FOV at each time point."""

        parts = [nucleus.info for nucleus in self._nuclei]
        parts = [p.assign(frame=list(range(self.num_frames))) for p in parts]
        infos = pd.concat(parts, ignore_index=True)
        infos["file"] = self._path
        infos["condition"] = self.condition
        # infos["id"] = list(range(len(infos)))

        return infos

    @property
    def stn_ratios(self) -> pd.DataFrame:
        """Signal-to-noise ratios (STN) for the FOV at each time point.

        The signal-to-noise ratio is computed as the ratio between the mean
        signal in all masked object and the mean noise in the background.
        """

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
        """Extract nuclei from the FOV using the specified options."""

        tiff_frames = TiffIterable(self._path)
        first_frame = process_frame(next(tiff_frames), opts["gamma_init"], opts)
        nuclei = [Nucleus(pm, self, i + 1) for i, pm in enumerate(first_frame)]

        for raw_frame in tqdm.tqdm(
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
        """Save a gif with the frames of the FOV.

        Parameters
        ----------
        path : str
            Path to the output gif file.
        modality : str
            What to display in each frame of the gif. Can be "raw", "labels",
            "mask", or "outlined".
        frame_duration : int
            Duration of each frame in the gif in milliseconds.
        """

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
        """Save individual frames of the FOV in a folder.

        Parameters
        ----------
        folder : str
            Path to the folder where the frames will be saved.
        modality : str
            What to display in each frame of the gif. Can be "raw", "labels",
            "mask", or "outlined".
        """

        os.makedirs(folder, exist_ok=True)
        for ind, frame in enumerate(self.get_frames(modality)):
            plot_frame(frame, f"{folder}/frame_{ind}.png")

    def get_frames(self, modality: str) -> Iterable[np.ndarray]:
        """Fetch the individual frames according to the specified modality.

        Return an iterable of frames as numpy arrays. What is shown in each frame
        depends on the modality selected. Available modalities are:
        - "raw": raw signal.
        - "labels": image with the labeled nuclei masks.
        - "mask": binary image with the nuclei masks.
        - "outlined": raw signal with the nuclei outlined in a given color.

        Parameters
        ----------
        modality : str
            Modality of the frames to return.

        Returns
        -------
        Iterable[np.ndarray]
            Iterable of frames as numpy arrays.
        """

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
