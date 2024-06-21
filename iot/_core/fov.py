"""Placeholder"""

import os
import pathlib

import numpy as np
import pandas as pd
import PIL

from .._utils import image_ops as imops
from .._utils.io import TiffIterable
from ..plotting import plot_frame
from .frame import Frame
from .nucleus import NucleiSeq
from ._process_frame import process_frame  # TODO: move


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

    def __init__(self, tif_path: str, opts: dict, condition: str | None = None):

        self._path: str = tif_path
        self._frames: list[Frame] = self._load_frames(opts)
        self._condition: str | None = condition
        self._name: str = pathlib.Path(tif_path).name

    @property
    def name(self) -> str:
        """Placeholder"""
        return self._name

    @property
    def condition(self) -> str | None:
        """Placeholder"""
        return self._condition

    @property
    def path(self) -> str:
        """Placeholder"""
        return self._path

    @property
    def nuclei_info(self) -> pd.DataFrame:
        """Placeholder"""

        datafs = [f.nuclei_info for f in self._frames]
        datafs = [df.assign(frame=ind) for ind, df in enumerate(datafs)]
        nuclei_dataf = pd.concat(datafs)
        nuclei_dataf["file"] = os.path.basename(self._path)
        nuclei_dataf["condition"] = self._condition

        return nuclei_dataf

    @property
    def stn_ratios(self) -> pd.DataFrame:
        """Placeholder"""

        stn_dataf = pd.DataFrame()
        stn_dataf["stn_ratio"] = [f.stn_ratio for f in self._frames]
        stn_dataf["frame"] = range(len(self._frames))
        stn_dataf["file"] = os.path.basename(self._path)

        return stn_dataf

    def _load_frames(self, opts: dict) -> list[Frame]:
        """Placeholder"""

        sequences = []

        for num, raw in enumerate(TiffIterable(self._path)):

            # TODO: Remove
            print(f"Processing frame {num}")

            # If it is the first frame, simply add the found nuclei to the list
            if num == 0:
                gamma_init = opts["gamma_init"]
                frame = Frame(raw, process_frame(raw, gamma_init, opts))
                sequences = [NucleiSeq(f, gamma_init) for f in frame.get_nuclei()]
                continue

            frames: dict[float, "Frame"] = {}
            for i in reversed(range(len(sequences))):

                seq = sequences[i]
                gamma = opts["gamma_init"]

                while True:

                    # Get the frame processed with the correct parameters
                    # Compute it if missing, else fetch from the cache
                    if gamma not in frames:
                        frames[gamma] = Frame(raw, process_frame(raw, gamma, opts))
                    frame = frames.get(gamma)

                    if nuclei := frame.get_nuclei():
                        success = seq.find_match(
                            nuclei,
                            opts["cell_movement_thr"],
                            opts["cell_match_frac"],
                        )
                    else:
                        break

                    if success:
                        seq.add_nucleus(success, gamma)
                        break

                    gamma += opts["gamma_step"]

                    if gamma > opts["gamma_max"]:
                        seq.add_nucleus(seq.last_nucleus, seq.last_gamma)
                        break

        nuclei_iterator = zip(*[seq.nuclei for seq in sequences])
        frames = [
            Frame.from_nuclei(raw, nuclei)
            for raw, nuclei in zip(TiffIterable(self._path), nuclei_iterator)
        ]
        return frames

    def save_gif(self, path: str, modality: str, frame_duration: int) -> None:
        """Placeholder"""

        frames = [f.get_image(modality) for f in self._frames]
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
        for ind, frame in enumerate(self._frames):
            img = frame.get_image(modality)
            plot_frame(img, f"{folder}/frame_{ind}.png")
