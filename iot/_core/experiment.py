"""Placeholder"""

import os
import pathlib

import pandas as pd

from .fov import Fov
from .._typing import OptsDict


class Experiment:
    """Complete experiment with one or more fields of view (FOVs).

    Class mostly used for automation of the analysis of multiple FOVs.
    Individual FOVs should be .tiff files. If a condition is associated
    with the individual FOV, it should be saved in a folder with the
    following structure:

    data_folder
    ├── condition1
    │   ├── fov1.tiff
    │   ├── fov2.tiff
    │   └── ...
    ├── condition2
    │   ├── fov1.tiff
    │   ├── fov2.tiff
    │   └── ...
    └── ...

    If no condition is associated with the FOV, simply place the .tiff
    files in the root folder of the experiment.

    Parameters
    ----------
    data_folder : str
        Path to the folder containing the FOVs.
    opts : dict
        Dictionary with the options for the analysis.
    """

    def __init__(self, data_folder: str, opts: OptsDict):

        path = pathlib.Path(data_folder)
        files = [str(file) for file in path.rglob("*") if file.is_file()]

        fovs: list[Fov] = []
        for num, fov in enumerate(files):
            print(f"Loading fov {num + 1}/{len(files)}: {fov}")

            # If the file is placed in a folder, set the folder as condition
            condition = str(pathlib.Path(fov).relative_to(path).parents[0])
            condition = condition if condition != "." else None

            fovs.append(Fov(fov, opts, condition))

        self._data = data_folder
        self._fovs = fovs

    @property
    def data_folder(self) -> str:
        """Get the path to the data folder."""
        return self._data

    @property
    def nuclei_info(self) -> pd.DataFrame:
        """Get information about all nuclei in the experiment in a DataFrame."""

        nuclei_info = pd.concat([fov.nuclei_info for fov in self._fovs])
        nuclei_info.reset_index(drop=True, inplace=True)
        nuclei_info["id"] = nuclei_info.groupby(["label", "file"]).ngroup()

        return nuclei_info

    @property
    def stn_ratios(self) -> pd.DataFrame:
        """Get the signal-to-noise ratios (overtime) for all FOVs in a DataFrame."""

        stn_ratios = pd.concat([fov.stn_ratios for fov in self._fovs])
        stn_ratios.reset_index(drop=True, inplace=True)
        stn_ratios["id"] = stn_ratios.groupby(["file"]).ngroup()

        return stn_ratios

    def get_fovs(self, condition: str | None = None) -> list[Fov]:
        """Get an iterable of FOVs from the experiment."""

        if condition:
            return [fov for fov in self._fovs if fov.condition == condition]

        return self._fovs

    def save_gifs(self, folder: str, modality: str, frame_duration: int) -> None:
        """Placeholder"""

        os.makedirs(folder, exist_ok=True)

        for fov in self._fovs:
            file_name = os.path.splitext(os.path.basename(fov.file_path))[0] + ".gif"

            if fov.condition:
                condition_folder = os.path.join(folder, fov.condition)
                os.makedirs(condition_folder, exist_ok=True)
                gif_path = os.path.join(condition_folder, file_name)
            else:
                gif_path = os.path.join(folder, file_name)
            fov.save_gif(gif_path, modality, frame_duration)

    def save_frames(self, folder: str, modality: str) -> None:
        """Placeholder"""

        os.makedirs(folder, exist_ok=True)
        for cond in self._fovs:
            cond.save_frames(f"{folder}/{cond.name}", modality)
