"""Module for the Experiment class.

Implements the Experiment class, used to handle multiple fields of view (FOVs)
from a single experiment. The class provides methods to extract information
from the experiment as a whole.
"""

import multiprocessing
import os
import pathlib

import pandas as pd

from .._typing import OptsDict
from .fov import Fov


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
    cores : int, optional
        Number of cores to use for parallel processing. If not provided,
        the number of cores is set to the number of available cores minus 2.
    """

    def __init__(self, data_folder: str, opts: OptsDict, cores: int | None = None):

        path = pathlib.Path(data_folder)
        cores = cores or os.cpu_count() - 2
        files = [str(file) for file in path.rglob("*") if file.is_file()]
        args = [(files[i], path, opts, i) for i in range(len(files))]

        print("Starting to load FOVs:")
        with multiprocessing.Pool(cores) as p:
            fovs = p.starmap(self._create_fov, args)
        print("\nFinished loading FOVs.")  # tqdm does not add newline at the end

        self._data = data_folder
        self._fovs = fovs

    def _create_fov(self, fov, path, opts, counter):
        """Load a Fov object. Separate for multiprocessing purposes."""

        new_fov = Fov(fov, opts, pbar_pos=counter)

        # If the file is placed in a folder, set the folder as condition
        condition = str(pathlib.Path(fov).relative_to(path).parents[0])
        condition = condition if condition != "." else None
        new_fov.condition = condition

        return new_fov

    @property
    def data_folder(self) -> str:
        """Get the path to the data folder."""
        return self._data

    @property
    def nuclei_info(self) -> pd.DataFrame:
        """Get information about all nuclei in the experiment in a DataFrame."""

        nuclei_info = pd.concat([fov.nuclei_info for fov in self._fovs])
        nuclei_info.reset_index(drop=True, inplace=True)
        # nuclei_info["id"] = range(len(nuclei_info))

        return nuclei_info

    @property
    def stn_ratios(self) -> pd.DataFrame:
        """Get the signal-to-noise ratios (overtime) for all FOVs in a DataFrame."""

        stn_ratios = pd.concat([fov.stn_ratios for fov in self._fovs])
        stn_ratios.reset_index(drop=True, inplace=True)
        # stn_ratios["id"] = stn_ratios.groupby(["file"]).ngroup()

        return stn_ratios

    def get_fovs(self, condition: str | None = None) -> list[Fov]:
        """Get an iterable of FOVs from the experiment.

        Parameters
        ----------
        condition : str, optional
            Only return FOVs from this condition. If not provided, return all FOVs.

        Returns
        -------
        list[Fov]
            List of FOVs from the experiment.
        """

        if condition:
            return [fov for fov in self._fovs if fov.condition == condition]

        return self._fovs

    def save_gifs(self, folder: str, modality: str, frame_duration: int) -> None:
        """Save a gif for each FOV in the experiment.

        Parameters
        ----------
        folder : str
            Path to the folder where the gifs will be saved.
        modality : str
            What to display in each frame of the gif. Options are "raw", "mask",
            "labels", "outlined".
        frame_duration : int
            Duration of each frame in the gif in milliseconds.
        """

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
        """Save frames for each FOV in the experiment in individual folders.

        For each Fov in the experiment, the frames are saved in a subfolder
        within the provided folde.

        Parameters
        ----------
        folder : str
            Path to the folder where the frames will be saved.
        modality : str
            What to display in each frame of the gif. Options are "raw", "mask",
            "labels", "outlined".
        """

        os.makedirs(folder, exist_ok=True)
        for cond in self._fovs:
            cond.save_frames(f"{folder}/{cond.name}", modality)
