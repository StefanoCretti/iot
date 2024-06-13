"""Placeholder"""

from __future__ import annotations

import os
from types import UnionType

import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure
import tqdm

from iot import image_ops as imops
from iot.process_frame import process_frame
from iot.frame_ops import fix_labels
from iot.plotting import plot_frame


TIF_EXT = [".tif", ".tiff"]


def get_cell_info(mask: np.ndarray, raw: np.ndarray) -> pd.DataFrame:
    """Placeholder"""

    props = [
        "label",
        "area",
        "centroid",
        "intensity_max",
        "intensity_mean",
        "intensity_min",
    ]

    label_props = measure.regionprops_table(mask, intensity_image=raw, properties=props)
    return pd.DataFrame(label_props)


class Cell:
    """Placeholder"""

    def __init__(self, mask: np.ndarray, tolerance: float = 0):

        self._bounds = imops.blob_boundaries(mask, tolerance)
        self._mask = imops.zoom_in(mask, self._bounds)
        self._full_size = mask.shape
        self._size: int = self._mask.sum()

    @property
    def bounds(self):
        """Placeholder"""
        return self._bounds

    @property
    def size(self):
        """Placeholder"""
        return self._size

    def get_mask(self, zoomed: bool = True) -> np.ndarray:
        """Placeholder"""

        if zoomed:
            return self._mask
        return imops.zoom_out(self._mask, self._bounds, np.zeros(self._full_size))

    def does_overlap(self, other: "Cell") -> bool:
        """Return whether cell bounding boxes have any intersection."""

        x_min_s, x_max_s, y_min_s, y_max_s = self._bounds
        x_min_o, x_max_o, y_min_o, y_max_o = other.bounds

        x_overlap = (x_min_o <= x_max_s <= x_max_o) or (x_min_o <= x_min_s <= x_max_o)
        y_overlap = (y_min_o <= y_max_s <= y_max_o) or (y_min_o <= y_min_s <= y_max_o)

        return x_overlap and y_overlap

    def __and__(self, other: "Cell") -> int:
        """Return amount of pixels shared by the two cells."""

        overlap = self.get_mask(zoomed=False) * other.get_mask(zoomed=False)
        return overlap.sum()

    def __or__(self, other: "Cell") -> int:
        """Placeholder"""
        inters = self.get_mask(zoomed=False) + other.get_mask(zoomed=False)
        inters[inters > 1] = 1
        return inters.sum()


class DynamFrame:
    """Individual frame of a time-lapse microscopy sequence."""

    def __init__(
        self,
        frame: np.ndarray,
        opts: dict,
        prev_frame: "DynamFrame" | None = None,
    ) -> None:
        self._raw_frame: np.ndarray = frame
        self._cell_mask: np.ndarray = self._get_mask(opts, prev_frame)
        self._cell_info: pd.DataFrame = pd.DataFrame()
        self._stn_ratio: float = np.NaN

        self._refresh_cell_info()
        self._refresh_stn_ratio()

    def _get_mask(self, opts: dict, prev_frame: "DynamFrame" | None) -> np.ndarray:
        """Placeholder"""

        options = opts.copy()

        stop_val = 2
        step_val = 0.1
        # num_vals = 1 if not prev_frame else int((stop_val - 1) / step_val) + 1
        num_vals = int((stop_val - 1) / step_val) + 1

        new_cells = []
        gammas = [1 + step_val * i for i in range(num_vals)]
        gammas = gammas if prev_frame else [1.2]
        for gamma in gammas:
            options["full_gamma_thr"] = gamma
            mask = process_frame(self._raw_frame, options)

            new_cells.extend([Cell(c) for c in imops.iterate_blobs(mask)])

        cells: list["Cell"] = []
        if prev_frame:
            prev_mask = prev_frame.get_image("labels")
            prev_cells = [Cell(c) for c in imops.iterate_blobs(prev_mask)]

            for old_cell in prev_cells:

                # Maybe add number of pixels in mask to the scoring

                best_match = None
                best_score = 0

                # print("-" * 79)

                for new_cell in new_cells:
                    if not old_cell.does_overlap(new_cell):
                        continue

                    # old_size, new_size = old_cell.size, new_cell.size
                    # size_score = (abs(old_size - new_size) + 1) / (old_size + new_size)
                    over_score = (old_cell & new_cell) / (old_cell | new_cell)
                    # print(
                    #     f"intersection: {old_cell & new_cell}, union: {old_cell | new_cell}, score{over_score}"
                    # )
                    # score = over_score - size_score
                    score = over_score

                    if score > best_score:
                        best_score = score
                        best_match = new_cell

                # Check that the score is at least a certain value
                if best_match:
                    cells.append(best_match)
        else:
            cells.extend(new_cells)

        new_frame = np.zeros_like(self._raw_frame)
        for ind, cell in enumerate(cells):
            new_frame[cell.get_mask(zoomed=False) == 1] = ind + 1

        return new_frame

        # Make all cells into a cohesive frame

    def _refresh_cell_info(self) -> None:
        """Compute cell information for the frame."""
        self._cell_info = get_cell_info(self._cell_mask, self._raw_frame)

    def _refresh_stn_ratio(self) -> None:
        """Compute the signal-to-noise ratio of the frame."""

        mask = self.get_image("mask").astype(bool)
        mean_signal = self._raw_frame[mask].mean()
        mean_noise = self._raw_frame[~mask].mean()

        self._stn_ratio = mean_signal / mean_noise

    @property
    def cell_info(self) -> pd.DataFrame:
        """Information on the cells in frame as a pd.DataFrame."""
        return self._cell_info

    @property
    def stn_ratio(self) -> float:
        """Return the signal-to-noise ratio of the frame.

        The signal-to-noise ratio is computed as the ratio between the mean
        pixel intensity within the mask (signal) and the mean pixel intensity
        outside the mask (noise).
        """
        return self._stn_ratio

    def get_image(self, modality: str) -> np.ndarray:
        """Fetch the image according to the specified modality."""

        match modality:
            case "raw":
                img = self._raw_frame.copy()
            case "mask":
                img = self._cell_mask.copy()
                img[img > 0] = 1
            case "outlined":
                img = imops.add_outline(self._raw_frame, self._cell_mask, "orange")
            case "labels":
                img = self._cell_mask.copy()
            case _:
                raise ValueError("Invalid modality")

        return img

    def reset_labels(self) -> None:
        """Placeholder"""

        mapper = {o: n for n, o in enumerate(np.unique(self._cell_mask))}
        self.rename_labels(mapper)

    def rename_labels(self, mapping: dict[int, int]) -> None:
        """Placeholder."""

        new_mask = np.zeros_like(self._cell_mask, dtype=int)
        for old, new in mapping.items():
            new_mask[self._cell_mask == old] = new

        self._cell_mask = new_mask

        self._refresh_cell_info()
        self._refresh_stn_ratio()


class DynamFov:
    """Handler for a full microscopy time-lapse.

    Object used to process microscopy time-lapse data stored in tiff-files.
    Information can be then fetched in different formats (image or tabular).

    Parameters
    ----------
    tif_path : str
        Path to the .tiff file in string form.
    """

    def __init__(self, tif_path: str, opts: dict) -> None:

        self._path: str = tif_path
        self._frames: list[DynamFrame] = self._load_frames(opts)

    def _load_frames(self, opts: dict) -> list[DynamFrame]:
        """Placeholder"""

        frames = []
        opts = opts.copy()

        tif_handle = Image.open(self._path)
        num_frames = tif_handle.n_frames

        with tqdm.tqdm(total=num_frames, desc="Processing frames") as pbar:

            for i in range(num_frames):
                tif_handle.seek(i)

                prev_frame = None if len(frames) == 0 else frames[-1]
                frame = DynamFrame(np.array(tif_handle), opts, prev_frame)
                frames.append(frame)
                # break
                pbar.update(1)

        fix_labels(frames, opts["cell_movement_thr"])
        return frames

    @property
    def cell_info(self) -> pd.DataFrame:
        """Placeholder"""

        datafs = [f.cell_info for f in self._frames]
        datafs = [df.assign(frame=ind) for ind, df in enumerate(datafs)]
        cell_dataf = pd.concat(datafs)
        cell_dataf["file"] = os.path.basename(self._path)

        return cell_dataf

    @property
    def stn_ratios(self) -> pd.DataFrame:
        """Placeholder"""

        stn_dataf = pd.DataFrame()
        stn_dataf["stn_ratio"] = [f.stn_ratio for f in self._frames]
        stn_dataf["frame"] = range(len(self._frames))
        stn_dataf["file"] = os.path.basename(self._path)

        return stn_dataf

    def save_gif(self, path: str, modality: str, frame_duration: int) -> None:
        """Placeholder"""

        frames = [f.get_image(modality) for f in self._frames]
        frames = np.stack([imops.to_pillow_range(a) for a in frames])
        images = [Image.fromarray(a_frame) for a_frame in frames]
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


class DynamCondition:
    """Placeholder"""

    def __init__(self, folder: str, opts: dict) -> None:

        self._fold: str = folder
        self._fovs: list[DynamFov] = self._load_fovs(opts)

    def _load_fovs(self, opts) -> list[DynamFov]:
        """Load the individual tif files as fovs."""

        files = [f for f in os.listdir(self._fold) if os.path.splitext(f)[1] in TIF_EXT]
        files = [os.path.join(self._fold, f) for f in files]

        fovs = []
        for num, fov in enumerate(files):
            fov_name = os.path.basename(fov)
            print(f"Processing fov {num + 1}/{len(files)}: {fov_name}")
            fovs.append(DynamFov(fov, opts))

        return fovs

    @property
    def cell_info(self) -> pd.DataFrame:
        """Placeholder"""

        datafs = [fov.cell_info for fov in self._fovs]
        datafs = [df.assign(fov=ind) for ind, df in enumerate(datafs)]
        cell_dataf = pd.concat(datafs)
        cell_dataf["condition"] = os.path.basename(self._fold)

        return cell_dataf

    @property
    def stn_ratios(self) -> pd.DataFrame:
        """Placeholder"""

        datafs = [fov.stn_ratios for fov in self._fovs]
        datafs = [df.assign(fov=ind) for ind, df in enumerate(datafs)]
        stn_dataf = pd.concat(datafs)
        stn_dataf["condition"] = os.path.basename(self._fold)

        return stn_dataf

    @property
    def name(self) -> str:
        """Placeholder"""

        return os.path.basename(self._fold)

    def save_gifs(self, folder: str, modality: str, frame_duration: int) -> None:
        """Placeholder"""

        os.makedirs(folder, exist_ok=True)
        for ind, fov in enumerate(self._fovs):
            fov.save_gif(f"{folder}/fov_{ind + 1}.gif", modality, frame_duration)

    def save_frames(self, folder: str, modality: str) -> None:
        """Placeholder"""

        os.makedirs(folder, exist_ok=True)
        for ind, fov in enumerate(self._fovs):
            fov.save_frames(f"{folder}/fov_{ind + 1}", modality)


class DynamFull:
    """Placeholder"""

    def __init__(self, root_folder, opts):
        self._root = root_folder
        self._conditions = self._load_conditions(opts)

    def _load_conditions(self, opts) -> list["DynamCondition"]:
        """Placeholder"""

        folders = [os.path.join(self._root, f) for f in os.listdir(self._root)]
        folders = [f for f in folders if os.path.isdir(f)]

        conditions = []
        for num, folder in enumerate(folders):
            folder_name = os.path.basename(folder)
            print(f"Processing condition {num + 1}/{len(folders)}: {folder_name}")
            conditions.append(DynamCondition(folder, opts))

        return conditions

    @property
    def cell_info(self) -> pd.DataFrame:
        """Placeholder"""

        cell_info = pd.concat([cond.cell_info for cond in self._conditions])
        cell_info.reset_index(drop=True, inplace=True)
        cell_info["id"] = cell_info.groupby(["label", "fov", "condition"]).ngroup()

        return cell_info

    @property
    def stn_ratios(self) -> pd.DataFrame:
        """Placeholder"""

        stn_ratios = pd.concat([cond.stn_ratios for cond in self._conditions])
        stn_ratios.reset_index(drop=True, inplace=True)
        stn_ratios["id"] = stn_ratios.groupby(["fov", "condition"]).ngroup()

        return stn_ratios

    def save_gifs(self, folder: str, modality: str, frame_duration: int) -> None:
        """Placeholder"""

        os.makedirs(folder, exist_ok=True)
        for cond in self._conditions:
            cond.save_gifs(f"{folder}/{cond.name}", modality, frame_duration)

    def save_frames(self, folder: str, modality: str) -> None:
        """Placeholder"""

        os.makedirs(folder, exist_ok=True)
        for cond in self._conditions:
            cond.save_frames(f"{folder}/{cond.name}", modality)
