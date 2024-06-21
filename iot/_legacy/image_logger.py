"""Placeholder"""

from dataclasses import dataclass
import os

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class LabeledImage:
    """Class to store an image with its corresponding label."""

    label: str
    image: np.ndarray


class ImageLogger:
    """Class to debug image processing by plotting the steps."""

    _frame_id: int = 0
    _cell_id: int = 0

    def __init__(self, log_path: str, is_frame: bool = False) -> None:
        self._imgs: list[LabeledImage] = []
        self._path: str = log_path
        self._grid_num: int = 0
        self._dpi: int = 300
        self._is_frame = is_frame

        if is_frame:
            ImageLogger._frame_id += 1
            ImageLogger._cell_id = 0
        else:
            ImageLogger._cell_id += 1

        self._frame_num = ImageLogger._frame_id
        self._cell_num = ImageLogger._cell_id

    def add_image(self, image: LabeledImage) -> None:
        """Add an image to the logger."""
        self._imgs.append(image)

    def clear_images(self) -> None:
        """Clear the images stored in the logger."""
        self._imgs = []

    def _get_file_root(self) -> str:
        """Return the root of the file name."""

        root = f"cell_{self._cell_num:04}_frame_{self._frame_num:04}"
        return os.path.join(self._path, root)

    def _plot_as_group(self):
        """Plot the saved images in a single big grid."""

        def get_grid_shape(num: int) -> np.ndarray:
            """Return shape and size of the table."""

            space = 1
            shape = np.array([1, 1])

            while space < num:
                shape[shape.argmin()] += 1
                space = shape.prod()

            return shape if shape.argmin() == 0 else shape[::-1]

        num_plots = len(self._imgs)
        num_rows, num_cols = get_grid_shape(num_plots)
        grid_space = num_rows * num_cols

        _, ax = plt.subplots(num_rows, num_cols)
        for pos in range(grid_space):

            axes = ax[pos // num_cols, pos % num_cols]
            axes.set_axis_off()

            if pos > num_plots:
                pass  # TODO: Make empty plot
            else:
                img = self._imgs[pos]
                axes.imshow(img.image)
                axes.set_title(img.label)

        root = self._get_file_root()

        plt.tight_layout()
        plt.savefig(root + ".png", dpi=self._dpi)
        plt.close()

    def _plot_as_pairs(self):
        """Plot the saved images in successive pairs."""

        for i in range(0, len(self._imgs) - 1):
            left, right = self._imgs[i], self._imgs[i + 1]

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(left.image)
            ax[1].imshow(right.image)

            fig.suptitle(right.label)
            plt.tight_layout()

            root = self._get_file_root()
            plt.savefig(root + f"_{right.label}.png", dpi=self._dpi)
            plt.close()

    def plot_images(self, modality: str):
        """Plot the images saved in the logger in the desired format."""

        if not os.path.exists(self._path):
            os.makedirs(self._path)

        match modality:
            case "group":
                self._plot_as_group()
            case "pairs":
                self._plot_as_pairs()
            case _:
                raise ValueError("Invalid modality")
