"""Placeholder"""

from typing import Iterable

import numpy as np
import PIL


class TiffIterable:
    """Placeholder"""

    def __init__(self, path: str):
        self._path = path
        self._tif_handle = PIL.Image.open(path)
        self._num_frames = self._tif_handle.n_frames
        self._current_frame = 0

    def __iter__(self) -> Iterable[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:

        if self._current_frame < self._num_frames:
            self._tif_handle.seek(self._current_frame)
            frame = np.array(self._tif_handle)  # TODO: 3D?
            self._current_frame += 1
            return frame

        self._tif_handle.close()
        raise StopIteration
