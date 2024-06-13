"""Placeholder"""

from dataclasses import dataclass

import numpy as np


@dataclass
class LabeledImage:
    """Class to store an image with its corresponding label."""

    label: str
    image: np.ndarray
