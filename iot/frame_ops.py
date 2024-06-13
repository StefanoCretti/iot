"""Placeholder"""

from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

if TYPE_CHECKING:
    from iot.dynamics import DynamFrame


def rm_excess_labels(frame_a: "DynamFrame", frame_b: "DynamFrame", dist_thr) -> None:
    """Remove labels from b if the closes centroid in a is above a threshold."""

    cols_centr = ["centroid-0", "centroid-1"]
    centr_a = frame_a.cell_info[cols_centr]
    centr_b = frame_b.cell_info[cols_centr]

    labels = frame_b.cell_info["label"].values
    distances = np.min(sp.spatial.distance.cdist(centr_a, centr_b), axis=0)
    to_discard = [l for l, d in zip(labels, distances) if d > dist_thr]
    mapper = {a: 0 if a in to_discard else a for a in labels}

    frame_b.rename_labels(mapper)


def match_labels(frame_a: "DynamFrame", frame_b: "DynamFrame") -> None:
    """Placeholder"""

    cols_centr = ["centroid-0", "centroid-1"]
    centr_a = frame_a.cell_info[cols_centr]
    centr_b = frame_b.cell_info[cols_centr]

    min_pos = np.argmin(sp.spatial.distance.cdist(centr_a, centr_b), axis=0)

    old_labels = frame_b.cell_info["label"].values
    new_labels = frame_a.cell_info["label"][min_pos].values

    frame_b.rename_labels(dict(zip(old_labels, new_labels)))


def fix_labels(frames: list["DynamFrame"], dist_thr: int) -> None:
    """Placeholder"""

    # Forward remove excess labels
    for i in range(1, len(frames)):
        rm_excess_labels(frames[i - 1], frames[i], dist_thr)

    # Backward remove excess labels
    for i in range(len(frames) - 2, -1, -1):
        rm_excess_labels(frames[i + 1], frames[i], dist_thr)

    # Reset first frame labels from 0 to n
    frames[0].reset_labels()

    # Match labels between frames
    for i in range(1, len(frames)):
        match_labels(frames[i - 1], frames[i])
