"""Placeholder"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_stats_overtime(
    df: pd.DataFrame,
    col: list[str],
    path: str | None = None,
    as_fraction: bool = False,
    hue: str | None = None,
) -> None:
    """Placeholder"""

    data = df.copy()

    # Normalize on initial tp
    if as_fraction:
        init_cond = data.query("frame == 0")
        data = data.merge(init_cond, on="id", suffixes=("", "_init"))
        for stat in col:
            data[stat] = data[stat] / data[f"{stat}_init"]
        data = data.drop(columns=[c for c in data.columns if c.endswith("_init")])

    _, axes = plt.subplots(len(col), 1, figsize=(8, 3 * len(col)), sharex=True)
    axes = [axes] if len(col) == 1 else axes

    for pos, stat in enumerate(col):
        sns.lineplot(
            data=data,
            x="frame",
            y=stat,
            hue=hue,
            ax=axes[pos],
            units="id",
            estimator=None,
        )
        if pos != 0:
            axes[pos].get_legend().remove()
        axes[pos].set(xlim=(0, data["frame"].max()))

    if path:
        plt.tight_layout()
        plt.savefig(path, dpi=300)
    else:
        plt.show()

    # plt.imshow(frame.get_image(modality))
    # for row in frame.cell_info.iterrows():
    #     label = row[1]["label"]
    #     centroid = row[1]["centroid-0"], row[1]["centroid-1"]
    #     plt.text(centroid[1], centroid[0], str(label), color="red")
    # plt.savefig(f"{path}/frame_{num}.png", dpi=300)
    # plt.close()

    # def save_frames(self, path: str, modality: str) -> None:
    #     """Placeholder"""

    # for num, frame in enumerate(self._frames):
    #     plt.imshow(frame.get_image(modality))
    #     for row in frame.cell_info.iterrows():
    #         label = row[1]["label"]
    #         centroid = row[1]["centroid-0"], row[1]["centroid-1"]
    #         plt.text(centroid[1], centroid[0], str(int(label)), color="red")
    #     plt.savefig(f"{path}/frame_{num}.png", dpi=300)
    #     plt.close()

    # frames = [Image.fromarray(f.get_image(modality)) for f in self._frames]
    # frames[0].save(
    #     path,
    #     save_all=True,
    #     append_images=frames[1:],
    #     duration=frame_duration,
    #     loop=0,
    # )


def plot_frame(img: np.ndarray, path: str) -> None:
    """Placeholder"""

    plt.imshow(img)
    plt.savefig(path, dpi=300)
    plt.close()
