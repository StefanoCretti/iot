"""Placeholder"""

import os
import sys
from typing import Any

import pandas as pd
from rich import print as rprint


def check_input(input_folder: str) -> None:
    """Check for the correct structure of the input folder.

    A valid input folder must satisfy the following conditions:

    - At least 1 subfolder.
    - At least 1 file per subfolder.
    - The main folder must not contain any file.
    """

    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]
    if len(subfolders) < 1:
        raise_err_msg("Input folder must contain at least 1 subfolder.")

    for subfolder in subfolders:
        files = [f.name for f in os.scandir(subfolder) if f.is_file()]
        if len(files) < 1:
            raise_err_msg(f"Subfolder {subfolder} must contain at least 1 file.")

    main_folder_files = [f.name for f in os.scandir(input_folder) if f.is_file()]
    if len(main_folder_files) > 0:
        raise_err_msg("Main folder must not contain any file.")


def check_output(output_folder: str, force: bool) -> None:
    """Check if the output folder exists and if it should be overwritten."""

    if os.path.exists(output_folder):
        if not force:
            raise_warn_msg("Output folder already exists. Use -f to overwrite.")
    else:
        os.makedirs(output_folder)


def raise_err_msg(msg: str) -> None:
    """Print an error message and raise a SystemExit exception."""

    rprint(f"[bold red]ERROR:[/bold red] {msg}")
    sys.exit(1)


def raise_warn_msg(msg: str) -> None:
    """Print a warning message and raise a SystemExit exception."""

    rprint(f"[bold yellow]WARNING:[/bold yellow] {msg}")
    sys.exit(1)


def read_opts(opts_path: str) -> dict[str, Any]:
    """Placeholder"""

    with open(opts_path, encoding="utf8") as file:
        lines = [l for l in file.readlines() if not l.startswith("#")]
        lines = [l.split(" = ") for l in lines if " = " in l]
        opts = {k: eval(v) for k, v in lines}  # pylint: disable=eval-used

    return opts


def get_stat_cols(df: pd.DataFrame) -> list[str]:
    """Placeholder"""

    if "stn_ratio" in df.columns:
        return ["stn_ratio"]
    return ["intensity_mean", "area"]


def get_hue_col(df: pd.DataFrame) -> str:
    """Placeholder"""

    if df.condition.nunique() > 1:
        return "condition"

    if "stn_ratio" in df.columns:
        return "fov"

    return "id"
