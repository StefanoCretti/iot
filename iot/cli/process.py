"""Placeholder"""

import os

import click

from ..dynamics import DynamFull
from . import cli
from ._utils import check_input, check_output, read_opts


DEFAULT_SETTINGS = "settings.txt"


@cli.command()
@click.argument("data", type=str, required=True)
@click.option(
    "-o",
    "--output",
    type=str,
    default="",
    help="Output folder (default: working directory).",
)
@click.option(
    "-s",
    "--settings",
    type=str,
    default=DEFAULT_SETTINGS,
    help="Settings file (change for custom ones).",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    default=False,
    help="Force file overwrite.",
)
@click.option(
    "-d",
    "--debug",
    is_flag=True,
    default=False,
    help="Generate debug files.",
)
def process(data, output, settings, force, debug):
    """Perform basic processing of the data in the DATA folder."""

    output = output or os.path.join(os.getcwd(), "results")

    check_input(data)
    check_output(output, force)

    opts = read_opts(settings)
    opts["debug"] = debug

    dyn = DynamFull(data, opts)
    dyn.cell_info.to_csv(os.path.join(output, "cell_info.csv"), index=False)
    dyn.stn_ratios.to_csv(os.path.join(output, "stn_ratios.csv"), index=False)
    dyn.save_gifs(os.path.join(output, "gifs"), "outlined", 500)
    dyn.save_frames(os.path.join(output, "frames"), "outlined")
