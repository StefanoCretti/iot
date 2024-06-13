"""Placeholder"""

import os

import click
import pandas

from . import cli
from ..plotting import plot_stats_overtime
from ._utils import get_stat_cols, get_hue_col


@cli.command()
@click.argument("stats", type=str)
@click.option(
    "-o",
    "--output",
    type=str,
    default="",
    help="Output folder (default: working directory).",
)
@click.option(
    "-c",
    "--columns",
    type=str,
    multiple=True,
    help="Columns to plot (default: auto).",
)
@click.option(
    "-f",
    "--fraction",
    type=bool,
    default=True,
    help="Normalize wrt tp zero (default: True).",
)
@click.option(
    "-H",
    "--hue",
    type=str,
    help="Column for color encoding (default: auto).",
)
def plot(stats, output, columns, fraction, hue):
    """Create over-time plot of some columns in the STATS file."""

    dataf = pandas.read_csv(stats)
    output = output or os.path.join(os.getcwd(), "plot.png")

    columns = columns or get_stat_cols(dataf)
    hue = hue or get_hue_col(dataf)

    plot_stats_overtime(dataf, columns, output, fraction, hue)
