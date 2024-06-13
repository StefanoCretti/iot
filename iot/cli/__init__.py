"""Command line interface to mainstream the most common tasks for IOT."""

import click

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """Command line interface for Intensity Over Time (IOT)."""


from . import (
    plot,
    process,
    settings,
)
