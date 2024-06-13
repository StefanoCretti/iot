"""Placeholder"""

import click

from . import cli


@cli.command()
@click.argument("pos", type=str)
def settings(pos):
    """Return an editable copy of the default settings file."""

    pass
