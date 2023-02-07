"""All commands here."""
from typing import NoReturn

import click

from jaxalgo.serv import train_resnet


@click.group()
def cli() -> NoReturn:
    """All clicks here."""


cli.add_command(train_resnet)
