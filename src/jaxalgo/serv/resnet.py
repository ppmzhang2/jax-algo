import click

from jaxalgo.resnet.runner import trainer


@click.command()
@click.option("--n-epoch", type=click.INT, required=True)
def train_resnet(n_epoch: int) -> None:
    return trainer(n_epoch)
