"""All commands here."""
import click

from jaxalgo.serv import resnet as resnet_cli
from jaxalgo.serv import yolov3 as yolov3_cli


@click.group()
def yolov3() -> None:
    """All YOLOv3 commands. Including data preparation, training model, etc."""


yolov3.add_command(yolov3_cli.coco_annot_to_csv)
yolov3.add_command(yolov3_cli.create_labels)
yolov3.add_command(yolov3_cli.db_reset)
yolov3.add_command(yolov3_cli.load_coco_annot)
yolov3.add_command(yolov3_cli.train)
yolov3.add_command(yolov3_cli.tuning)


@click.group()
def resnet() -> None:
    """All clicks here."""


resnet.add_command(resnet_cli.train_resnet)
