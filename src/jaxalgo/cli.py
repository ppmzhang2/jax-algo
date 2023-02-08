"""All commands here."""
import click

from jaxalgo.serv.resnet import train_resnet
from jaxalgo.serv.yolov3 import coco_annot_to_csv
from jaxalgo.serv.yolov3 import create_labels
from jaxalgo.serv.yolov3 import db_reset
from jaxalgo.serv.yolov3 import load_coco_annot
from jaxalgo.serv.yolov3 import train_yolo


@click.group()
def yolov3() -> None:
    """All YOLOv3 commands. Including data preparation, training model, etc."""


yolov3.add_command(coco_annot_to_csv)
yolov3.add_command(create_labels)
yolov3.add_command(db_reset)
yolov3.add_command(load_coco_annot)
yolov3.add_command(train_yolo)


@click.group()
def resnet() -> None:
    """All clicks here."""


resnet.add_command(train_resnet)
