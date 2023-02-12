"""All commands here."""
import json
import logging

import click

from jaxalgo.datasets import CocoAnnotation
from jaxalgo.yolov3 import runner

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--train", type=click.BOOL, required=True)
def db_reset(train: bool) -> None:
    return CocoAnnotation.db_reset(train)


@click.command()
@click.option("--in-json", type=click.STRING, required=True)
@click.option("--img-folder", type=click.STRING, required=True)
@click.option("--imgtag-csv", type=click.STRING, required=True)
@click.option("--cate-csv", type=click.STRING, required=True)
@click.option("--box-csv", type=click.STRING, required=True)
def coco_annot_to_csv(
    in_json: str,
    img_folder: str,
    imgtag_csv: str,
    cate_csv: str,
    box_csv: str,
):
    with open(in_json) as f:
        data_json = json.load(f)
    CocoAnnotation.imgtag2csv(data_json["images"], imgtag_csv, img_folder)
    CocoAnnotation.cate2csv(data_json["categories"], cate_csv)
    CocoAnnotation.box2csv(data_json["annotations"], box_csv)


@click.command()
@click.option("--imgtag-csv", type=click.STRING, required=True)
@click.option("--cate-csv", type=click.STRING, required=True)
@click.option("--box-csv", type=click.STRING, required=True)
@click.option("--train", type=click.BOOL, required=True)
def load_coco_annot(
    imgtag_csv: str,
    cate_csv: str,
    box_csv: str,
    train: bool,
) -> None:
    return CocoAnnotation.load_annot_csv(imgtag_csv, cate_csv, box_csv, train)


@click.command()
@click.option("--train", type=click.BOOL, required=True)
def create_labels(train: bool) -> None:
    return CocoAnnotation.create_labels(train)


@click.command()
@click.option("--seed", type=click.INT, required=True)
@click.option("--n-epoch", type=click.INT, required=True)
@click.option("--lr", type=click.FLOAT, required=True, help="learning rate")
@click.option("--batch-train", type=click.INT, required=True)
@click.option("--batch-valid", type=click.INT, required=True)
@click.option("--eval-span", type=click.INT, required=True)
@click.option("--eval-loop", type=click.INT, required=True)
def train(
    seed: int,
    n_epoch: int,
    lr: float,
    batch_train: int,
    batch_valid: int,
    eval_span: int,
    eval_loop: int,
) -> None:
    return runner.train(seed, n_epoch, lr, batch_train, batch_valid, eval_span,
                        eval_loop)


@click.command()
@click.option("--path-params", type=click.STRING, required=True)
@click.option("--path-states", type=click.STRING, required=True)
@click.option("--seed", type=click.INT, required=True)
@click.option("--n-epoch", type=click.INT, required=True)
@click.option("--lr", type=click.FLOAT, required=True, help="learning rate")
@click.option("--batch-train", type=click.INT, required=True)
@click.option("--batch-valid", type=click.INT, required=True)
@click.option("--eval-span", type=click.INT, required=True)
@click.option("--eval-loop", type=click.INT, required=True)
def tuning(
    path_params: str,
    path_states: str,
    seed: int,
    n_epoch: int,
    lr: float,
    batch_train: int,
    batch_valid: int,
    eval_span: int,
    eval_loop: int,
) -> None:
    return runner.tuning(path_params, path_states, seed, n_epoch, lr,
                         batch_train, batch_valid, eval_span, eval_loop)
