"""All commands here."""
import json
import logging

import click
import cv2
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from jaxalgo.datasets import CocoAnnotation
from jaxalgo.datasets import CocoDataset
from jaxalgo.yolov3 import runner
from jaxalgo.yolov3.box import bbox
from jaxalgo.yolov3.box import dbox
from jaxalgo.yolov3.box import pbox
from jaxalgo.yolov3.loader import load_cv_var
from jaxalgo.yolov3.nms import three2one_1img
from jaxalgo.yolov3.runner import ModelState
from jaxalgo.yolov3.runner import save_state

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--train", type=click.BOOL, required=True)
def db_reset(*, train: bool) -> None:
    """DB reset."""
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
) -> None:
    """Convert COCO json annotation to csv."""
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
    *,
    train: bool,
) -> None:
    """Load COCO annotation csv files."""
    return CocoAnnotation.load_annot_csv(imgtag_csv, cate_csv, box_csv, train)


@click.command()
@click.option("--train", type=click.BOOL, required=True)
def create_labels(*, train: bool) -> None:
    """Create COCO label table."""
    return CocoAnnotation.create_labels(train)


@click.command()
@click.option("--row-id", type=click.INT, required=True)
@click.option("--file-name", type=click.STRING, required=True)
def show_true_box(row_id: int, file_name: str) -> None:
    """Show image with ground truth boxes."""
    batch = CocoDataset(mode="TEST", batch=1).top_batch(row_id - 1)
    bx = jnp.concatenate(
        [
            bbox.objects(batch.label_s),
            bbox.objects(batch.label_m),
            bbox.objects(batch.label_l),
        ],
        axis=0,
    )
    bx = dbox.rel2act(bbox.asdbox(bx))
    img = dbox.img_add_box(np.array(batch.image[0, ...]), np.array(bx))
    return Image.fromarray(img).save(file_name)


@click.command()
@click.option("--row-id", type=click.INT, required=True)
@click.option("--file-name", type=click.STRING, required=True)
@click.option("--seed", type=click.INT, required=True)
@click.option("--params-path", type=click.STRING, required=True)
@click.option("--states-path", type=click.STRING, required=True)
@click.option("--conf-th", type=click.FLOAT, required=True)
@click.option("--iou-th", type=click.FLOAT, required=True)
def show_predict_box(  # noqa: PLR0913
    row_id: int,
    file_name: str,
    seed: int,
    params_path: str,
    states_path: str,
    conf_th: float,
    iou_th: float,
) -> None:
    """Show image with predicted boxes."""
    key = jax.random.PRNGKey(seed)
    xfm = runner.get_xfm()
    var = runner.load_state(params_path, states_path)
    batch = CocoDataset(mode="TEST", batch=1).top_batch(row_id - 1)

    seq_pbox, _ = xfm.apply(var.params, var.states, key, batch.image / 255.)
    seq_bbox = [pbox.asbbox(prd) for prd in seq_pbox]

    bx = three2one_1img(
        seq_bbox[0],
        seq_bbox[1],
        seq_bbox[2],
        conf_th=conf_th,
        iou_th=iou_th,
        from_logits=True,
    )
    if bx is None:
        LOGGER.warning("no boxes found")
        return None

    bx = dbox.rel2act(bbox.asdbox(bx))
    img = dbox.img_add_box(np.array(batch.image[0, ...]), np.array(bx))
    return Image.fromarray(img).save(file_name)


@click.command()
@click.option("--seed", type=click.INT, required=True)
@click.option("--n-epoch", type=click.INT, required=True)
@click.option("--lr", type=click.FLOAT, required=True, help="learning rate")
@click.option("--batch-train", type=click.INT, required=True)
@click.option("--batch-valid", type=click.INT, required=True)
@click.option("--eval-span", type=click.INT, required=True)
@click.option("--eval-loop", type=click.INT, required=True)
def train(  # noqa: PLR0913
    seed: int,
    n_epoch: int,
    lr: float,
    batch_train: int,
    batch_valid: int,
    eval_span: int,
    eval_loop: int,
) -> None:
    """Train a model from scratch."""
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
def tuning(  # noqa: PLR0913
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
    """Tuning an existing model."""
    return runner.tuning(path_params, path_states, seed, n_epoch, lr,
                         batch_train, batch_valid, eval_span, eval_loop)


@click.command()
@click.option("--darknet-cfg", type=click.STRING, required=True)
@click.option("--darknet-weight", type=click.STRING, required=True)
@click.option("--params-path", type=click.STRING, required=True)
@click.option("--states-path", type=click.STRING, required=True)
def load_darknet_weights(
    darknet_cfg: str,
    darknet_weight: str,
    params_path: str,
    states_path: str,
) -> None:
    """Load and save YOLOv3 darknet weights."""
    net = cv2.dnn.readNetFromDarknet(darknet_cfg, darknet_weight)
    params, states = load_cv_var(net)
    var = ModelState(params, states)
    return save_state(var, params_path, states_path)
