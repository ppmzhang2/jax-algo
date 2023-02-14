"""Manipulate diagonal boxes.

shape: [..., M], where M >= 6:
    - M = 6 for ground truth label;
    - M > 6 for model prediction with class logit e.g. M = 86 if N_CLASS = 80

format: (x_min, y_min, x_max, y_max, [conf], classid, [logit_1, logit_2, ...])
"""
import cv2
import jax.numpy as jnp
import numpy as np

from jaxalgo.datasets.coco.const import COCO_CATE
from jaxalgo.datasets.coco.const import YOLO_IN_PX

RGB_RED = (255, 0, 0)
RGB_WHITE = (255, 255, 255)
BOX_THICKNESS = 1  # an integer
BOX_FONTSCALE = 0.35  # a float
TXT_THICKNESS = 1
FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = 0.35

# from predicted class serial number to class name
CATE_MAP = {sn: name for sn, _, name in COCO_CATE}


def pmax(dbox: jnp.ndarray) -> jnp.ndarray:
    """Get bottom-right point from a diagonal box."""
    return dbox[..., 2:4]


def pmin(dbox: jnp.ndarray) -> jnp.ndarray:
    """Get top-left point from a diagonal box."""
    return dbox[..., 0:2]


def interarea(dbox_pred: jnp.ndarray, dbox_label: jnp.ndarray) -> jnp.ndarray:
    """Get intersection area of two Diagonal boxes.

    Returns:
        jnp.ndarray: intersection area tensor with the same shape of the input
        only without the last rank
    """
    left_ups = jnp.maximum(pmin(dbox_pred), pmin(dbox_label))
    right_downs = jnp.minimum(pmax(dbox_pred), pmax(dbox_label))

    inter = jnp.maximum(right_downs - left_ups, 0.0)
    return jnp.multiply(inter[..., 0], inter[..., 1])


def rel2act(dbox: jnp.ndarray) -> jnp.ndarray:
    """Convert relative represented dbox into actual size."""
    n_coord = 4  # xmin, ymin, xmax, ymax
    *shape_pre, rankz = dbox.shape  # shape prefix and the last rank
    coef = jnp.concatenate(
        [
            jnp.tile(np.array(YOLO_IN_PX), (*shape_pre, n_coord)),
            jnp.ones((*shape_pre, rankz - n_coord))
        ],
        axis=-1,
    )
    return coef * dbox


def img_add_box(img: np.ndarray, dboxes: np.ndarray) -> np.ndarray:
    """Add bounding boxes to an image array.

    TODO: add confidence

    Args:
        img (np.ndarray): image NumPy array
        dboxes (np.ndarray): diagonal boxes array of shape (N_BOX, 6)

    Returns:
        np.ndarray: image NumPy array with bounding boxes added
    """
    for xmin, ymin, xmax, ymax, _, cls_id in dboxes.astype(
            np.int32)[..., :6].reshape(-1, 6):
        class_name = CATE_MAP[int(cls_id)]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      color=RGB_RED,
                      thickness=BOX_THICKNESS)

        (text_width, text_height), _ = cv2.getTextSize(class_name, FONTFACE,
                                                       FONTSCALE,
                                                       TXT_THICKNESS)
        cv2.rectangle(img, (xmin, ymin - int(1.3 * text_height)),
                      (xmin + text_width, ymin), RGB_RED, -1)
        cv2.putText(
            img,
            text=class_name,
            org=(xmin, ymin - int(0.3 * text_height)),
            fontFace=FONTFACE,
            fontScale=FONTSCALE,
            color=RGB_WHITE,
            lineType=cv2.LINE_AA,
        )
    return img.astype(np.uint8)  # for Image.fromarray
