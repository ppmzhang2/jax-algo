"""Manipulate bounding boxes.

shape: [..., M], where M >= 10:
    - M = 10 for ground truth label;
    - M > 10 for model prediction with class logit e.g. M = 90 if N_CLASS = 80

format: (x, y, w, h, x_offset, y_offset, w_exp, h_exp, conf, class ID,
         [logit_1, logit_2, ...])
    - ground truth label:
        (x, y, w, h, x_offset, y_offset, w_exp, h_exp, conf_flag, class ID)
    - TRANSFORMED prediction:
        (x, y, w, h, x_offset, y_offset, w_exp, h_exp, conf_logit, class ID,
         class_logit_1, class_logit_2, ...)

A raw model prediction is NOT a bounding box;
transform it using functions in `pbox`
"""
import math

import jax
import jax.numpy as jnp

from jaxalgo.yolov3.box import dbox

EPSILON = 1e-3


@jax.jit
def xy(bbox: jnp.ndarray) -> jnp.ndarray:
    """From bounding box get coordinates tensor of shape (..., 2).

    Args:
        bbox (jnp.ndarray): bounding box
    """
    return bbox[..., :2]


@jax.jit
def wh(bbox: jnp.ndarray) -> jnp.ndarray:
    """From bbox get width and hieght tensor of shape (..., 2).

    Args:
        bbox (jnp.ndarray): bounding box
    """
    return bbox[..., 2:4]


@jax.jit
def xywh(bbox: jnp.ndarray) -> jnp.ndarray:
    """Get x-y coordinatestensor, width and height from a tensor.

    Args:
        bbox (jnp.ndarray): bounding box
    """
    return bbox[..., :4]


@jax.jit
def xy_offset(bbox: jnp.ndarray) -> jnp.ndarray:
    """From bounding box get center poiont coordinates offset.

    Args:
        bbox (jnp.ndarray): bounding box
    """
    return bbox[..., 4:6]


@jax.jit
def wh_exp(bbox: jnp.ndarray) -> jnp.ndarray:
    """From bbox get width and height exponent of shape (..., 2).

    Args:
        bbox (jnp.ndarray): bounding box
    """
    return bbox[..., 6:8]


@jax.jit
def conf1d(bbox: jnp.ndarray) -> jnp.ndarray:
    """Get object confidence from a tensor (squeezed).
    Suppose the number of ranks of the input tensor is R, the #rank of the
    output tensor will be R - 1 is `squeezed`.
    Otherwise the #rank of the output will remain as R, and the last
    rank contains only 1 dimension

    Args:
        bbox (jnp.ndarray): bounding box
    """
    return bbox[..., 8]


@jax.jit
def confnd(bbox: jnp.ndarray) -> jnp.ndarray:
    """Get object confidence from a tensor (un-squeezed).
    Suppose the number of ranks of the input tensor is R, the #rank of the
    output tensor will be R - 1 is `squeezed`.
    Otherwise the #rank of the output will remain as R, and the last
    rank contains only 1 dimension

    Args:
        bbox (jnp.ndarray): bounding box
    """
    return bbox[..., 8:9]


def class_sn(bbox: jnp.ndarray, *, squeezed: bool = False) -> jnp.ndarray:
    """Get class ID from a tensor.

    Args:
        bbox (jnp.ndarray): bounding box
        squeezed (bool): suppose the number of ranks of the input tensor is R,
            the #rank of the output tensor will be R - 1 is `squeezed = True`.
            Otherwise the #rank of the output will remain as R, and the last
            rank contains only 1 dimension
    """
    if squeezed:
        return bbox[..., 9]
    return bbox[..., 9:10]


@jax.jit
def class_logits(bbox: jnp.ndarray) -> jnp.ndarray:
    """Get class logits from a tensor / array.

    Args:
        bbox (jnp.ndarray): bounding box
    """
    return bbox[..., 10:]


@jax.jit
def area(bbox: jnp.ndarray) -> jnp.ndarray:
    """Calculate area of a bounding box."""
    return bbox[..., 2] * bbox[..., 3]


@jax.jit
def asdbox(bbox: jnp.ndarray) -> jnp.ndarray:
    """Transform a bounding box into a diagonal box."""
    return jnp.concatenate(
        [
            xy(bbox) - wh(bbox) * 0.5,
            xy(bbox) + wh(bbox) * 0.5,
            bbox[..., 8:],
        ],
        axis=-1,
    )


@jax.jit
def interarea(bbox_pred: jnp.ndarray, bbox_label: jnp.ndarray) -> jnp.ndarray:
    """Intersection area of two bounding boxes."""
    dbox_pred = asdbox(bbox_pred)
    dbox_label = asdbox(bbox_label)
    return dbox.interarea(dbox_pred, dbox_label)


@jax.jit
def iou(bbox_pred: jnp.ndarray, bbox_label: jnp.ndarray) -> jnp.ndarray:
    """Calculate IoU of two bounding boxes."""
    area_pred = area(bbox_pred)
    area_label = area(bbox_label)
    area_inter = interarea(bbox_pred, bbox_label)
    area_union = area_pred + area_label - area_inter
    return (area_inter + EPSILON) / (area_union + EPSILON)


@jax.jit
def _inverse_sig(x: float) -> float:
    return math.log(x / (1 - x))


def objects(
    bbox: jnp.ndarray,
    *,
    from_logits: bool = False,
    conf_th: float = 0.1,
) -> jnp.ndarray:
    """Get bounding boxes only with high confidence scores.
    TODO: computing tensors with undetermined shape cannot use JIT

    Args:
        bbox (jnp.ndarray): any bounding box of any valid shape
        from_logits (bool): whether the confidence score is logit or
            probability, default False
        conf_th (float): confidence threshold, default 0.1

    Return:
        jnp.ndarray: tensor of shape [N, 10] where N is the number of boxes
        containing an object, filtered by class ID
    """
    # get indices where class ID <> 0
    if from_logits:
        conf_th = _inverse_sig(conf_th)
    return bbox[conf1d(bbox) >= conf_th]
