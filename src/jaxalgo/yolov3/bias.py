"""Training and loss."""
import logging

import jax
import jax.numpy as jnp
import numpy as np

from jaxalgo.yolov3.box import bbox
from jaxalgo.yolov3.box import pbox
from jaxalgo.yolov3.const import V3_ANCHORS
from jaxalgo.yolov3.const import V3_GRIDSIZE
from jaxalgo.yolov3.const import V3_INRESOLUT
from jaxalgo.yolov3.utils import onehot_offset

LOGGER = logging.getLogger(__name__)

N_CLASS = 80

STRIDE_MAP = {scale: V3_INRESOLUT // scale for scale in V3_GRIDSIZE}

# anchors measured in corresponding strides
ANCHORS_IN_STRIDE = (np.array(V3_ANCHORS).T /
                     np.array(sorted(STRIDE_MAP.values()))).T

# map from grid size to anchors measured in stride
ANCHORS_MAP = {
    scale: ANCHORS_IN_STRIDE[i]
    for i, scale in enumerate(V3_GRIDSIZE)
}


def _all_but_last_dim(x: jnp.ndarray) -> tuple:
    return tuple(range(x.ndim)[1:])


def _smooth(label: jnp.ndarray, alpha: float = 1e-3):
    """Label smoothing ported from tensorflow."""
    label_ = jnp.clip(label, 0.0, 1.0)
    return label_ * (1.0 - alpha) + 0.5 * alpha


def _log(prob: jnp.ndarray, epsilon: float = 1e-3) -> jnp.ndarray:
    """Safe logarithm."""
    return jnp.log(jnp.abs(prob) + epsilon)


def _bce_prob(label: jnp.ndarray, prob: jnp.ndarray) -> jnp.ndarray:
    """Computes cross entropy given probabilities.

    z * log(x) + (1 - z) * log(1-x)
    """
    return -label * _log(prob) - (1 - label) * _log(1 - prob)


def _bce_logits(label: jnp.ndarray, logit: jnp.ndarray) -> jnp.ndarray:
    """Computes sigmoid cross entropy given logits.

    Ported from tensorflow:
        max(x, 0) - x * z + log(1 + exp(-abs(x)))

    equivalent to:
        z * -log_sigmoid(x) + (1 - z) * -log_sigmoid(-x)
    """
    return (jax.nn.relu(logit) - logit * label +
            jnp.log(1 + jnp.exp(-jnp.abs(logit))))


def bce(
    lab: jnp.ndarray,
    prd: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    logit: bool = True,
) -> jnp.ndarray:
    """Computes sum binary cross entropy either given logits or probability."""
    lab = _smooth(lab)
    if logit:
        raw = _bce_logits(lab, prd)
    else:
        raw = _bce_prob(lab, prd)
    return jnp.sum(raw * mask, axis=_all_but_last_dim(raw))


def sse(lab: jnp.ndarray, prd: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Sum Squared Error."""
    return jnp.sum(jnp.square(lab - prd) * mask, axis=_all_but_last_dim(lab))


def fce(
    lab: jnp.ndarray,
    prd: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    logit: bool = True,
) -> jnp.ndarray:
    """Compute sum focal cross entropy either given logits or probability."""
    lab = _smooth(lab)
    if logit:
        raw = _bce_logits(lab, prd)
    else:
        raw = _bce_prob(lab, prd)
    pt = jnp.exp(-raw)
    alpha = (2.0 * lab + 1.0) * 0.25  # 0.25 for 0, 0.75 for 1
    return jnp.sum(alpha * jnp.power((1 - pt), 2) * raw * mask,
                   axis=_all_but_last_dim(raw))


def bias(
    pred: jnp.ndarray,
    label: jnp.ndarray,
    lambda_obj: float = 1.0,
    lambda_bgd: float = 1.0,
    lambda_xy: float = 1.0,
    lambda_wh: float = 1.0,
    lambda_class: float = 1.0,
    conf_th: float = 0.5,
) -> jnp.ndarray:
    """Calculate loss."""
    pred_ = pbox.scaled_bbox(pred)

    mask_obj = (bbox.confnd(label) > conf_th).astype(jnp.float32)
    # TODO: multiply best iou
    mask_bgd = (bbox.confnd(label) < conf_th).astype(jnp.float32)

    ious = bbox.iou(pred_, label)[..., jnp.newaxis]

    conf_bias_iou = (2.0 - bbox.area(label) / 416**2)[..., jnp.newaxis]
    bias_iou = jnp.sum(conf_bias_iou * mask_obj * (1.0 - ious))

    conf_focal = jnp.power(
        bbox.confnd(label) - jax.nn.sigmoid(bbox.confnd(pred_)), 2)

    # background loss
    bias_bgd = bce(bbox.confnd(label), bbox.confnd(pred_),
                   mask_bgd * conf_focal)

    # object loss
    # label probability should be `1 * IOU` score according to the YOLO paper
    # TBD: weight with confidence
    bias_obj = bce(ious * bbox.confnd(label), bbox.confnd(pred_),
                   mask_obj * conf_focal)

    # object center coordinates (xy) loss
    bias_xy = sse(bbox.xy_offset(label), bbox.xy_offset(pred_), mask_obj)

    # box size (wh) loss
    bias_wh = sse(bbox.wh_exp(label), bbox.wh_exp(pred_), mask_obj)

    # class loss
    bias_class = bce(
        onehot_offset(bbox.class_sn(label, squeezed=True), N_CLASS),
        bbox.class_logits(pred_),
        mask_obj,
    )

    # LOGGER.info("\n"
    #             f"    IoU Bias={bias_iou};\n"
    #             f"    Background Bias={bias_bgd};\n"
    #             f"    Object Bias={bias_obj};\n"
    #             f"    XY Bias={bias_xy};\n"
    #             f"    WH Bias={bias_wh};\n"
    #             f"    Class Bias={bias_class}")
    return (lambda_bgd * jnp.mean(bias_bgd) + lambda_obj * jnp.mean(bias_obj) +
            lambda_xy * jnp.mean(bias_xy) + lambda_wh * jnp.mean(bias_wh) +
            lambda_class * jnp.mean(bias_class) + bias_iou)
