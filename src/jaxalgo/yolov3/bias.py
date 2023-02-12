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


def bce(lab: jnp.ndarray, prd: jnp.ndarray, logit: bool = True) -> jnp.ndarray:
    """Computes binary cross entropy either given logits or probability."""
    lab = _smooth(lab)
    if logit:
        raw = _bce_logits(lab, prd)
    else:
        raw = _bce_prob(lab, prd)
    return jnp.mean(raw, axis=-1)


def mse(lab: jnp.ndarray, prd: jnp.ndarray) -> jnp.ndarray:
    """Mean Squared Error."""
    return jnp.mean(jnp.square(lab - prd), axis=-1)


def fce(
    lab: jnp.ndarray,
    prd: jnp.ndarray,
    *,
    logit: bool = True,
) -> jnp.ndarray:
    """Computes focal cross entropy either given logits or probability."""
    lab = _smooth(lab)
    if logit:
        raw = _bce_logits(lab, prd)
    else:
        raw = _bce_prob(lab, prd)
    pt = jnp.exp(-raw)
    alpha = (2.0 * lab + 1.0) * 0.25  # 0.25 for 0, 0.75 for 1
    return jnp.mean(alpha * jnp.power((1 - pt), 2) * raw, axis=-1)


def bias(
    pred: jnp.ndarray,
    label: jnp.ndarray,
    lambda_obj: float = 2.0,
    lambda_bgd: float = 5.0,
    lambda_coord: float = 2.0,
    lambda_class: float = 2.0,
    conf_th: float = 0.5,
) -> jnp.ndarray:
    """Calculate loss."""
    pred_ = pbox.scaled_bbox(pred)

    indices_obj = (bbox.conf1d(label) > conf_th).astype(jnp.float32)
    indices_bgd = (bbox.conf1d(label) < conf_th).astype(jnp.float32)
    n_obj = indices_obj.sum()
    n_bgd = indices_bgd.sum()

    ious = bbox.iou(pred_, label)[..., jnp.newaxis]
    # background loss
    # TBD: weight with confidence
    bias_bgd = indices_bgd * fce(bbox.confnd(label), bbox.confnd(pred_))

    # object loss
    # label probability should be `1 * IOU` score according to the YOLO paper
    # TBD: weight with confidence
    bias_obj = indices_obj * fce(ious * bbox.confnd(label), bbox.confnd(pred_))

    # object center coordinates (xy) loss
    bias_xy = indices_obj * mse(bbox.xy_offset(label), bbox.xy_offset(pred_))

    # box size (wh) loss
    bias_wh = indices_obj * mse(bbox.wh_exp(label), bbox.wh_exp(pred_))

    # class loss
    bias_class = indices_obj * bce(
        onehot_offset(bbox.class_sn(label, squeezed=True), N_CLASS),
        bbox.class_logits(pred_),
    )

    # LOGGER.info("\n"
    # f"    Background Bias={bias_bgd};\n"
    # f"    Object Bias={bias_obj};\n"
    # f"    XY Bias={bias_xy};\n"
    # f"    WH Bias={bias_wh};\n"
    # f"    Class Bias={np.array(bias_class)}")
    return (lambda_bgd * jnp.sum(bias_bgd) / n_bgd +
            lambda_obj * jnp.sum(bias_obj) / n_obj +
            lambda_coord * jnp.sum(bias_xy) / n_obj +
            jnp.sum(bias_wh) / n_obj +
            lambda_class * jnp.sum(bias_class) / n_obj)
