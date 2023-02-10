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
    return label * (1.0 - alpha) + 0.5 * alpha


def _log(prob: jnp.ndarray, epsilon: float = 1e-3) -> jnp.ndarray:
    """Safe logarithm."""
    return jnp.log(jnp.abs(prob) + epsilon)


def _bce_prob(label: jnp.ndarray, prob: jnp.ndarray) -> jnp.ndarray:
    """Computes cross entropy given probabilities.

    z * log(x) + (1 - z) * log(1-x)
    """
    label = _smooth(label)
    return -label * _log(prob) - (1 - label) * _log(1 - prob)


def _bce_logits(label: jnp.ndarray, logit: jnp.ndarray) -> jnp.ndarray:
    """Computes sigmoid cross entropy given logits.

    Ported from tensorflow:
        max(x, 0) - x * z + log(1 + exp(-abs(x)))

    equivalent to:
        z * -log_sigmoid(x) + (1 - z) * -log_sigmoid(-x)
    """
    label = _smooth(label)
    return (jax.nn.relu(logit) - logit * label +
            jnp.log(1 + jnp.exp(-jnp.abs(logit))))


@jax.jit
def bce(lab: jnp.ndarray, prd: jnp.ndarray, logit: bool = True) -> jnp.ndarray:
    """Computes binary cross entropy either given logits or probability."""
    if logit:
        raw = _bce_logits(lab, prd)
    else:
        raw = _bce_prob(lab, prd)
    return jnp.mean(raw, axis=-1)


@jax.jit
def mse(lab: jnp.ndarray, prd: jnp.ndarray) -> jnp.ndarray:
    """Mean Squared Error."""
    return jnp.mean(jnp.square(lab - prd), axis=-1)


def bias(
    pred: jnp.ndarray,
    label: jnp.ndarray,
    lambda_obj: float = 1.0,
    lambda_bgd: float = 1.0,
    lambda_coord: float = 5.0,
    conf_th: float = 0.5,
) -> jnp.ndarray:
    """Calculate loss."""
    pred_ = pbox.scaled_bbox(pred)

    indices_obj = bbox.conf1d(label) > conf_th
    indices_bgd = bbox.conf1d(label) < conf_th

    prd_obj = pred_[indices_obj]
    prd_bgd = pred_[indices_bgd]
    lab_obj = label[indices_obj]
    lab_bgd = label[indices_bgd]

    ious = bbox.iou(prd_obj, lab_obj)[..., jnp.newaxis]
    # background loss
    # TBD: weight with confidence
    bias_bgd = (lambda_bgd *
                jnp.mean(bce(bbox.confnd(lab_bgd), bbox.confnd(prd_bgd))))

    # object loss
    # label probability should be `1 * IOU` score according to the YOLO paper
    # TBD: weight with confidence
    bias_obj = (lambda_obj *
                jnp.mean(bce(ious * bbox.confnd(lab_obj), bbox.confnd(prd_obj))))

    # object center coordinates (xy) loss
    bias_xy = (lambda_coord *
               jnp.mean(mse(bbox.xy_offset(lab_obj), bbox.xy_offset(prd_obj))))

    # box size (wh) loss
    bias_wh = jnp.mean(mse(bbox.wh_exp(lab_obj), bbox.wh_exp(prd_obj)))

    # class loss
    bias_class = jnp.mean(
        bce(
            onehot_offset(bbox.class_sn(lab_obj, squeezed=True), N_CLASS),
            bbox.class_logits(prd_obj),
        ))

    # TODO: add back log
    # LOGGER.info("\n"
    #             f"    Background Bias={bias_bgd};\n"
    #             f"    Object Bias={bias_obj};\n"
    #             f"    XY Bias={bias_xy};\n"
    #             f"    WH Bias={bias_wh};\n"
    #             f"    Class Bias={bias_class}")
    return bias_bgd + bias_obj + bias_xy + bias_wh + bias_class
