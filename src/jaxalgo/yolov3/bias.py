"""Training and loss."""
import logging

import jax
import jax.numpy as jnp

from jaxalgo.datasets.coco.const import YOLO_IN_PX
from jaxalgo.yolov3.box import bbox
from jaxalgo.yolov3.box import pbox
from jaxalgo.yolov3.utils import onehot_offset

LOGGER = logging.getLogger(__name__)

N_CLASS = 80


def _all_but_last_dim(x: jnp.ndarray) -> tuple:
    return tuple(range(x.ndim)[1:])


def _smooth(label: jnp.ndarray, alpha: float = 1e-3) -> jnp.ndarray:
    """Label smoothing ported from tensorflow."""
    label_ = jnp.clip(label, 0.0, 1.0)
    return label_ * (1.0 - alpha) + 0.5 * alpha


def _log(prob: jnp.ndarray, epsilon: float = 1e-3) -> jnp.ndarray:
    """Safe logarithm."""
    return jnp.log(jnp.abs(prob) + epsilon)


def _bce_prob(label: jnp.ndarray, prob: jnp.ndarray) -> jnp.ndarray:
    """Compute cross entropy given probabilities.

    z * log(x) + (1 - z) * log(1-x)
    """
    return -label * _log(prob) - (1 - label) * _log(1 - prob)


def _bce_logits(label: jnp.ndarray, logit: jnp.ndarray) -> jnp.ndarray:
    """Compute sigmoid cross entropy given logits.

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
    """Compute sum binary cross entropy either given logits or probability."""
    lab = _smooth(lab)
    raw = _bce_logits(lab, prd) if logit else _bce_prob(lab, prd)
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
    raw = _bce_logits(lab, prd) if logit else _bce_prob(lab, prd)
    pt = jnp.exp(-raw)
    alpha = (2.0 * lab + 1.0) * 0.25  # 0.25 for 0, 0.75 for 1
    return jnp.sum(alpha * jnp.power((1 - pt), 2) * raw * mask,
                   axis=_all_but_last_dim(raw))


def _iou_max(pred: jnp.ndarray, boxes: jnp.ndarray) -> jnp.ndarray:
    """Calculate max IoU comparing with ALL ground truth boxes.

    Args:
        pred (jnp.ndarray): predictions of shape (B, Grid_H, Grid_W, 3, 10)
        boxes (jnp.ndarray): ground truth boxes of dataset of shape (20, 10)

    Returns:
        jnp.ndarray: a tensor of the same shape (B, Grid_H, Grid_W, 3)
    """
    n_batch = pred.shape[0]
    grid_size = pred.shape[1]
    n_anchor = pred.shape[3]
    # shape: (20, 1, 1, 1, 1, 10)
    boxes_ = boxes[:, jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    # shape: (20, B, Grid_H, Grid_W, 3, 10)
    boxes_ = jnp.tile(boxes_, (1, n_batch, grid_size, grid_size, n_anchor, 1))
    iou_raw = bbox.iou(pred, boxes_)
    return jnp.max(iou_raw, axis=0)


def bias(
    prd: jnp.ndarray,
    lab: jnp.ndarray,
    boxes: jnp.ndarray,
    *,
    lambda_obj: float = 0.05,
    lambda_bgd: float = 20.0,
    lambda_xy: float = 1.0,
    lambda_wh: float = 1.0,
    lambda_class: float = 1.0,
    iou_th: float = 0.3,
) -> jnp.ndarray:
    """Calculate loss."""
    # -------------------------------------------------------------------------
    # coeficients: mask, IoU, focal
    #
    # - background mask: excluding only ground truth boxes is not the true
    #   background; it may contain predictions, not of the exact ground truth
    #   grid cells but with a high IoU with ground truth boxes
    # - focal: do trade-off between the predicted confidence and their best IoU
    #   value; however, without the IoU bias this coefficient can lead to both
    #   `ious` and predicted confidence down to zero
    # -------------------------------------------------------------------------
    mask_obj = bbox.confnd(lab).astype(jnp.float32)
    ious = _iou_max(prd, boxes)[..., jnp.newaxis]
    coef_sbox = 2.0 - 1.0 * bbox.area(lab) / (416**2)
    mask_bgd = (1. - mask_obj) * (ious < iou_th)
    # mask_obj = mask_obj * (ious > iou_th)
    focal = jnp.power(ious - jax.nn.sigmoid(bbox.confnd(prd)), 2)

    # -------------------------------------------------------------------------
    # IoU Bias
    # (1 - max(IoU)) of object grid cells
    # -------------------------------------------------------------------------
    bias_iou = jnp.sum(mask_obj * coef_sbox[..., jnp.newaxis] * (1 - ious),
                       [1, 2, 3, 4])

    # -------------------------------------------------------------------------
    # Background Bias
    # -------------------------------------------------------------------------
    bias_bgd = bce(bbox.confnd(lab), bbox.confnd(prd), mask_bgd * focal)

    # -------------------------------------------------------------------------
    # Object Bias
    # label probability should be `1 * IOU` score according to the YOLO paper
    # TBD: weight with confidence
    # -------------------------------------------------------------------------
    bias_obj = bce(bbox.confnd(lab), bbox.confnd(prd), mask_obj * focal)

    # -------------------------------------------------------------------------
    # Object Center Coordinates (xy) Bias
    # -------------------------------------------------------------------------
    bias_xy = sse(bbox.xy_offset(lab), bbox.xy_offset(prd), mask_obj)

    # -------------------------------------------------------------------------
    # Object Size (wh) Bias
    # -------------------------------------------------------------------------
    bias_wh = sse(bbox.wh_exp(lab), bbox.wh_exp(prd), mask_obj)

    # -------------------------------------------------------------------------
    # Classification Bias
    # -------------------------------------------------------------------------
    bias_class = bce(
        onehot_offset(bbox.class_sn(lab, squeezed=True), N_CLASS),
        bbox.class_logits(prd),
        mask_obj,
    )

    LOGGER.info("\n"
                f"    IoU Bias={bias_iou};\n"
                f"    Background Bias={bias_bgd};\n"
                f"    Object Bias={bias_obj};\n"
                f"    XY Bias={bias_xy};\n"
                f"    WH Bias={bias_wh};\n"
                f"    Class Bias={bias_class}")
    return (lambda_bgd * jnp.mean(bias_bgd) + lambda_obj * jnp.mean(bias_obj) +
            # lambda_xy * jnp.mean(bias_xy) + lambda_wh * jnp.mean(bias_wh) +
            lambda_class * jnp.mean(bias_class) + jnp.mean(bias_iou))
