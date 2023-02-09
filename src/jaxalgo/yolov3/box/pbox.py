"""Manipulate prediction box.

shape: [..., M], where M = 5 + N_CLASS e.g. M = 85 when N_CLASS = 80

format: (x, y, w, h, conf_logit, class_logit_1, class_logit_2, ...)
"""
import jax
import jax.numpy as jnp

from jaxalgo.yolov3.const import V3_ANCHORS
from jaxalgo.yolov3.const import V3_GRIDSIZE
from jaxalgo.yolov3.const import V3_INRESOLUT
from jaxalgo.yolov3.utils import onecold_offset

STRIDE_MAP = {scale: V3_INRESOLUT // scale for scale in V3_GRIDSIZE}

# anchors measured in corresponding strides
ANCHORS_IN_STRIDE = (jnp.array(V3_ANCHORS).T /
                     jnp.array(sorted(STRIDE_MAP.values()))).T

# map from grid size to anchors measured in stride
ANCHORS_MAP = {
    scale: ANCHORS_IN_STRIDE[i]
    for i, scale in enumerate(V3_GRIDSIZE)
}


@jax.jit
def xy(pbox: jnp.ndarray) -> jnp.ndarray:
    """Get x-y coordinatestensor from a tensor / array.

    Args:
        pbox (jnp.ndarray): model predicted box
    """
    return pbox[..., :2]


@jax.jit
def wh(pbox: jnp.ndarray) -> jnp.ndarray:
    """Get width and height from a tensor / array.

    Args:
        pbox (jnp.ndarray): model predicted box
    """
    return pbox[..., 2:4]


@jax.jit
def xywh(pbox: jnp.ndarray) -> jnp.ndarray:
    """Get x-y coordinatestensor, width and height from a tensor / array.

    Args:
        pbox (jnp.ndarray): bounding box
    """
    return pbox[..., :4]


@jax.jit
def conf(pbox: jnp.ndarray) -> jnp.ndarray:
    """Get object confidence from a tensor / array.

    Args:
        pbox (jnp.ndarray): model predicted box
    """
    return pbox[..., 4:5]


@jax.jit
def class_logits(pbox: jnp.ndarray) -> jnp.ndarray:
    """Get class logits from a tensor / array.

    Args:
        pbox (jnp.ndarray): model predicted box
    """
    return pbox[..., 5:]


def _grid_coord(batch_size: int, grid_size: int, n_anchor: int) -> jnp.ndarray:
    """Top-left coordinates of each grid cells.

    created usually out of a output tensor

    Args:
        batch_size (int): batch size
        grid_size (int): grid size, could be 13 (large), 26 (medium) or 52
            (small)
        n_anchor (int): number of anchors of a specific grid size, usually
            should be 3
    """
    vec = jnp.arange(0, grid_size, dtype=jnp.float32)  # x or y range
    xs = vec[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
    ys = vec[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]
    xss = jnp.tile(xs, (batch_size, 1, grid_size, n_anchor))
    yss = jnp.tile(ys, (batch_size, grid_size, 1, n_anchor))
    return jnp.stack([xss, yss], axis=-1)


def scaled_bbox(y: jnp.ndarray) -> jnp.ndarray:
    """Transform prediction to actual sized `bbox`."""
    batch_size = y.shape[0]
    grid_size = y.shape[1]  # 52, 26 or 13
    n_anchor = y.shape[3]  # 3
    stride = STRIDE_MAP[grid_size]
    anchors = ANCHORS_MAP[grid_size]
    topleft_coords = _grid_coord(batch_size, grid_size, n_anchor)
    xy_sig = jax.nn.sigmoid(xy(y))
    wh_exp = wh(y)
    cate_logit = class_logits(y)

    xy_act = (xy_sig + topleft_coords) * stride
    wh_act = (jnp.exp(wh_exp) * anchors) * stride
    cate_sn = onecold_offset(cate_logit)
    return jnp.concatenate(
        [
            xy_act,
            wh_act,
            xy_sig,
            wh_exp,
            conf(y),
            cate_sn[..., jnp.newaxis],
            cate_logit,
        ],
        axis=-1,
    )
