"""Manipulate prediction box.

shape: [..., M], where M = 5 + N_CLASS e.g. M = 85 when N_CLASS = 80

format: (x, y, w, h, conf_logit, class_logit_1, class_logit_2, ...)
"""
import jax
import jax.numpy as jnp

from jaxalgo.datasets.coco.const import YOLO_GRIDS
from jaxalgo.datasets.coco.const import YoloScale
from jaxalgo.yolov3.utils import onecold_offset


@jax.jit
def xy(pbox: jnp.ndarray) -> jnp.ndarray:
    """Get x-y coordinatestensor from a tensor / array.

    Args:
        pbox (jnp.ndarray): model predicted box
    """
    # return jnp.concatenate([pbox[..., 1:2], pbox[..., 0:1]], axis=-1)
    return pbox[..., 0:2]


@jax.jit
def wh(pbox: jnp.ndarray) -> jnp.ndarray:
    """Get width and height from a tensor / array.

    Args:
        pbox (jnp.ndarray): model predicted box
    """
    # return jnp.concatenate([pbox[..., 3:4], pbox[..., 2:3]], axis=-1)
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

    Returns:
        jnp.ndarray: [N_BATCH, H, W, N_ANCHOR, 2]
    """
    vec = jnp.arange(0, grid_size, dtype=jnp.float32)  # x or y range
    xs = vec[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
    ys = vec[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]
    xss = jnp.tile(xs, (batch_size, 1, grid_size, n_anchor))
    yss = jnp.tile(ys, (batch_size, grid_size, 1, n_anchor))
    return jnp.stack([yss, xss], axis=-1)  # (H, W), NOT the other way around


@jax.jit
def asbbox(y: jnp.ndarray) -> jnp.ndarray:
    """Transform prediction to actual sized `bbox`."""
    batch_size = y.shape[0]
    grid = YOLO_GRIDS[YoloScale(y.shape[1])]
    n_anchor = y.shape[3]  # 3
    topleft_coords = _grid_coord(batch_size, grid.size, n_anchor)
    xy_offset = jax.nn.sigmoid(xy(y))
    wh_exp = wh(y)
    cate_logit = class_logits(y)

    xy_rel = (xy_offset + topleft_coords) * grid.stride
    wh_rel = jnp.exp(wh_exp) * grid.anchors
    cate_sn = onecold_offset(cate_logit)
    return jnp.concatenate(
        [
            xy_rel,
            wh_rel,
            xy_offset,
            wh_exp,
            conf(y),
            cate_sn[..., jnp.newaxis],
            cate_logit,
        ],
        axis=-1,
    )
