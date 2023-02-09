"""Manipulate width-height boxes."""
import jax.numpy as jnp

__all__ = ["iou"]

EPSILON = 1e-3


def iou(whs1: jnp.ndarray, whs2: jnp.ndarray) -> jnp.ndarray:
    """IoU calculated without x and y, by aligning boxes along two edges.

    Args:
        whs1 (jnp.ndarray): width and height of the first bounding boxes
        whs2 (jnp.ndarray): width and height of the second bounding boxes

    Returns:
        jnp.ndarray: Intersection over union of the corresponding boxes
    """
    intersection = (jnp.minimum(whs1[..., 0], whs2[..., 0]) *
                    jnp.minimum(whs1[..., 1], whs2[..., 1]))
    union = (whs1[..., 0] * whs1[..., 1] + whs2[..., 0] * whs2[..., 1] -
             intersection)
    return (intersection + EPSILON) / (union + EPSILON)
