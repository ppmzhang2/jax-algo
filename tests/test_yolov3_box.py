"""Test IoU functions of module box."""
from typing import NoReturn

import jax.numpy as jnp

from jaxalgo.yolov3.box import bbox

E = 1e-3
BBOX1 = (2.5, 3.4, 5.0, 6.0)
BBOX2 = (2.6, 4.3, 6.0, 4.0)
LMT_UP = 0.5883
LMT_LOW = 0.5882

bboxes1 = jnp.array((BBOX1, BBOX1))
bboxes2 = jnp.array((BBOX2, BBOX2))


def test_bbox_iou() -> NoReturn:
    """Test bbox.iou."""
    res = bbox.iou(bboxes1, bboxes2)
    assert jnp.all(res > LMT_LOW)
    assert jnp.all(res < LMT_UP)
