"""Test IoU functions of module box."""
from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from jaxalgo.yolov3 import nms

E = 1e-3


@dataclass(frozen=True)
class Data:
    """Input dataset and output shapes."""
    boxes: jnp.ndarray
    th: float
    res: jnp.ndarray


arr = jnp.array(
    [[1., 1., 2., 2., 0., 0., 0., 0., 0.6, 1.],
     [2., 2., 2., 2., 0., 0., 0., 0., 0.4, 1.],
     [3., 3., 2., 2., 0., 0., 0., 0., 0.6, 1.]],
    dtype=jnp.float32,
)

dataset = [
    Data(boxes=arr,
         th=1.5,
         res=jnp.array(
             [[1., 1., 2., 2., 0., 0., 0., 0., 0.6, 1.],
              [3., 3., 2., 2., 0., 0., 0., 0., 0.6, 1.]],
             dtype=jnp.float32,
         )),
    Data(boxes=arr, th=1.4, res=arr),
]


@pytest.mark.parametrize("data", dataset)
def test_nms(data: Data) -> None:
    """Test nms."""
    res = nms.nms(data.boxes, data.th)
    assert (res - data.res).sum() < E
