"""Test YOLOv3 Modles."""
from dataclasses import dataclass
from typing import NoReturn

import haiku as hk
import jax
import jax.numpy as jnp
import pytest

from jaxalgo.yolov3.runner import get_xfm

N = 2
N_CLASS = 80
N_BBOX = 5


@dataclass(frozen=True)
class Data:
    """Input dataset and output shapes."""
    x: jnp.ndarray
    shape_s: tuple[int, int, int, int, int]
    shape_m: tuple[int, int, int, int, int]
    shape_l: tuple[int, int, int, int, int]


in_shape = (N, 416, 416, 3)
seed = 0
key = jax.random.PRNGKey(seed)
x = jax.random.normal(key, in_shape, dtype=jnp.float32)

dataset = [
    Data(x=x,
         shape_s=(N, 52, 52, 3, N_CLASS + N_BBOX),
         shape_m=(N, 26, 26, 3, N_CLASS + N_BBOX),
         shape_l=(N, 13, 13, 3, N_CLASS + N_BBOX)),
]


@pytest.mark.parametrize("data", dataset)
def test_resnet_model(data: Data) -> NoReturn:
    """Test `netv3` YOLO model."""
    modelf = get_xfm()
    params, states = modelf.init(key, data.x)
    (t1, t2, t3), _ = modelf.apply(params, states, key, data.x)

    assert t1.shape == data.shape_s
    assert t2.shape == data.shape_m
    assert t3.shape == data.shape_l
