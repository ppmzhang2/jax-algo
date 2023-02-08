"""Test YOLOv3 Modles."""
from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from jaxalgo.yolov3.bias import bce


@dataclass(frozen=True)
class Data:
    """Input dataset and output shapes."""
    lab: jnp.ndarray
    prd: jnp.ndarray
    res: jnp.ndarray
    logit: bool


dataset = [
    Data(
        lab=jnp.array([0., 0., 0., 1., 1., 1., 0.5]),
        prd=jnp.array([1., -1., 0., 1., -1., 0., 0.]),
        res=jnp.array(0.7618),
        logit=True,
    ),
    Data(
        lab=jnp.array([[0, 1], [0, 0]], dtype=jnp.float32),
        prd=jnp.array([[0.6, 0.4], [0.4, 0.6]]),
        res=jnp.array([0.9136, 0.7115]),
        logit=False,
    ),
]


@pytest.mark.parametrize("data", dataset)
def test_resnet_model(data: Data) -> None:
    """Test `bce` loss function."""
    res = bce(data.lab, data.prd, logit=data.logit)
    assert jnp.all(jnp.abs(res - data.res) < 1e-2)
