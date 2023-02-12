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
    msk: jnp.ndarray
    res: jnp.ndarray
    logit: bool


dataset = [
    Data(
        lab=jnp.array([0., 0., 0., 1., 1., 1., 0.5]),
        prd=jnp.array([1., -1., 0., 1., -1., 0., 0.]),
        msk=jnp.array([1., 1., 1., 1., 1., 1., 1.]),
        res=jnp.array([1.3128, 0.3138, 0.6931, 0.3138, 1.3128, 0.6931,
                       0.6931]),
        logit=True,
    ),
    Data(
        lab=jnp.array([[0, 1], [0, 0]], dtype=jnp.float32),
        prd=jnp.array([[0.6, 0.4], [0.4, 0.6]]),
        msk=jnp.array([[1., 1.], [1., 1.]]),
        res=jnp.array([1.8272, 1.423]),
        logit=False,
    ),
]


@pytest.mark.parametrize("data", dataset)
def test_bce(data: Data) -> None:
    """Test `bce` loss function."""
    res = bce(data.lab, data.prd, data.msk, logit=data.logit)
    assert jnp.all(jnp.abs(res - data.res) < 1e-2)
