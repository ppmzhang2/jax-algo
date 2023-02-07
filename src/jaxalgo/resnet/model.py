"""ResNet Model."""
from typing import NoReturn

import haiku as hk
import jax
import jax.numpy as jnp


class IdBlockV1(hk.Module):
    """The identity block v1 (basic block) of ResNet."""

    def __init__(self, filters: int) -> NoReturn:
        """Instantiate a `ResBlock`.

        Args:
            filters: number of out channels
        """
        super().__init__()
        self._filters = filters

    def __call__(self, x: jnp.ndarray, *, training: bool = True):
        """Transform input."""
        y = hk.Conv2D(output_channels=self._filters,
                      kernel_shape=(3, 3),
                      stride=1,
                      with_bias=False,
                      padding="SAME")(x)
        y = hk.BatchNorm(create_scale=True, create_offset=True,
                         decay_rate=0.9)(y, is_training=training)
        y = jax.nn.relu(y)
        y = hk.Conv2D(output_channels=self._filters,
                      kernel_shape=(3, 3),
                      stride=1,
                      with_bias=False,
                      padding="SAME")(y)
        y = hk.BatchNorm(create_scale=True, create_offset=True,
                         decay_rate=0.9)(y, is_training=training)
        return jax.nn.relu(x + y)


class DsBlockV1(hk.Module):
    """The downsampling block v1 (basic block) of ResNet."""

    def __init__(self, filters: int) -> NoReturn:
        """Instantiate a `ResBlock`.

        Args:
            filters: number of out channels
        """
        super().__init__()
        self._filters = filters

    def __call__(self, x: jnp.ndarray, *, training: bool = True):
        """Transform input."""
        y = hk.Conv2D(output_channels=self._filters,
                      kernel_shape=(3, 3),
                      stride=2,
                      with_bias=False,
                      padding="SAME")(x)
        y = hk.BatchNorm(create_scale=True, create_offset=True,
                         decay_rate=0.9)(y, is_training=training)
        y = jax.nn.relu(y)
        y = hk.Conv2D(output_channels=self._filters,
                      kernel_shape=(3, 3),
                      stride=1,
                      with_bias=False,
                      padding="SAME")(y)
        y = hk.BatchNorm(create_scale=True, create_offset=True,
                         decay_rate=0.9)(y, is_training=training)
        x_ = hk.Conv2D(output_channels=self._filters,
                       kernel_shape=(1, 1),
                       stride=2,
                       with_bias=False,
                       padding="SAME")(x)
        return jax.nn.relu(x_ + y)


class ResNet18(hk.Module):
    """The ResNet-18 Model."""

    def __init__(self, ndim_out: int):
        """Instantiate a `ResBlock`."""
        super().__init__()
        self._ndim_out = ndim_out

    def __call__(self, x: jnp.ndarray, *, training: bool = True):
        """Transform input."""
        y = hk.Conv2D(output_channels=64,
                      kernel_shape=(7, 7),
                      stride=2,
                      with_bias=False,
                      padding="SAME")(x)
        y = hk.BatchNorm(create_scale=True, create_offset=True,
                         decay_rate=0.9)(y, is_training=training)
        y = jax.nn.relu(y)
        y = hk.MaxPool(window_shape=(3, 3), strides=2, padding="SAME")(y)
        y = IdBlockV1(64)(y, training=training)
        y = IdBlockV1(64)(y, training=training)
        y = DsBlockV1(128)(y, training=training)
        y = IdBlockV1(128)(y, training=training)
        y = DsBlockV1(256)(y, training=training)
        y = IdBlockV1(256)(y, training=training)
        y = DsBlockV1(512)(y, training=training)
        y = IdBlockV1(512)(y, training=training)
        y = hk.AvgPool(window_shape=(2, 2), strides=1, padding="SAME")(y)
        y = hk.Flatten()(y)
        y = hk.nets.MLP([512, self._ndim_out])(y)
        return y
