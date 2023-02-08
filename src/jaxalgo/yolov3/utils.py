"""Util Functions."""
import jax
import jax.numpy as jnp

__all__ = ["onehot_offset", "onecold_offset"]


def onehot_offset(
    sn: jnp.ndarray,
    n_class: int,
    offset: int = 1,
) -> jnp.ndarray:
    """Onehot coding for object category SN.

    Object category has three integer attributes: index, SN, ID. For COCO
    data (N_CLASS=80):
        - the ID can be well above 80, not suitable for `tf.one_hot`
        - index (ranges from 0 to 79) in ground truth cannot be used as 0
          is occupied to indicate background
        - to encode SN of range [1, 80], the onehot depth must be added to
          81, before removing the first dimension from the last rank of the
          output
    """
    return jax.nn.one_hot(sn.astype(jnp.int32), n_class + offset)[..., offset:]


def onecold_offset(logits: jnp.ndarray, offset: int = 1) -> jnp.ndarray:
    """Recover object category SN from category logits.

    Since the output range of `tf.argmax` is [0, N_CLASS - 1], `tf.ones` should
    be added to get the SN starting from 1.
    The output tensor will contain all the ranks of the input tensor except
    the last one.
    """
    outshape = logits.shape[:-1]
    return (offset * jnp.ones(outshape, dtype=jnp.float32) +
            jnp.argmax(logits, axis=-1).astype(jnp.float32))
