"""Trainer and evaluator."""
import logging
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax.random import KeyArray

from jaxalgo.datasets import MnistBatch
from jaxalgo.datasets import MnistDataset
from jaxalgo.resnet.model import ResNet18

LOGGER = logging.getLogger(__name__)

NUM_CLASS = 10


def model(x: jnp.ndarray) -> jnp.ndarray:
    """Apply network."""
    net = ResNet18(NUM_CLASS)
    return net(x)


class TrainVar(NamedTuple):
    """Contain variables (parameters + states) during training."""
    params: hk.Params
    states: hk.State
    opt_states: optax.OptState


def loss(
    params: hk.Params,
    states: hk.State,
    modelf: hk.TransformedWithState,
    key: KeyArray,
    batch: MnistBatch,
) -> tuple[jnp.ndarray, hk.State]:
    """Loss function."""

    def bias(prd: jnp.ndarray, lab: jnp.ndarray,
             num_class: int) -> jnp.ndarray:
        """Bias function.

        Args:
            prd (jnp.ndarray): predicted classification logits
            lab (jnp.ndarray): label
            num_class (int): number of classes
        """
        loss_prob = -jax.nn.one_hot(lab, num_class) * jax.nn.log_softmax(prd)
        return jnp.sum(loss_prob) / loss_prob.shape[0]

    def reg_l2(params: hk.Params) -> jnp.ndarray:
        """L2 variance."""
        params_sq = jax.tree_map(lambda x: jnp.square(x), params)
        return 0.5 * sum(
            jnp.sum(p) for p in jax.tree_util.tree_leaves(params_sq))

    prd, states = modelf.apply(params, states, key, batch.image / 255.)
    return bias(prd, batch.label, NUM_CLASS) + 1e-3 * reg_l2(params), states


def run_init(
    key: KeyArray,
    batch: MnistBatch,
) -> tuple[TrainVar, hk.TransformedWithState, optax.GradientTransformation]:
    """Initialize runner."""
    modelf = hk.transform_with_state(model)
    optim = optax.adam(1e-3)
    params, states = modelf.init(key, batch.image)
    opt_states = optim.init(params)
    return TrainVar(params, states, opt_states), modelf, optim


def trainer(n_epoch: int) -> None:
    """Trainer."""

    @jax.jit
    def train_step(
        var: TrainVar,
        key: KeyArray,
        batch: MnistBatch,
    ) -> tuple[TrainVar, jnp.ndarray]:
        """Update parameters and states."""
        (los, states), grads = jax.value_and_grad(loss, has_aux=True)(
            var.params,
            var.states,
            modelf,
            key,
            batch,
        )
        updates, opt_states = optim.update(grads, var.opt_states)
        params = optax.apply_updates(var.params, updates)
        return TrainVar(params, states, opt_states), los

    @jax.jit
    def evaluate(
        var: TrainVar,
        key: KeyArray,
        batch: MnistBatch,
    ) -> jnp.ndarray:
        """Evaluate classification accuracy."""
        logits, _ = modelf.apply(var.params, var.states, key,
                                 batch.image / 255.)
        predictions = jnp.argmax(logits, axis=-1)
        return jnp.mean(predictions == batch.label)

    seed = 0
    key = jax.random.PRNGKey(seed)
    ds_train = MnistDataset()
    k_model, k_sample, k_evaldata, subkey = jax.random.split(key, num=4)
    var, modelf, optim = run_init(k_model, ds_train.rand_batch(k_sample))
    batch_test = MnistDataset(mode="TEST", batch=500).rand_batch(k_evaldata)
    keys_epoch = jax.random.split(subkey, num=n_epoch)
    for idx, k in enumerate(keys_epoch):
        (k_train, k_eval, k_traindata) = jax.random.split(k, num=3)
        var, los = train_step(var, k_train, ds_train.rand_batch(k_traindata))
        LOGGER.info(f"training loss: {los}")
        if idx % 10 == 0:
            accuracy = evaluate(var, k_eval, batch_test)
            LOGGER.info(f"eval accuracy: {accuracy}")
