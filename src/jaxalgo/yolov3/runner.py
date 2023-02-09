"""Trainer and evaluator."""
import logging
import os
import shutil
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax.random import KeyArray

from jaxalgo import cfg
from jaxalgo.datasets.coco import CocoDataset
from jaxalgo.datasets.coco import Yolov3Batch
from jaxalgo.yolov3.bias import bias
from jaxalgo.yolov3.model import YoloV3

LOGGER = logging.getLogger(__name__)

NUM_CLASS = 80
PATH_PARAMS = os.path.abspath(
    os.path.join(cfg.DATADIR, "model_yolov3_params.npy"))
PATH_STATES = os.path.abspath(
    os.path.join(cfg.DATADIR, "model_yolov3_states.npy"))


def model(x: jnp.ndarray) -> jnp.ndarray:
    """Apply network."""
    net = YoloV3(NUM_CLASS)
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
    batch: Yolov3Batch,
) -> tuple[jnp.ndarray, hk.State]:
    """Loss function."""

    (prd_s, prd_m, prd_l), states = modelf.apply(params, states, key,
                                                 batch.image / 255.)
    los = (bias(prd_s, batch.label_s) + bias(prd_m, batch.label_m) +
           bias(prd_l, batch.label_l))
    return los, states


def save_var(baseline: jnp.ndarray):
    """Returns a function to save best model's variables.

    Args:
        baseline (jnp.ndarray): baseline score
    """

    def _saver(var: TrainVar, score: jnp.ndarray) -> jnp.ndarray:
        if score > baseline:
            shutil.rmtree(PATH_PARAMS, ignore_errors=True)
            shutil.rmtree(PATH_STATES, ignore_errors=True)
            jnp.save(PATH_PARAMS, var.params)
            jnp.save(PATH_STATES, var.states)
            return score
        return baseline

    return _saver


def run_init(
    key: KeyArray,
    batch: Yolov3Batch,
) -> tuple[TrainVar, hk.TransformedWithState, optax.GradientTransformation]:
    """Initialize runner."""
    modelf = hk.transform_with_state(model)
    optim = optax.adam(1e-3)
    params, states = modelf.init(key, batch.image)
    opt_states = optim.init(params)
    return TrainVar(params, states, opt_states), modelf, optim


def trainer(n_epoch: int) -> None:
    """Trainer."""

    # @jax.jit
    def train_step(
        var: TrainVar,
        key: KeyArray,
        batch: Yolov3Batch,
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

    # @jax.jit
    def eval_loss(
        var: TrainVar,
        key: KeyArray,
        batch: Yolov3Batch,
    ) -> jnp.ndarray:
        """Evaluate classification accuracy."""
        los, _ = loss(var.params, var.states, modelf, key, batch)
        return los

    seed = 0
    key = jax.random.PRNGKey(seed)
    ds_train = CocoDataset(mode="TRAIN")
    k_model, k_sample, k_evaldata, subkey = jax.random.split(key, num=4)
    var, modelf, optim = run_init(k_model, ds_train.rand_batch(k_sample))
    batch_test = CocoDataset(mode="TEST", batch=8).rand_batch(k_evaldata)
    keys_epoch = jax.random.split(subkey, num=n_epoch)
    save_var_fn = save_var(-jnp.inf)  # function to save variables

    for idx, k in enumerate(keys_epoch):
        (k_train, k_eval, k_traindata) = jax.random.split(k, num=3)
        var, los = train_step(var, k_train, ds_train.rand_batch(k_traindata))
        LOGGER.info(f"training loss: {los}")
        if idx % 10 == 9:
            eval_los = eval_loss(var, k_eval, batch_test)
            LOGGER.info(f"evaluation loss: {eval_los}")
            best_score = save_var_fn(var, -eval_los)
            save_var_fn = save_var(best_score)
