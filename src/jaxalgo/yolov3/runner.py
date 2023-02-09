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


def model_fn(x: jnp.ndarray) -> jnp.ndarray:
    """Apply network."""
    net = YoloV3(NUM_CLASS)
    return net(x)


class ModelState(NamedTuple):
    """Contain variables (parameters + states) during training."""
    params: hk.Params
    states: hk.State


def loss(
    params: hk.Params,
    states: hk.State,
    xfm: hk.TransformedWithState,
    key: KeyArray,
    batch: Yolov3Batch,
) -> tuple[jnp.ndarray, hk.State]:
    """Loss function."""

    (prd_s, prd_m, prd_l), states = xfm.apply(params, states, key,
                                              batch.image / 255.)
    los = (bias(prd_s, batch.label_s) + bias(prd_m, batch.label_m) +
           bias(prd_l, batch.label_l))
    return los, states


def best_state(baseline: jnp.ndarray):
    """Returns a function to save / load model's states.
    Save current model's state if if performs better than baseline, and load
    saved state otherwise.

    Args:
        baseline (jnp.ndarray): baseline score
    """

    def _helper(
        var: ModelState,
        score: jnp.ndarray,
    ) -> tuple[jnp.ndarray, ModelState]:
        if score <= baseline:
            params = jnp.load(PATH_PARAMS)
            states = jnp.load(PATH_STATES)
            return baseline, ModelState(params, states)

        shutil.rmtree(PATH_PARAMS, ignore_errors=True)
        shutil.rmtree(PATH_STATES, ignore_errors=True)
        jnp.save(PATH_PARAMS, var.params)
        jnp.save(PATH_STATES, var.states)
        return score, var

    return _helper


def model_init(
    xfm: hk.TransformedWithState,
    key: KeyArray,
    batch: Yolov3Batch,
) -> ModelState:
    """Get parameters and states from a model."""
    params, states = xfm.init(key, batch.image)
    return ModelState(params, states)


def trainer(seed: int, n_epoch: int) -> None:
    """Trainer."""

    # @jax.jit
    def train_step(
        var: ModelState,
        opt_state: optax.OptState,
        key: KeyArray,
        batch: Yolov3Batch,
    ) -> tuple[ModelState, optax.OptState, jnp.ndarray]:
        """Update parameters and states."""
        (los, states), grads = jax.value_and_grad(loss, has_aux=True)(
            var.params,
            var.states,
            xfm,
            key,
            batch,
        )
        updates, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(var.params, updates)
        return ModelState(params, states), opt_state, los

    # @jax.jit
    def eval_loss(
        var: ModelState,
        key: KeyArray,
        batch: Yolov3Batch,
    ) -> jnp.ndarray:
        """Evaluate classification accuracy."""
        los, _ = loss(var.params, var.states, xfm, key, batch)
        return los

    ds_train = CocoDataset(mode="TRAIN")
    ds_test = CocoDataset(mode="TEST", batch=8)
    key = jax.random.PRNGKey(seed)
    k_model, k_data, subkey = jax.random.split(key, num=3)

    best_state_fn = best_state(-jnp.inf)  # function to load variables
    xfm = hk.transform_with_state(model_fn)
    var = model_init(xfm, k_model, ds_train.rand_batch(k_data))
    optim = optax.adam(1e-3)
    opt_state = optim.init(var.params)

    for idx, k in enumerate(jax.random.split(subkey, num=n_epoch)):
        (k_train, k_eval, k_traindata, k_evaldata) = jax.random.split(k, num=4)
        var, opt_state, los = train_step(
            var,
            opt_state,
            k_train,
            ds_train.rand_batch(k_traindata),
        )
        LOGGER.info(f"training loss: {los}")
        if idx % 10 == 9:
            batch_test = ds_test.rand_batch(k_evaldata)
            eval_los = eval_loss(var, k_eval, batch_test)
            LOGGER.info(f"evaluation loss: {eval_los}")
            best_score, var = best_state_fn(var, -eval_los)
            best_state_fn = best_state(best_score)
