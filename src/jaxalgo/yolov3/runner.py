"""Trainer and evaluator."""
import logging
import os
import pickle
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
    os.path.join(cfg.DATADIR, "model_yolov3_params.pickle"))
PATH_STATES = os.path.abspath(
    os.path.join(cfg.DATADIR, "model_yolov3_states.pickle"))


def get_var_path(epoch: int, *, params: bool = True):
    if params:
        return os.path.abspath(
            os.path.join(cfg.DATADIR, f"model_yolov3_params_{epoch}.pickle"))
    return os.path.abspath(
        os.path.join(cfg.DATADIR, f"model_yolov3_states_{epoch}.pickle"))


def model_fn(x: jnp.ndarray) -> jnp.ndarray:
    """Apply network."""
    net = YoloV3(NUM_CLASS)
    return net(x)


class ModelState(NamedTuple):
    """Contain variables (parameters + states) during training."""
    params: hk.Params
    states: hk.State


def reg_l2(params: hk.Params) -> jnp.ndarray:
    """L2 variance."""
    params_sq = jax.tree_map(lambda x: jnp.square(x), params)
    return 0.5 * sum(jnp.sum(p) for p in jax.tree_util.tree_leaves(params_sq))


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


def l2loss(
    params: hk.Params,
    states: hk.State,
    xfm: hk.TransformedWithState,
    key: KeyArray,
    batch: Yolov3Batch,
    *,
    lambda_reg: float = 1e-3,
) -> tuple[jnp.ndarray, hk.State]:
    """Loss function with L2 regulation."""

    los, states = loss(params, states, xfm, key, batch)
    return los + lambda_reg * reg_l2(params), states


def load_state(path_params: str, path_states: str) -> ModelState:
    """Load a model's state from pickles."""
    with open(path_params, "rb") as f_params:
        params = pickle.load(f_params)
    with open(path_states, "rb") as f_states:
        states = pickle.load(f_states)
    return ModelState(params, states)


def save_state(var: ModelState, path_params: str, path_states: str) -> None:
    """Save a model's state from pickles."""
    shutil.rmtree(path_params, ignore_errors=True)
    shutil.rmtree(path_states, ignore_errors=True)
    with open(path_params, "wb") as f_params:
        pickle.dump(var.params, f_params, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path_states, "wb") as f_states:
        pickle.dump(var.states, f_states, protocol=pickle.HIGHEST_PROTOCOL)


def best_state(best_eval_score: jnp.ndarray):
    """Returns a function to save / load model's states.
    Save current model's state if if performs better than baseline, and load
    saved state otherwise.

    Args:
        best_eval_score (jnp.ndarray): previous best evaluation score
    """

    def _helper(
        var: ModelState,
        eval_score: jnp.ndarray,
    ) -> tuple[jnp.ndarray, ModelState]:
        if eval_score <= best_eval_score:
            LOGGER.debug("---- restore to previous best...")
            return best_eval_score, load_state(PATH_PARAMS, PATH_STATES)
        LOGGER.debug("---- saving new best state...")
        save_state(var, PATH_PARAMS, PATH_STATES)
        return eval_score, var

    return _helper


def state_checkpoint(best_eval_score: jnp.ndarray):
    """Returns a checkpoint function to save model's states.
    Save current model's state if if performs better than baseline.

    Args:
        best_eval_score (jnp.ndarray): previous best evaluation score
    """

    def _helper(
        var: ModelState,
        eval_score: jnp.ndarray,
    ) -> jnp.ndarray:
        if eval_score <= best_eval_score:
            LOGGER.debug(f"---- best is still {best_eval_score:5.4f}")
            return best_eval_score
        LOGGER.debug(f"---- new best state: {eval_score:5.4f}")
        save_state(var, PATH_PARAMS, PATH_STATES)
        return eval_score

    return _helper


def model_init(
    xfm: hk.TransformedWithState,
    key: KeyArray,
    batch: Yolov3Batch,
) -> ModelState:
    """Get parameters and states from a model."""
    params, states = xfm.init(key, batch.image)
    return ModelState(params, states)


def train_step(
    xfm: hk.TransformedWithState,
    optim: optax.GradientTransformation,
):

    def _train_step_fn(
        var: ModelState,
        opt_state: optax.OptState,
        key: KeyArray,
        batch: Yolov3Batch,
    ) -> tuple[ModelState, optax.OptState, jnp.ndarray]:
        """Update parameters and states."""
        (train_los, states), grads = jax.value_and_grad(l2loss, has_aux=True)(
            var.params,
            var.states,
            xfm,
            key,
            batch,
        )
        updates, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(var.params, updates)
        return ModelState(params, states), opt_state, train_los

    return _train_step_fn


def train(
    seed: int,
    n_epoch: int,
    lr: float,
    batch_train: int,
    batch_valid: int,
    eval_span: int,
    eval_loop: int,
) -> None:
    """Trainer.

    Args:
        seed (int): seed to generate random key
        n_epoch (int): number of epochs
        lr (float): learning rate
        batch_train (int): batch size for training
        batch_valid (int): batch size for validation
        eval_span (int): span of epochs between each evaluation
    """

    @jax.jit
    def train_step_fn(
        var: ModelState,
        opt_state: optax.OptState,
        key: KeyArray,
        batch: Yolov3Batch,
    ) -> tuple[ModelState, optax.OptState, jnp.ndarray]:
        """Update parameters and states."""
        return train_step(xfm, optim)(var, opt_state, key, batch)

    @jax.jit
    def loss_acc(var: ModelState, key: KeyArray, batch: Yolov3Batch,
                 acc: float) -> jnp.ndarray:
        return acc + loss(var.params, var.states, xfm, key, batch)[0]

    ds_train = CocoDataset(mode="TRAIN", batch=batch_train)
    ds_test = CocoDataset(mode="TEST", batch=batch_valid)
    key = jax.random.PRNGKey(seed)
    k_model, k_data, subkey = jax.random.split(key, num=3)

    xfm = hk.transform_with_state(model_fn)
    var = model_init(xfm, k_model, ds_train.rand_batch(k_data))
    optim = optax.adam(lr)
    opt_state = optim.init(var.params)

    for idx, k in enumerate(jax.random.split(subkey, num=n_epoch)):
        (k_train, k_eval, k_traindata) = jax.random.split(k, num=3)
        var, opt_state, train_los = train_step_fn(
            var,
            opt_state,
            k_train,
            ds_train.rand_batch(k_traindata),
        )
        LOGGER.info(f"training loss: {train_los}")
        if idx % eval_span == eval_span - 1:

            los_sum = 0
            for idx, k in enumerate(jax.random.split(k_eval, num=eval_loop),
                                    start=1):
                batch = ds_test.top_batch(idx * eval_loop)
                los_sum = loss_acc(var, k, batch, los_sum)
            eval_los = los_sum / eval_loop

            LOGGER.info(f"evaluation loss: {eval_los}")
            save_state(var, get_var_path(idx), get_var_path(idx, params=False))


def tuning(
    path_params: str,
    path_states: str,
    seed: int,
    n_epoch: int,
    lr: float,
    batch_train: int,
    batch_valid: int,
    eval_span: int,
    eval_loop: int,
) -> None:
    """Fine-tuning from pre-trained model.

    Args:
        path_params (str): path of parameters' pickle
        path_states (str): path of states' pickle
        seed (int): seed to generate random key
        n_epoch (int): number of epochs
        lr (float): learning rate
        batch_train (int): batch size for training
        batch_valid (int): batch size for validation
        eval_span (int): span of epochs between each evaluation
        eval_loop (int): number of evaluation loop, and number of evaluated
            records will be batch_valid * eval_loop
    """

    @jax.jit
    def train_step_fn(
        var: ModelState,
        opt_state: optax.OptState,
        key: KeyArray,
        batch: Yolov3Batch,
    ) -> tuple[ModelState, optax.OptState, jnp.ndarray]:
        """Update parameters and states."""
        return train_step(xfm, optim)(var, opt_state, key, batch)

    @jax.jit
    def loss_acc(var: ModelState, key: KeyArray, batch: Yolov3Batch,
                 acc: float) -> jnp.ndarray:
        return acc + loss(var.params, var.states, xfm, key, batch)[0]

    var = load_state(path_params, path_states)
    ds_train = CocoDataset(mode="TRAIN", batch=batch_train)
    ds_test = CocoDataset(mode="TEST", batch=batch_valid)

    xfm = hk.transform_with_state(model_fn)
    optim = optax.adam(lr)
    opt_state = optim.init(var.params)

    best_score = -9999.9
    key = jax.random.PRNGKey(seed)
    for idx, k in enumerate(jax.random.split(key, num=n_epoch)):
        (k_train, k_eval, k_traindata) = jax.random.split(k, num=3)
        if idx % eval_span == 0:  # evaluate before training starts

            los_sum = 0
            for idx, k in enumerate(jax.random.split(k_eval, num=eval_loop),
                                    start=1):
                batch = ds_test.top_batch(idx * eval_loop)
                los_sum = loss_acc(var, k, batch, los_sum)
            eval_los = los_sum / eval_loop

            LOGGER.info(f"evaluation loss: {eval_los}")
            best_state_fn = best_state(best_score)
            best_score, var = best_state_fn(var, -eval_los)

        var, opt_state, train_los = train_step_fn(
            var,
            opt_state,
            k_train,
            ds_train.rand_batch(k_traindata),
        )
        LOGGER.info(f"training loss: {train_los}")
