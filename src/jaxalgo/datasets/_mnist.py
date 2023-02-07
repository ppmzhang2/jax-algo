import os
import urllib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import KeyArray

from jaxalgo import cfg
from jaxalgo.datasets._base import BaseDataset
from jaxalgo.datasets._types import Mode

TESTURL = "https://pjreddie.com/media/files/mnist_test.csv"
TRAINURL = "https://pjreddie.com/media/files/mnist_train.csv"
TEST_NAME = "mnist_test.npy"
TRAIN_NAME = "mnist_train.npy"
SHAPE = (28, 28, 1)

__all__ = ["MnistBatch", "MnistDataset"]


class MnistBatch(NamedTuple):
    image: jnp.ndarray  # [N, 28, 28, 1]
    label: jnp.ndarray  # [N]


class MnistDataset(BaseDataset):

    def __init__(self,
                 batch: int = 128,
                 mode: str = "TRAIN",
                 data_dir: str = cfg.DATADIR):
        super().__init__(batch, mode, data_dir)
        self._url = TRAINURL if self._mode is Mode.TRAIN else TESTURL
        self._file_name = TRAIN_NAME if self._mode is Mode.TRAIN else TEST_NAME
        self._file_path = os.path.join(data_dir, self._file_name)

    @staticmethod
    def _fetch(url: str) -> jnp.ndarray:
        """Fetch array from a URL."""
        with urllib.request.urlopen(url) as f:
            arr = np.genfromtxt(f, delimiter=",", dtype=np.float32)
        return arr

    def _save(self, arr: jnp.ndarray) -> None:
        if not os.path.isfile(self._file_path):
            jnp.save(self._file_path, arr)

    def _load(self) -> jnp.ndarray:
        if os.path.isfile(self._file_path):
            return jnp.load(self._file_path)
        arr = self._fetch(self._url)
        self._save(arr)
        return arr

    def rand_batch(self, key: KeyArray) -> MnistBatch:
        arr = jax.random.permutation(
            key,
            self._load(),
            axis=0,
            independent=False,
        )[:self._batch]
        return MnistBatch(
            image=arr[:, 1:].reshape(self._batch, *SHAPE),
            label=arr[:, 0],
        )
