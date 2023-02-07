from jaxalgo import cfg
from jaxalgo.datasets._types import Mode


class BaseDataset:

    def __init__(self,
                 batch: int,
                 mode: str = "TRAIN",
                 data_dir: str = cfg.DATADIR):
        self._batch = batch
        self._mode = Mode[mode]
        self._data_dir = data_dir
