"""dataset for YOLOv3.

data format:
    - [N_BATCH, 416, 416, 3] for image feature tensor
    - a tuple of three tensors (for each grid scale, i.e. 52, 26, 13
      representing small, medium and large grid respectively) for labels;
      each tensor has a shape like
          [GRID_SIZE, GRID_SIZE, N_MEASURE_PER_GRID, 10].
      The last rank contains the following dimensions in order:
          x, y, w, h, x_offset, y_offset, w_exp, h_exp, conf, classid

TODO: data augmentation
"""
import math
from typing import NamedTuple

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from jax.random import KeyArray

from jaxalgo import cfg
from jaxalgo.datasets._base import BaseDataset
from jaxalgo.datasets._types import Mode
from jaxalgo.datasets.coco._dao import dao_test
from jaxalgo.datasets.coco._dao import dao_train
from jaxalgo.datasets.coco.const import COCO_CATE
from jaxalgo.datasets.coco.const import YOLO_GRIDS
from jaxalgo.datasets.coco.const import YOLO_IN_PX
from jaxalgo.datasets.coco.const import YoloScale
from jaxalgo.yolov3.box import whbox

__all__ = ["CocoDataset", "Yolov3Batch"]

# 3*3*2 tensor repreneting anchors of 3 different scales,
# i.e. small, medium, large; and three different measures.
# Its 1st rank represents three anchor scales and the 2nd rank represents three
# measures of each scale
#
# smalls: ANCHORS[0, ...], mediums: ANCHORS[1, ...] larges: ANCHORS[2, ...]
# [[[0.02403846, 0.03125   ],
#    [0.03846154, 0.07211538],
#    [0.07932692, 0.05528846]],
#
#   [[0.07211538, 0.14663462],
#    [0.14903846, 0.10817308],
#    [0.14182692, 0.28605769]],
#
#   [[0.27884615, 0.21634615],
#    [0.375     , 0.47596154],
#    [0.89663462, 0.78365385]]]
ALL_ANCHORS = jnp.stack(
    [
        YOLO_GRIDS[YoloScale.S].anchors,
        YOLO_GRIDS[YoloScale.M].anchors,
        YOLO_GRIDS[YoloScale.L].anchors,
    ],
    axis=0,
)

SCALES = (YoloScale.S, YoloScale.M, YoloScale.L)
N_IMG_TEST = 40504
N_IMG_TRAIN = 82783
N_MAX_BOX = 20
E = 1e-4

# map from COCO original class ID to class serial number
CATEID_MAP = {cateid: sn for sn, cateid, _ in COCO_CATE}

IDX_X = 0
IDX_Y = 1
IDX_W = 2
IDX_H = 3
IDX_X_OFF = 4
IDX_Y_OFF = 5
IDX_W_EXP = 6
IDX_H_EXP = 7
IDX_CONF = 8
IDX_CLS = 9


class Yolov3Batch(NamedTuple):
    image: jnp.ndarray  # [N, 416, 416, 3]
    label_s: jnp.ndarray  # [N, 52, 52, 3, 10]
    label_m: jnp.ndarray  # [N, 26, 26, 3, 10]
    label_l: jnp.ndarray  # [N, 13, 13, 3, 10]
    gt_bx_s: jnp.ndarray  # [N_MAX_BOX, 10]
    gt_bx_m: jnp.ndarray  # [N_MAX_BOX, 10]
    gt_bx_l: jnp.ndarray  # [N_MAX_BOX, 10]


class CocoDataset(BaseDataset):

    def __init__(self,
                 batch: int = 4,
                 mode: str = "TRAIN",
                 data_dir: str = cfg.DATADIR):
        super().__init__(batch, mode, data_dir)
        self._dao = dao_train if self._mode is Mode.TRAIN else dao_test
        self._n_img = N_IMG_TRAIN if self._mode is Mode.TRAIN else N_IMG_TEST

    @staticmethod
    def readimg(path: str) -> np.ndarray:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)

    @staticmethod
    def max_iou_index(
        wh: jnp.ndarray,
        anchors: jnp.ndarray,
    ) -> jnp.ndarray:
        """Get maximum IoU anchors' indices.

        Args:
            wh (np.ndarray): width-height box of shape [2, ]
            anchors (np.ndarray): anchors array of shape [3, 3, 2]

        Returns:
            jnp.ndarray: array of shape [3, ]

        By comparing a width-height box, get the most similar (i.e. largest
        IoU score) anchors' indices by each of their prier ranks.

        This method assumes that every ground truth box has a UNIQUE
        (center-cell, anchor scale, anchor measure) combination, for otherwise
        they will overwrite each other

        Example:
            criteria: IOU (only of width and height)
            by calculating the IOU between the (3, 3, 2) anchors and one box of
            shape (2, ), e.g. [0.075, 0.075], the result is (3, 3) tensor:

            [[0.13354701, 0.49309665, 0.70710654],
             [0.50122092, 0.34890323, 0.13864692],
             [0.09324138, 0.03151515, 0.00800539]]

            rank it by the last axis (comparing by each row):

            [2, 0, 0]

            It indicate the index of the most similar anchor within each scale:
            [small, 2], [medium, 0] and [large, 0], format: [scale, measure]
        """
        scores = whbox.iou(wh, anchors)
        return jnp.argmax(scores, axis=-1)

    def label_by_id(
        self,
        img_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get for training one label by ID."""
        # TBD: random noise
        n_anchor_per_scale = 3
        ndim_last_rank = 10
        seq_label = [
            np.zeros((s, s, n_anchor_per_scale, ndim_last_rank),
                     dtype=np.float32) for s in SCALES
        ]  # [height, width, n_measure, 10]
        seq_row = self._dao.labels_by_img_id(img_id)
        for row in seq_row:
            indices_measure = self.max_iou_index(
                jnp.array([row.x, row.y], dtype=jnp.float32), ALL_ANCHORS)
            # 0: small, 1: medium, 2: large
            for idx_scale, idx_measure in zip(range(len(SCALES)),
                                              indices_measure,
                                              strict=True):
                scale = SCALES[idx_scale]
                grid = YOLO_GRIDS[scale]
                # offset and top-left gridcell width index
                x_offset, i_ = math.modf(row.x / grid.stride)
                # offset and top-left gridcell height index
                y_offset, j_ = math.modf(row.y / grid.stride)
                i, j = int(i_), int(j_)
                # reverse operation of anchor_w * e^{w_exp}
                w_exp = math.log(row.w / grid.anchors[idx_measure][0] + E)
                # reverse operation of anchor_h * e^{h_exp}
                h_exp = math.log(row.h / grid.anchors[idx_measure][1] + E)
                cate_sn = CATEID_MAP[row.cateid]
                # fill in
                # ATTENTION! j: height, i, width
                seq_label[idx_scale][j, i, idx_measure, IDX_X] = row.x
                seq_label[idx_scale][j, i, idx_measure, IDX_Y] = row.y
                seq_label[idx_scale][j, i, idx_measure, IDX_W] = row.w
                seq_label[idx_scale][j, i, idx_measure, IDX_H] = row.h
                seq_label[idx_scale][j, i, idx_measure, IDX_X_OFF] = x_offset
                seq_label[idx_scale][j, i, idx_measure, IDX_Y_OFF] = y_offset
                seq_label[idx_scale][j, i, idx_measure, IDX_W_EXP] = w_exp
                seq_label[idx_scale][j, i, idx_measure, IDX_H_EXP] = h_exp
                seq_label[idx_scale][j, i, idx_measure, IDX_CONF] = 1.0
                seq_label[idx_scale][j, i, idx_measure, IDX_CLS] = cate_sn

        return tuple(seq_label)

    def _rand_rowids(self, key: KeyArray) -> jnp.ndarray:
        return jax.random.choice(
            key,
            jnp.arange(1, self._n_img + 1),
            shape=(self._batch, ),
            replace=False,
        )

    def _top_rowids(self, offset: int) -> jnp.ndarray:
        start = min(offset + 1, self._n_img)
        stop = min(offset + 1 + self._batch, self._n_img + 1)
        return jnp.arange(start, stop)

    def _get_img(self, rowid: int, size: int) -> tuple[int, np.ndarray]:
        row = self._dao.lookup_image_rowid(rowid)
        if row is None:
            raise StopIteration
        rgb = self.readimg(row.path)
        return (row.imageid,
                cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA))

    @staticmethod
    def _get_gt_box(lab: jnp.ndarray) -> jnp.ndarray:
        lab_ = lab[lab[..., IDX_CONF] == 1]
        n_replica = N_MAX_BOX // lab.shape[0] + 1
        return jnp.tile(lab_, (n_replica, 1))[:N_MAX_BOX, :]

    def _fetch(self, rowids: jnp.ndarray) -> Yolov3Batch:
        images, labels_s, labels_m, labels_l = [], [], [], []
        for i in rowids:
            img_id, rgb_resized = self._get_img(i.item(), YOLO_IN_PX)
            label_s, label_m, label_l = self.label_by_id(img_id=img_id)
            images += [rgb_resized]
            labels_s += [label_s]
            labels_m += [label_m]
            labels_l += [label_l]

        images_ = jnp.stack(images)
        labels_s_ = jnp.stack(labels_s)
        labels_m_ = jnp.stack(labels_m)
        labels_l_ = jnp.stack(labels_l)
        return Yolov3Batch(
            images_,
            labels_s_,
            labels_m_,
            labels_l_,
            self._get_gt_box(labels_s_),
            self._get_gt_box(labels_m_),
            self._get_gt_box(labels_l_),
        )

    def rand_batch(self, key: KeyArray) -> Yolov3Batch:
        return self._fetch(self._rand_rowids(key))

    def top_batch(self, offset: int) -> Yolov3Batch:
        return self._fetch(self._top_rowids(offset))
