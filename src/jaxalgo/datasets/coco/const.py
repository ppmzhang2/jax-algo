"""YOLOv3 constants"""
from enum import IntEnum
from typing import NamedTuple

import jax.numpy as jnp

# YOLO constants
# sequences all in the same order: small, medium, large


class YoloScale(IntEnum):
    S = 52
    M = 26
    L = 13


class YoloGrid(NamedTuple):
    size: int
    stride: float
    anchors: jnp.ndarray  # 3*2 matrix


YOLO_GRIDS = {
    YoloScale.S:
    YoloGrid(
        size=52,
        stride=0.01923,  # 8 / 416
        anchors=jnp.array([
            [0.02404, 0.03125],  # (10 / 416, 13 / 416)
            [0.03846, 0.07212],  # (16 / 416, 30 / 416)
            [0.07933, 0.05529],  # (33 / 416, 23 / 416)
        ])),
    YoloScale.M:
    YoloGrid(
        size=26,
        stride=0.03846,  # 16 / 416
        anchors=jnp.array([
            [0.07212, 0.14663],  # (30 / 416, 61 / 416)
            [0.14904, 0.10817],  # (62 / 416, 45 / 416)
            [0.14183, 0.28606],  # (59 / 416, 119 / 416)
        ])),
    YoloScale.L:
    YoloGrid(
        size=13,
        stride=0.07692,  # 32 / 416
        anchors=jnp.array([
            [0.27885, 0.21635],  # (116 / 416, 90 / 416)
            [0.37500, 0.47596],  # (156 / 416, 198 / 416)
            [0.89663, 0.78365],  # (373 / 416, 326 / 416)
        ])),
}

YOLO_IN_PX = 416  # input pixel resolution: 416 by 416

# COCO category mapping: SN, class ID, class name
COCO_CATE = (
    (1, 1, "person"),
    (2, 2, "bicycle"),
    (3, 3, "car"),
    (4, 4, "motorcycle"),
    (5, 5, "airplane"),
    (6, 6, "bus"),
    (7, 7, "train"),
    (8, 8, "truck"),
    (9, 9, "boat"),
    (10, 10, "traffic light"),
    (11, 11, "fire hydrant"),
    (12, 13, "stop sign"),
    (13, 14, "parking meter"),
    (14, 15, "bench"),
    (15, 16, "bird"),
    (16, 17, "cat"),
    (17, 18, "dog"),
    (18, 19, "horse"),
    (19, 20, "sheep"),
    (20, 21, "cow"),
    (21, 22, "elephant"),
    (22, 23, "bear"),
    (23, 24, "zebra"),
    (24, 25, "giraffe"),
    (25, 27, "backpack"),
    (26, 28, "umbrella"),
    (27, 31, "handbag"),
    (28, 32, "tie"),
    (29, 33, "suitcase"),
    (30, 34, "frisbee"),
    (31, 35, "skis"),
    (32, 36, "snowboard"),
    (33, 37, "sports ball"),
    (34, 38, "kite"),
    (35, 39, "baseball bat"),
    (36, 40, "baseball glove"),
    (37, 41, "skateboard"),
    (38, 42, "surfboard"),
    (39, 43, "tennis racket"),
    (40, 44, "bottle"),
    (41, 46, "wine glass"),
    (42, 47, "cup"),
    (43, 48, "fork"),
    (44, 49, "knife"),
    (45, 50, "spoon"),
    (46, 51, "bowl"),
    (47, 52, "banana"),
    (48, 53, "apple"),
    (49, 54, "sandwich"),
    (50, 55, "orange"),
    (51, 56, "broccoli"),
    (52, 57, "carrot"),
    (53, 58, "hot dog"),
    (54, 59, "pizza"),
    (55, 60, "donut"),
    (56, 61, "cake"),
    (57, 62, "chair"),
    (58, 63, "couch"),
    (59, 64, "potted plant"),
    (60, 65, "bed"),
    (61, 67, "dining table"),
    (62, 70, "toilet"),
    (63, 72, "tv"),
    (64, 73, "laptop"),
    (65, 74, "mouse"),
    (66, 75, "remote"),
    (67, 76, "keyboard"),
    (68, 77, "cell phone"),
    (69, 78, "microwave"),
    (70, 79, "oven"),
    (71, 80, "toaster"),
    (72, 81, "sink"),
    (73, 82, "refrigerator"),
    (74, 84, "book"),
    (75, 85, "clock"),
    (76, 86, "vase"),
    (77, 87, "scissors"),
    (78, 88, "teddy bear"),
    (79, 89, "hair drier"),
    (80, 90, "toothbrush"),
)
