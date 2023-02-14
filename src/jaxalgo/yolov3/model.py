"""YOLOv3 Model."""
from typing import NoReturn

import haiku as hk
import jax
import jax.numpy as jnp


class CnnBlock(hk.Module):
    """Conv2D Block."""

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        *,
        downsample: bool = False,
        bn: bool = True,
    ) -> NoReturn:
        """Instantiate a `ResBlock`.

        Args:
            filters (int): number of output channels
            kernel_size (int): kernel size, e.g. (3, 3), (1, 1)
            downsample (bool): down-sampling flag i.e. set stride to 2 if
                downsample add stride 1 otherwise.
                Always do the 0 padding i.e. padding = 'same'
            bn (bool): batch-normalization flag, True means:
                1. this layer is for prediction
                2. use batch-normalization
                3. Leaky-ReLU activation
        """
        super().__init__()
        self._filters = filters
        self._kernel_size = kernel_size
        self._downsample = downsample
        self._bn = bn

    def __call__(self, x: jnp.ndarray, *, training: bool = True):
        """Transform input."""
        y = hk.Conv2D(output_channels=self._filters,
                      kernel_shape=(self._kernel_size, self._kernel_size),
                      stride=2 if self._downsample else 1,
                      with_bias=not self._bn,
                      padding="SAME")(x)
        if self._bn:
            y = hk.BatchNorm(
                create_scale=True,
                create_offset=True,
                decay_rate=0.9,
            )(y, is_training=training)
            y = jax.nn.leaky_relu(y)
        return y


class ResBlock(hk.Module):
    """Residual Network Block.

    In YOLOv3, the residual network block has two consecutive convolution
    layers which:
        1. half the channels with kernel size 1
        2. double the channels with kernel size 3. i.e. resume to the original
           number of channels
        3. add the original input and the output of last step
    """

    def __init__(self, channels: int) -> NoReturn:
        """Instantiate a `ResBlock`.

        Args:
            channels (int): number of output (input) channels
        """
        super().__init__()
        self._channels = channels

    def __call__(self, x: jnp.ndarray, *, training: bool = True):
        """Transform input."""
        y = CnnBlock(self._channels // 2, 1)(x, training=training)
        y = CnnBlock(self._channels, 3)(y, training=training)
        return x + y


class Dn52Block(hk.Module):
    """DarkNet53 block."""

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        training: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Transform input.

        Args:
            x (jnp.ndarray): input tensor
            training (bool): training flag

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: two intermediate
            results and one final result:
                1. intermediate one: 256 out channels, after the first eight
                   residual block
                2. intermediate two: 512 out channels, after the second eight
                   residual blocks
                3. final result: 1024 out channels
        """
        # ---------------------------------------------------------------------
        # yolo_v3/dn52_block/cnn_block
        # ---------------------------------------------------------------------
        y = CnnBlock(32, 3)(x, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/dn52_block/cnn_block_1
        # ---------------------------------------------------------------------
        y = CnnBlock(64, 3, downsample=True)(y, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/dn52_block/res_block/cnn_block
        # yolo_v3/dn52_block/res_block/cnn_block_1
        # ---------------------------------------------------------------------
        y = ResBlock(64)(y, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/dn52_block/cnn_block_2
        # ---------------------------------------------------------------------
        y = CnnBlock(128, 3, downsample=True)(y, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/dn52_block/res_block_1/cnn_block
        # yolo_v3/dn52_block/res_block_1/cnn_block_1
        # yolo_v3/dn52_block/res_block_2/cnn_block
        # yolo_v3/dn52_block/res_block_2/cnn_block_1
        # ---------------------------------------------------------------------
        for _ in range(2):
            y = ResBlock(128)(y, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/dn52_block/cnn_block_3
        # ---------------------------------------------------------------------
        y = CnnBlock(256, 3, downsample=True)(y, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/dn52_block/res_block_3/cnn_block
        # yolo_v3/dn52_block/res_block_3/cnn_block_1
        # yolo_v3/dn52_block/res_block_4/cnn_block
        # yolo_v3/dn52_block/res_block_4/cnn_block_1
        # yolo_v3/dn52_block/res_block_5/cnn_block
        # yolo_v3/dn52_block/res_block_5/cnn_block_1
        # yolo_v3/dn52_block/res_block_6/cnn_block
        # yolo_v3/dn52_block/res_block_6/cnn_block_1
        # yolo_v3/dn52_block/res_block_7/cnn_block
        # yolo_v3/dn52_block/res_block_7/cnn_block_1
        # yolo_v3/dn52_block/res_block_8/cnn_block
        # yolo_v3/dn52_block/res_block_8/cnn_block_1
        # yolo_v3/dn52_block/res_block_9/cnn_block
        # yolo_v3/dn52_block/res_block_9/cnn_block_1
        # yolo_v3/dn52_block/res_block_10/cnn_block
        # yolo_v3/dn52_block/res_block_10/cnn_block_1
        # ---------------------------------------------------------------------
        for _ in range(8):
            y = ResBlock(256)(y, training=training)
        inter_res_1 = y
        # ---------------------------------------------------------------------
        # yolo_v3/dn52_block/cnn_block_4
        # ---------------------------------------------------------------------
        y = CnnBlock(512, 3, downsample=True)(y, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/dn52_block/res_block_11/cnn_block
        # yolo_v3/dn52_block/res_block_11/cnn_block_1
        # yolo_v3/dn52_block/res_block_12/cnn_block
        # yolo_v3/dn52_block/res_block_12/cnn_block_1
        # yolo_v3/dn52_block/res_block_13/cnn_block
        # yolo_v3/dn52_block/res_block_13/cnn_block_1
        # yolo_v3/dn52_block/res_block_14/cnn_block
        # yolo_v3/dn52_block/res_block_14/cnn_block_1
        # yolo_v3/dn52_block/res_block_15/cnn_block
        # yolo_v3/dn52_block/res_block_15/cnn_block_1
        # yolo_v3/dn52_block/res_block_16/cnn_block
        # yolo_v3/dn52_block/res_block_16/cnn_block_1
        # yolo_v3/dn52_block/res_block_17/cnn_block
        # yolo_v3/dn52_block/res_block_17/cnn_block_1
        # yolo_v3/dn52_block/res_block_18/cnn_block
        # yolo_v3/dn52_block/res_block_18/cnn_block_1
        # ---------------------------------------------------------------------
        for _ in range(8):
            y = ResBlock(512)(y, training=training)
        inter_res_2 = y
        # ---------------------------------------------------------------------
        # yolo_v3/dn52_block/cnn_block_5
        # ---------------------------------------------------------------------
        y = CnnBlock(1024, 3, downsample=True)(y, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/dn52_block/res_block_19/cnn_block
        # yolo_v3/dn52_block/res_block_19/cnn_block_1
        # yolo_v3/dn52_block/res_block_20/cnn_block
        # yolo_v3/dn52_block/res_block_20/cnn_block_1
        # yolo_v3/dn52_block/res_block_21/cnn_block
        # yolo_v3/dn52_block/res_block_21/cnn_block_1
        # yolo_v3/dn52_block/res_block_22/cnn_block
        # yolo_v3/dn52_block/res_block_22/cnn_block_1
        # ---------------------------------------------------------------------
        for _ in range(4):
            y = ResBlock(1024)(y, training=training)

        return inter_res_1, inter_res_2, y


class UpSample2D(hk.Module):
    """2D upsampling."""

    def __call__(self, x: jnp.ndarray):
        """Transform input."""
        n_, h_, w_, c_ = x.shape
        out_shape = (n_, h_ * 2, w_ * 2, c_)
        return jax.image.resize(x, out_shape, method="nearest")


class YoloV3(hk.Module):
    """YOLOv3 network."""

    def __init__(self, n_class: int = 80):
        """Instantiate a `YoloV3`."""
        super().__init__()
        self._n_class = n_class

    @staticmethod
    def _reshape(x: jnp.ndarray) -> jnp.ndarray:
        """Reshape the output as [N, W, H, 3, N_CLASS+5]."""
        shape = x.shape
        return x.reshape(shape[0], shape[1], shape[2], 3, -1)

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        training: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Transform input.

        Args:
            x (jnp.ndarray): input tensor
            training (bool): training flag

        Returns:
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: output labels of
                small, medium and large scales respectively
        """
        # number of channels for prediction
        pred_channels = 3 * (self._n_class + 5)

        inter_1, inter_2, x_ = Dn52Block()(x, training=training)

        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L551-L557
        # ---------------------------------------------------------------------
        x_ = CnnBlock(512, 1)(x_, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_1
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L559-L565
        # ---------------------------------------------------------------------
        x_ = CnnBlock(1024, 3)(x_, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_2
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L567-L573
        # ---------------------------------------------------------------------
        x_ = CnnBlock(512, 1)(x_, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_3
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L575-L581
        # ---------------------------------------------------------------------
        x_ = CnnBlock(1024, 3)(x_, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_4
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L583-L589
        # ---------------------------------------------------------------------
        x_ = CnnBlock(512, 1)(x_, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_5
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L591-L597
        # ---------------------------------------------------------------------
        x_l = CnnBlock(1024, 3)(x_, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_6
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L599-L604
        # predict large-sized objects; shape = [None, 13, 13, 255]
        # ---------------------------------------------------------------------
        box_l = CnnBlock(pred_channels, 1, bn=False)(x_l, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_7
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L618-L627
        # ---------------------------------------------------------------------
        x_ = CnnBlock(256, 1)(x_, training=training)

        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L629-L630
        x_ = UpSample2D()(x_)
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L632-L633
        x_ = jnp.concatenate([x_, inter_2], axis=-1)

        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_8
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L637-L643
        # ---------------------------------------------------------------------
        x_ = CnnBlock(256, 1)(x_, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_9
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L645-L651
        # ---------------------------------------------------------------------
        x_ = CnnBlock(512, 3)(x_, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_10
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L653-L659
        # ---------------------------------------------------------------------
        x_ = CnnBlock(256, 1)(x_, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_11
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L661-L667
        # ---------------------------------------------------------------------
        x_ = CnnBlock(512, 3)(x_, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_12
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L669-L675
        # ---------------------------------------------------------------------
        x_ = CnnBlock(256, 1)(x_, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_13
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L677-L683
        # ---------------------------------------------------------------------
        x_m = CnnBlock(512, 3)(x_, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_14
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L685-L690
        # medium-sized objects, shape = [None, 26, 26, 255]
        # ---------------------------------------------------------------------
        box_m = CnnBlock(pred_channels, 1, bn=False)(x_m, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_15
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L705-L714
        # ---------------------------------------------------------------------
        x_ = CnnBlock(128, 1)(x_, training=training)

        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L716-L717
        x_ = UpSample2D()(x_)
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L719-L720
        x_ = jnp.concatenate([x_, inter_1], axis=-1)

        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_16
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L724-L730
        # ---------------------------------------------------------------------
        x_ = CnnBlock(128, 1)(x_, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_17
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L732-L738
        # ---------------------------------------------------------------------
        x_ = CnnBlock(256, 3)(x_, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_18
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L740-L746
        # ---------------------------------------------------------------------
        x_ = CnnBlock(128, 1)(x_, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_19
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L748-L754
        # ---------------------------------------------------------------------
        x_ = CnnBlock(256, 3)(x_, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_20
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L756-L762
        # ---------------------------------------------------------------------
        x_ = CnnBlock(128, 1)(x_, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_21
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L764-L770
        # ---------------------------------------------------------------------
        x_s = CnnBlock(256, 3)(x_, training=training)
        # ---------------------------------------------------------------------
        # yolo_v3/cnn_block_22
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L772-L777
        # predict small size objects, shape = [None, 52, 52, 255]
        # ---------------------------------------------------------------------
        box_s = CnnBlock(pred_channels, 1, bn=False)(x_s, training=training)

        return tuple(map(self._reshape, (box_s, box_m, box_l)))
