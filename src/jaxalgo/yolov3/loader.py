"""Load model weights."""
from abc import abstractmethod
from dataclasses import dataclass
from typing import TypeAlias

import cv2
import jax.numpy as jnp
import numpy as np

__all__ = ["load_cv_var"]

ParamState: TypeAlias = dict[str, dict[str, jnp.ndarray]]


@dataclass(frozen=True)
class LayerMap:
    """One layer of opencv and tensorflow layer names respectively."""
    cv: str
    hk: str

    def _cv_layer(self, net: cv2.dnn.Net) -> cv2.dnn.Layer:
        return net.getLayer(self.cv)

    def _cv_vars(self, net: cv2.dnn.Net) -> tuple[np.ndarray, ...]:
        """CV parameters and states."""
        return self._cv_layer(net).blobs

    @abstractmethod
    def hk_params(self, net: cv2.dnn.Net) -> ParamState:
        """Get haiku parameters."""

    @abstractmethod
    def hk_states(self, net: cv2.dnn.Net) -> ParamState:
        """Get haiku states."""


class BnLayerMap(LayerMap):
    """Batch Normalization Layer Mapping."""

    def cv_mean(self, net: cv2.dnn.Net) -> np.ndarray:
        """Get BN mean in opencv format."""
        return self._cv_vars(net)[0]

    def cv_var(self, net: cv2.dnn.Net) -> np.ndarray:
        """Get BN var in opencv format."""
        return self._cv_vars(net)[1]

    def cv_gamma(self, net: cv2.dnn.Net) -> np.ndarray:
        """Get BN gamma in opencv format."""
        return self._cv_vars(net)[2]

    def cv_beta(self, net: cv2.dnn.Net) -> np.ndarray:
        """Get BN beta in opencv format."""
        return self._cv_vars(net)[3]

    @staticmethod
    def _cv2hk(arr: np.ndarray) -> jnp.ndarray:
        """Convert opencv BN variables to haiku format.

        BN variable format in cv: (1, channel)
        BN variable format in hk: (1, 1, 1, channel)
        """
        return jnp.array(arr, dtype=jnp.float32)[jnp.newaxis, jnp.newaxis, ...]

    def hk_scale(self, net: cv2.dnn.Net) -> jnp.ndarray:
        """Get BN gamma in haiku format."""
        return self._cv2hk(self.cv_gamma(net))

    def hk_offset(self, net: cv2.dnn.Net) -> jnp.ndarray:
        """Get BN beta in haiku format."""
        return self._cv2hk(self.cv_beta(net))

    def hk_mean(self, net: cv2.dnn.Net) -> jnp.ndarray:
        """Get BN mean in haiku format."""
        return self._cv2hk(self.cv_mean(net))

    def hk_var(self, net: cv2.dnn.Net) -> jnp.ndarray:
        """Get BN var in haiku format."""
        return self._cv2hk(self.cv_var(net))

    def hk_params(self, net: cv2.dnn.Net) -> ParamState:
        """Get haiku parameters."""
        return {
            self.hk: {
                "scale": self.hk_scale(net),
                "offset": self.hk_offset(net),
            },
        }

    def hk_states(self, net: cv2.dnn.Net) -> ParamState:
        """Get haiku states."""
        k_mean = "/".join([self.hk, "~", "mean_ema"])
        k_var = "/".join([self.hk, "~", "var_ema"])
        k_avg = "average"
        k_hid = "hidden"
        k_cnt = "counter"
        mean, var = self.hk_mean(net), self.hk_var(net)
        hid = jnp.zeros_like(mean, dtype=jnp.float32)
        cnt = jnp.array(0, dtype=jnp.int32)
        return {
            k_mean: {
                k_avg: mean,
                k_hid: hid,
                k_cnt: cnt,
            },
            k_var: {
                k_avg: var,
                k_hid: hid,
                k_cnt: cnt,
            },
        }


class ConvLayerMap(LayerMap):
    """Convolutional Layer Mapping."""

    def cv_weight(self, net: cv2.dnn.Net) -> np.ndarray:
        """Get Conv2D weight in opencv format."""
        return self._cv_vars(net)[0]

    def cv_bias(self, net: cv2.dnn.Net) -> np.ndarray | None:
        """Get Conv2D bias in opencv format."""
        try:
            return self._cv_vars(net)[1]
        except IndexError:
            return None

    def hk_weight(self, net: cv2.dnn.Net) -> jnp.ndarray:
        """Transform open-cv conv weights into haiku ones.

        permutating ranks:
        cv Conv weigh format: (channel_out, channel_in, height, width)
        hk Conv weight format: (height, width, channel_in, channel_out)
        """
        return jnp.transpose(self.cv_weight(net),
                             [2, 3, 1, 0]).astype(jnp.float32)

    def hk_bias(self, net: cv2.dnn.Net) -> jnp.ndarray | None:
        """Transform open-cv conv bias into haiku ones.

        cv Conv bias format: (1, channel_in)
        hk Conv bias format: (channel_in, ) i.e. squeezed vector
        """
        b = self.cv_bias(net)
        if b is None:
            return None
        return jnp.squeeze(self.cv_bias(net), axis=0).astype(jnp.float32)

    def hk_params(self, net: cv2.dnn.Net) -> ParamState:
        """Get haiku parameters."""
        w, b = self.hk_weight(net), self.hk_bias(net)
        if b is None:
            return {self.hk: {"w": w}}
        return {self.hk: {"w": w, "b": b}}

    def hk_states(self, _: cv2.dnn.Net) -> ParamState:
        """Get haiku states."""
        return {}


layers: tuple[LayerMap, ...] = (
    ConvLayerMap(
        cv="conv_0",
        hk="yolo_v3/dn52_block/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_0",
        hk="yolo_v3/dn52_block/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_1",
        hk="yolo_v3/dn52_block/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_1",
        hk="yolo_v3/dn52_block/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_2",
        hk="yolo_v3/dn52_block/res_block/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_2",
        hk="yolo_v3/dn52_block/res_block/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_3",
        hk="yolo_v3/dn52_block/res_block/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_3",
        hk="yolo_v3/dn52_block/res_block/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_5",
        hk="yolo_v3/dn52_block/cnn_block_2/conv2_d",
    ),
    BnLayerMap(
        cv="bn_5",
        hk="yolo_v3/dn52_block/cnn_block_2/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_6",
        hk="yolo_v3/dn52_block/res_block_1/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_6",
        hk="yolo_v3/dn52_block/res_block_1/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_7",
        hk="yolo_v3/dn52_block/res_block_1/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_7",
        hk="yolo_v3/dn52_block/res_block_1/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_9",
        hk="yolo_v3/dn52_block/res_block_2/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_9",
        hk="yolo_v3/dn52_block/res_block_2/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_10",
        hk="yolo_v3/dn52_block/res_block_2/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_10",
        hk="yolo_v3/dn52_block/res_block_2/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_12",
        hk="yolo_v3/dn52_block/cnn_block_3/conv2_d",
    ),
    BnLayerMap(
        cv="bn_12",
        hk="yolo_v3/dn52_block/cnn_block_3/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_13",
        hk="yolo_v3/dn52_block/res_block_3/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_13",
        hk="yolo_v3/dn52_block/res_block_3/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_14",
        hk="yolo_v3/dn52_block/res_block_3/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_14",
        hk="yolo_v3/dn52_block/res_block_3/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_16",
        hk="yolo_v3/dn52_block/res_block_4/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_16",
        hk="yolo_v3/dn52_block/res_block_4/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_17",
        hk="yolo_v3/dn52_block/res_block_4/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_17",
        hk="yolo_v3/dn52_block/res_block_4/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_19",
        hk="yolo_v3/dn52_block/res_block_5/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_19",
        hk="yolo_v3/dn52_block/res_block_5/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_20",
        hk="yolo_v3/dn52_block/res_block_5/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_20",
        hk="yolo_v3/dn52_block/res_block_5/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_22",
        hk="yolo_v3/dn52_block/res_block_6/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_22",
        hk="yolo_v3/dn52_block/res_block_6/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_23",
        hk="yolo_v3/dn52_block/res_block_6/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_23",
        hk="yolo_v3/dn52_block/res_block_6/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_25",
        hk="yolo_v3/dn52_block/res_block_7/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_25",
        hk="yolo_v3/dn52_block/res_block_7/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_26",
        hk="yolo_v3/dn52_block/res_block_7/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_26",
        hk="yolo_v3/dn52_block/res_block_7/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_28",
        hk="yolo_v3/dn52_block/res_block_8/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_28",
        hk="yolo_v3/dn52_block/res_block_8/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_29",
        hk="yolo_v3/dn52_block/res_block_8/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_29",
        hk="yolo_v3/dn52_block/res_block_8/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_31",
        hk="yolo_v3/dn52_block/res_block_9/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_31",
        hk="yolo_v3/dn52_block/res_block_9/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_32",
        hk="yolo_v3/dn52_block/res_block_9/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_32",
        hk="yolo_v3/dn52_block/res_block_9/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_34",
        hk="yolo_v3/dn52_block/res_block_10/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_34",
        hk="yolo_v3/dn52_block/res_block_10/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_35",
        hk="yolo_v3/dn52_block/res_block_10/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_35",
        hk="yolo_v3/dn52_block/res_block_10/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_37",
        hk="yolo_v3/dn52_block/cnn_block_4/conv2_d",
    ),
    BnLayerMap(
        cv="bn_37",
        hk="yolo_v3/dn52_block/cnn_block_4/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_38",
        hk="yolo_v3/dn52_block/res_block_11/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_38",
        hk="yolo_v3/dn52_block/res_block_11/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_39",
        hk="yolo_v3/dn52_block/res_block_11/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_39",
        hk="yolo_v3/dn52_block/res_block_11/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_41",
        hk="yolo_v3/dn52_block/res_block_12/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_41",
        hk="yolo_v3/dn52_block/res_block_12/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_42",
        hk="yolo_v3/dn52_block/res_block_12/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_42",
        hk="yolo_v3/dn52_block/res_block_12/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_44",
        hk="yolo_v3/dn52_block/res_block_13/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_44",
        hk="yolo_v3/dn52_block/res_block_13/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_45",
        hk="yolo_v3/dn52_block/res_block_13/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_45",
        hk="yolo_v3/dn52_block/res_block_13/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_47",
        hk="yolo_v3/dn52_block/res_block_14/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_47",
        hk="yolo_v3/dn52_block/res_block_14/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_48",
        hk="yolo_v3/dn52_block/res_block_14/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_48",
        hk="yolo_v3/dn52_block/res_block_14/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_50",
        hk="yolo_v3/dn52_block/res_block_15/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_50",
        hk="yolo_v3/dn52_block/res_block_15/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_51",
        hk="yolo_v3/dn52_block/res_block_15/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_51",
        hk="yolo_v3/dn52_block/res_block_15/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_53",
        hk="yolo_v3/dn52_block/res_block_16/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_53",
        hk="yolo_v3/dn52_block/res_block_16/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_54",
        hk="yolo_v3/dn52_block/res_block_16/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_54",
        hk="yolo_v3/dn52_block/res_block_16/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_56",
        hk="yolo_v3/dn52_block/res_block_17/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_56",
        hk="yolo_v3/dn52_block/res_block_17/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_57",
        hk="yolo_v3/dn52_block/res_block_17/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_57",
        hk="yolo_v3/dn52_block/res_block_17/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_59",
        hk="yolo_v3/dn52_block/res_block_18/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_59",
        hk="yolo_v3/dn52_block/res_block_18/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_60",
        hk="yolo_v3/dn52_block/res_block_18/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_60",
        hk="yolo_v3/dn52_block/res_block_18/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_62",
        hk="yolo_v3/dn52_block/cnn_block_5/conv2_d",
    ),
    BnLayerMap(
        cv="bn_62",
        hk="yolo_v3/dn52_block/cnn_block_5/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_63",
        hk="yolo_v3/dn52_block/res_block_19/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_63",
        hk="yolo_v3/dn52_block/res_block_19/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_64",
        hk="yolo_v3/dn52_block/res_block_19/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_64",
        hk="yolo_v3/dn52_block/res_block_19/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_66",
        hk="yolo_v3/dn52_block/res_block_20/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_66",
        hk="yolo_v3/dn52_block/res_block_20/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_67",
        hk="yolo_v3/dn52_block/res_block_20/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_67",
        hk="yolo_v3/dn52_block/res_block_20/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_69",
        hk="yolo_v3/dn52_block/res_block_21/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_69",
        hk="yolo_v3/dn52_block/res_block_21/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_70",
        hk="yolo_v3/dn52_block/res_block_21/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_70",
        hk="yolo_v3/dn52_block/res_block_21/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_72",
        hk="yolo_v3/dn52_block/res_block_22/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_72",
        hk="yolo_v3/dn52_block/res_block_22/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_73",
        hk="yolo_v3/dn52_block/res_block_22/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_73",
        hk="yolo_v3/dn52_block/res_block_22/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_75",
        hk="yolo_v3/cnn_block/conv2_d",
    ),
    BnLayerMap(
        cv="bn_75",
        hk="yolo_v3/cnn_block/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_76",
        hk="yolo_v3/cnn_block_1/conv2_d",
    ),
    BnLayerMap(
        cv="bn_76",
        hk="yolo_v3/cnn_block_1/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_77",
        hk="yolo_v3/cnn_block_2/conv2_d",
    ),
    BnLayerMap(
        cv="bn_77",
        hk="yolo_v3/cnn_block_2/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_78",
        hk="yolo_v3/cnn_block_3/conv2_d",
    ),
    BnLayerMap(
        cv="bn_78",
        hk="yolo_v3/cnn_block_3/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_79",
        hk="yolo_v3/cnn_block_4/conv2_d",
    ),
    BnLayerMap(
        cv="bn_79",
        hk="yolo_v3/cnn_block_4/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_80",
        hk="yolo_v3/cnn_block_5/conv2_d",
    ),
    BnLayerMap(
        cv="bn_80",
        hk="yolo_v3/cnn_block_5/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_81",
        hk="yolo_v3/cnn_block_6/conv2_d",
    ),
    ConvLayerMap(
        cv="conv_84",
        hk="yolo_v3/cnn_block_7/conv2_d",
    ),
    BnLayerMap(
        cv="bn_84",
        hk="yolo_v3/cnn_block_7/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_87",
        hk="yolo_v3/cnn_block_8/conv2_d",
    ),
    BnLayerMap(
        cv="bn_87",
        hk="yolo_v3/cnn_block_8/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_88",
        hk="yolo_v3/cnn_block_9/conv2_d",
    ),
    BnLayerMap(
        cv="bn_88",
        hk="yolo_v3/cnn_block_9/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_89",
        hk="yolo_v3/cnn_block_10/conv2_d",
    ),
    BnLayerMap(
        cv="bn_89",
        hk="yolo_v3/cnn_block_10/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_90",
        hk="yolo_v3/cnn_block_11/conv2_d",
    ),
    BnLayerMap(
        cv="bn_90",
        hk="yolo_v3/cnn_block_11/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_91",
        hk="yolo_v3/cnn_block_12/conv2_d",
    ),
    BnLayerMap(
        cv="bn_91",
        hk="yolo_v3/cnn_block_12/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_92",
        hk="yolo_v3/cnn_block_13/conv2_d",
    ),
    BnLayerMap(
        cv="bn_92",
        hk="yolo_v3/cnn_block_13/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_93",
        hk="yolo_v3/cnn_block_14/conv2_d",
    ),
    ConvLayerMap(
        cv="conv_96",
        hk="yolo_v3/cnn_block_15/conv2_d",
    ),
    BnLayerMap(
        cv="bn_96",
        hk="yolo_v3/cnn_block_15/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_99",
        hk="yolo_v3/cnn_block_16/conv2_d",
    ),
    BnLayerMap(
        cv="bn_99",
        hk="yolo_v3/cnn_block_16/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_100",
        hk="yolo_v3/cnn_block_17/conv2_d",
    ),
    BnLayerMap(
        cv="bn_100",
        hk="yolo_v3/cnn_block_17/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_101",
        hk="yolo_v3/cnn_block_18/conv2_d",
    ),
    BnLayerMap(
        cv="bn_101",
        hk="yolo_v3/cnn_block_18/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_102",
        hk="yolo_v3/cnn_block_19/conv2_d",
    ),
    BnLayerMap(
        cv="bn_102",
        hk="yolo_v3/cnn_block_19/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_103",
        hk="yolo_v3/cnn_block_20/conv2_d",
    ),
    BnLayerMap(
        cv="bn_103",
        hk="yolo_v3/cnn_block_20/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_104",
        hk="yolo_v3/cnn_block_21/conv2_d",
    ),
    BnLayerMap(
        cv="bn_104",
        hk="yolo_v3/cnn_block_21/batch_norm",
    ),
    ConvLayerMap(
        cv="conv_105",
        hk="yolo_v3/cnn_block_22/conv2_d",
    ),
)


def load_cv_var(net: cv2.dnn.Net) -> tuple[ParamState, ParamState]:
    """Load weights from an open-cv model into a haiku format."""
    params = {k: v for tp in layers for k, v in tp.hk_params(net).items()}
    states = {k: v for tp in layers for k, v in tp.hk_states(net).items()}
    return params, states
