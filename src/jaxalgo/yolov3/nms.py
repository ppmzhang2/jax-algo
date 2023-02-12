import jax.numpy as jnp
import numpy as np

from jaxalgo.yolov3.box import bbox


def _unique_class(boxes: jnp.ndarray) -> set:
    cates = bbox.class_sn(boxes, squeezed=True)
    return set(np.unique(cates))


def nms_1class(boxes: jnp.ndarray, iou_th: float) -> jnp.ndarray:
    """Non-maximum suppression for only one class.

    Args:
        boxes (jnp.ndarray): 2nd order tensor
        iou_th (float): IoU threshold
    """

    def _valid_indices(
        idx: int,
        indices: jnp.ndarray,
        boxes: jnp.ndarray,
        iou_th: float,
        acc: jnp.ndarray,
    ) -> tuple[jnp.ndarray, tuple]:
        """Returns valid indices"""
        ious = bbox.iou(boxes[idx, :], boxes[indices, :])
        return indices[ious < iou_th], jnp.array([*acc, idx])

    confs = bbox.conf1d(boxes)
    indices_ = jnp.argsort(-confs, axis=-1)

    acc = jnp.array([])
    while True:
        if indices_.size == 0:
            return boxes[acc]
        idx, indices_ = indices_[0], indices_[1:]  # idx will always be saved
        indices_, acc = _valid_indices(idx, indices_, boxes, iou_th, acc)


def nms(boxes: jnp.ndarray, iou_th: float) -> jnp.ndarray:
    cate_sns = _unique_class(boxes)  # class SNs
    seq = []
    for sn in cate_sns:
        t = nms_1class(bbox.classof(boxes, sn), iou_th)
        if t.size != 0:
            seq.append(t)

    return jnp.concatenate(seq, axis=0)


def _three2one_1img(
    ys: jnp.ndarray,
    ym: jnp.ndarray,
    yl: jnp.ndarray,
    conf_th: float,
    iou_th: float,
    *,
    from_logits: bool = False,
) -> jnp.ndarray:
    """Combile three label tensors into one.

    The input three tensors represent three grid size of a single image.

    Args:
        ys (jnp.ndarray): (52, 52, 3, 10)
        ym (jnp.ndarray): (26, 26, 3, 10)
        yl (jnp.ndarray): (13, 13, 3, 10)
        conf_th (float): confidence threshold
        iou_th (float): IoU threshold
        from_logits (bool): True if confidence score is in logit format 
    """
    if from_logits:

        def fn_objects(bx: jnp.ndarray) -> jnp.ndarray:
            return bbox.objects(bx, from_logits=True, conf_th=conf_th)
    else:

        def fn_objects(bx: jnp.ndarray) -> jnp.ndarray:
            return bbox.objects(bx, conf_th=conf_th)

    lab = jnp.concatenate(
        [fn_objects(ys), fn_objects(ym),
         fn_objects(yl)], axis=0)
    return nms(lab, iou_th)


def three2one(
    ys: jnp.ndarray,
    ym: jnp.ndarray,
    yl: jnp.ndarray,
    conf_th: float,
    iou_th: float,
    *,
    from_logits: bool = False,
) -> tuple[jnp.ndarray, ...]:
    """Combile three label tensors into one."""
    n_batch = ys.shape[0]
    return tuple(
        _three2one_1img(
            ys[i, ...],
            ym[i, ...],
            yl[i, ...],
            conf_th,
            iou_th,
            from_logits=from_logits,
        ) for i in range(n_batch))
