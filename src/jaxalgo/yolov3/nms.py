import jax
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
    confs = bbox.conf1d(boxes)
    # indices to keep, only deleted ones should be removed
    indices_keep = jnp.argsort(-confs, axis=-1)
    # indices for tracking, both added and deleted should be removed from it
    indices_track = jnp.copy(indices_keep)

    for idx1 in indices_keep:
        if indices_track.size == 0:
            break
        indices_track = indices_track[indices_track != idx1]
        for idx2 in indices_track:
            iou_ = bbox.iou(boxes[idx1], boxes[idx2])
            if iou_ > iou_th:
                indices_keep = indices_keep[indices_keep != idx2]
                indices_track = indices_track[indices_track != idx2]

    return boxes[indices_keep]


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
