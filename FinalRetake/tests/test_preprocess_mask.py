import numpy as np
import cv2
from navigation.preprocess import path_mask_from_bgr


def _make_synthetic(w=400, h=250, thickness=30):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (20, 20, 20)
    pts = np.array([[30, 200], [120, 160], [240, 190], [360, 80]], dtype=np.int32)
    cv2.polylines(img, [pts], False, (235, 235, 235), thickness=thickness, lineType=cv2.LINE_AA)

    # ground-truth-ish mask: draw same polyline but as binary
    gt = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(gt, [pts], False, 255, thickness=thickness, lineType=cv2.LINE_AA)
    gt = (gt > 0).astype(np.uint8)
    return img, gt


def test_path_mask_on_synthetic_has_high_iou():
    bgr, gt01 = _make_synthetic()
    mask = path_mask_from_bgr(bgr, invert="auto")
    pred01 = (mask > 0).astype(np.uint8)

    inter = int(np.sum(pred01 & gt01))
    union = int(np.sum(pred01 | gt01))
    iou = inter / union if union else 1.0

    assert iou > 0.90


def test_path_mask_is_uint8_0_or_255():
    bgr, _ = _make_synthetic()
    mask = path_mask_from_bgr(bgr)
    assert mask.dtype == np.uint8
    vals = set(np.unique(mask).tolist())
    assert vals.issubset({0, 255})
