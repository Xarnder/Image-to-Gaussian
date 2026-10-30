"""Face geometry from a single image, with models ComfyUI already has (InsightFace buffalo_l, Apache/MIT code,
research-use weights): SCRFD face box + 1k3d68 (68 landmarks with a relative depth per landmark).

The 1k3d68 depth is the network's estimate of the face's own relief (how far the nose stands in front of the
cheeks and jaw), learned from 3D face fits. It knows nothing about our body model or our generated views,
so it is an independent reference for "is the face too far forward or back relative to the sides of the head".

68-point layout (iBUG): 0-16 jaw contour (0 = image-left ear level), 17-26 brows, 27-30 nose bridge (30 = tip),
31-35 nostrils, 36-47 eyes, 48-67 mouth.
"""
from __future__ import annotations

import cv2
import numpy as np

from i2g.evaluate import INSIGHTFACE_DIR

NOSE_TIP, NOSE_ROOT = 30, 27
JAW_UPPER = (1, 15)          # jaw contour at about the tragus height: the sides of the face
EYES = (36, 39, 42, 45)


class FaceGeo:
    def __init__(self):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        cpu = ["CPUExecutionProvider"]
        self.det = ort.InferenceSession(str(INSIGHTFACE_DIR / "det_10g.onnx"), opts, providers=cpu)
        self.lmk = ort.InferenceSession(str(INSIGHTFACE_DIR / "1k3d68.onnx"), opts, providers=cpu)

    def box(self, rgb: np.ndarray, thresh: float = 0.4) -> np.ndarray | None:
        """Best face box (x0, y0, x1, y1) in image pixels, SCRFD as in insightface."""
        size = 640
        h, w = rgb.shape[:2]
        scale = size / max(h, w)
        img = np.zeros((size, size, 3), np.float32)
        img[:round(h * scale), :round(w * scale)] = cv2.resize(rgb, (round(w * scale), round(h * scale)))
        blob = ((img - 127.5) / 128.0).transpose(2, 0, 1)[None].astype(np.float32)
        outs = self.det.run(None, {self.det.get_inputs()[0].name: blob})
        best, best_s = None, thresh
        for idx, stride in enumerate((8, 16, 32)):
            scores = outs[idx].reshape(-1)
            d = outs[idx + 3].reshape(-1, 4) * stride
            g = size // stride
            centers = np.repeat(np.stack(np.mgrid[:g, :g][::-1], -1).reshape(-1, 2).astype(np.float32) * stride, 2, 0)
            j = int(scores.argmax())
            if scores[j] > best_s:
                best_s = scores[j]
                best = np.array([centers[j, 0] - d[j, 0], centers[j, 1] - d[j, 1],
                                 centers[j, 0] + d[j, 2], centers[j, 1] + d[j, 3]]) / scale
        return best

    def landmarks(self, rgb: np.ndarray, box: np.ndarray | None = None) -> np.ndarray | None:
        """(68, 3): x, y in image pixels, z in the same pixel units, larger z = nearer the camera."""
        box = self.box(rgb) if box is None else box
        if box is None:
            return None
        center = (box[:2] + box[2:]) / 2
        s = 192.0 / (max(box[2] - box[0], box[3] - box[1]) * 1.5)
        M = np.array([[s, 0, 96 - s * center[0]], [0, s, 96 - s * center[1]]], np.float64)
        crop = cv2.warpAffine(rgb, M, (192, 192), borderValue=0)
        blob = crop.astype(np.float32).transpose(2, 0, 1)[None]          # this model normalises internally
        pred = self.lmk.run(None, {self.lmk.get_inputs()[0].name: blob})[0][0].reshape(-1, 3)[-68:].copy()
        pred[:, :2] = (pred[:, :2] + 1) * 96
        pred[:, 2] *= 96
        pred[:, :2] = (pred[:, :2] - M[:, 2]) / s                        # crop -> image
        pred[:, 2] /= s
        # make "nearer = larger": the nose tip is the nearest point of a face seen from the front
        if pred[NOSE_TIP, 2] < pred[list(JAW_UPPER), 2].mean():
            pred[:, 2] = -pred[:, 2]
        return pred


def relief_px(lm: np.ndarray) -> float:
    """How far the nose tip stands in front of the sides of the face (jaw contour at the tragus), in pixels."""
    return float(lm[NOSE_TIP, 2] - lm[list(JAW_UPPER), 2].mean())
