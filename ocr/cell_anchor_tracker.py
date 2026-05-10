from collections import Counter, defaultdict

import numpy as np
from norfair import Detection as NorfairDetection
from norfair import Tracker as NorfairTracker


def _calculate_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _distance(c1, c2):
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    return (dx * dx + dy * dy) ** 0.5


def _group_within_pool(ocr_detections, spatial_threshold):
    # Сырые рамки OCR в одну зону на кадре: одна точка на зону для Norfair.
    if not ocr_detections:
        return []

    for det in ocr_detections:
        det["center"] = _calculate_center(det["bbox"])

    used = [False] * len(ocr_detections)
    groups = []

    for i, det in enumerate(ocr_detections):
        if used[i]:
            continue

        group_detections = [det]
        used[i] = True
        group_center = det["center"]

        for j, other in enumerate(ocr_detections):
            if used[j]:
                continue

            if _distance(group_center, other["center"]) < spatial_threshold:
                group_detections.append(other)
                used[j] = True
                all_centers = [d["center"] for d in group_detections]
                group_center = (
                    sum(c[0] for c in all_centers) / len(all_centers),
                    sum(c[1] for c in all_centers) / len(all_centers),
                )

        texts = [d["text"] for d in group_detections]
        most_common_text = Counter(texts).most_common(1)[0][0]
        max_confidence = max(d["confidence"] for d in group_detections)

        groups.append(
            {
                "text": most_common_text,
                "confidence": max_confidence,
                "center": group_center,
                "raw_detections": group_detections,
            }
        )

    return groups


def group_detections_by_spatial_proximity(ocr_detections, spatial_threshold, partition_by_text=False):
    if not ocr_detections:
        return []
    if not partition_by_text:
        return _group_within_pool(ocr_detections, spatial_threshold)

    buckets = defaultdict(list)
    for d in ocr_detections:
        buckets[d["text"]].append(d)
    out = []
    for pool in buckets.values():
        out.extend(_group_within_pool(pool, spatial_threshold))
    return out


def _matched_this_frame(obj):
    return obj.last_detection is not None and obj.last_detection.age == obj.age


def _majority_text_for_object(obj):
    texts = []
    for d in obj.past_detections:
        if d.data is not None and isinstance(d.data, dict):
            t = d.data.get("text")
            if t is not None:
                texts.append(t)
    ld = obj.last_detection
    if ld is not None and ld.data is not None and isinstance(ld.data, dict):
        t = ld.data.get("text")
        if t is not None:
            texts.append(t)
    if not texts:
        return None
    return Counter(texts).most_common(1)[0][0]


def _estimate_center_tuple(obj):
    est = obj.estimate
    if est is None or len(est) == 0:
        return None
    return (float(est[0, 0]), float(est[0, 1]))


class BarcodeTracker:
    def __init__(
        self,
        history_size=7,
        spatial_threshold=150,
        association_gate=None,
        max_missed_frames=5,
        partition_by_text=False,
        norfair_initialization_delay=None,
    ):
        self.history_size = history_size
        self.spatial_threshold = spatial_threshold
        self.association_gate = association_gate if association_gate is not None else spatial_threshold * 1.2
        self.max_missed_frames = max_missed_frames
        self.partition_by_text = partition_by_text
        nid = norfair_initialization_delay
        self._norfair_init_delay = 0 if nid is None else nid

        hc = max(4, max_missed_frames + 2)
        init_d = max(0, min(hc - 1, self._norfair_init_delay))
        self._norfair = NorfairTracker(
            distance_function="euclidean",
            distance_threshold=self.association_gate,
            hit_counter_max=hc,
            initialization_delay=init_d,
            past_detections_length=max(1, history_size),
        )

    def update(self, ocr_detections):
        cells = group_detections_by_spatial_proximity(
            ocr_detections,
            self.spatial_threshold,
            partition_by_text=self.partition_by_text,
        )
        if cells:
            norfair_dets = []
            for cell in cells:
                cx, cy = cell["center"]
                norfair_dets.append(
                    NorfairDetection(
                        points=np.array([[float(cx), float(cy)]], dtype=np.float64),
                        scores=None,
                        data={
                            "text": cell["text"],
                            "confidence": cell["confidence"],
                            "center": cell["center"],
                        },
                    )
                )
            active_objects = self._norfair.update(norfair_dets)
        else:
            active_objects = self._norfair.update(None)

        instances = []
        for obj in active_objects:
            if obj.id is None or not _matched_this_frame(obj):
                continue
            data = obj.last_detection.data
            if not isinstance(data, dict):
                continue
            center = data.get("center")
            if center is None:
                center = _estimate_center_tuple(obj)
            instances.append(
                {
                    "track_id": obj.id,
                    "text": _majority_text_for_object(obj),
                    "text_frame": data.get("text"),
                    "center": center,
                    "confidence": data.get("confidence", 0.0),
                }
            )

        return {"instances": instances}


if __name__ == "__main__":
    tracker = BarcodeTracker(
        spatial_threshold=80,
        association_gate=120,
        max_missed_frames=4,
    )

    frames = [
        [
            {"text": "A12", "confidence": 0.9, "bbox": [10, 10, 30, 30]},
            {"text": "A12", "confidence": 0.85, "bbox": [200, 200, 222, 222]},
        ],
        [
            {"text": "A12", "confidence": 0.88, "bbox": [12, 11, 32, 31]},
            {"text": "A1Z", "confidence": 0.55, "bbox": [202, 198, 224, 220]},
        ],
        [
            {"text": "A12", "confidence": 0.91, "bbox": [13, 12, 33, 32]},
            {"text": "A12", "confidence": 0.72, "bbox": [204, 200, 226, 222]},
        ],
    ]

    for i, det in enumerate(frames):
        r = tracker.update(det)
        print(f"кадр {i}: всего экземпляров {len(r['instances'])}", r["instances"])
