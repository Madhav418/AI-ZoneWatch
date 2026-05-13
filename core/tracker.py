import cv2
import numpy as np

from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracker:

    def __init__(self):

        self.tracker = DeepSort(

            # Keep tracks alive longer through short/medium occlusions.
            max_age=90,

            n_init=3,

            max_cosine_distance=0.35,

            nn_budget=100,

            embedder="mobilenet",

            half=True,

            bgr=True
        )

        # Persistent ID layer to reduce ID switches after occlusion.
        self.frame_index = 0
        self.next_stable_id = 1
        self.raw_to_stable = {}
        self.stable_memory = {}

        # Re-association parameters.
        self.reid_max_gap_frames = 300
        self.reid_iou_threshold = 0.15
        self.reid_center_dist_threshold = 140.0
        self.reid_hist_min_similarity = 0.55

    @staticmethod
    def _clip_bbox(bbox, frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)
        return [x1, y1, x2, y2]

    def _compute_histogram(self, frame, bbox):
        """Compute a compact normalized HSV histogram for person appearance matching."""
        x1, y1, x2, y2 = self._clip_bbox(bbox, frame)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist

    @staticmethod
    def _hist_similarity(hist_a, hist_b):
        if hist_a is None or hist_b is None:
            return 0.0
        # Correlation in [-1, 1]; map to [0, 1].
        score = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
        return max(0.0, min(1.0, (score + 1.0) / 2.0))

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area

        if union <= 0:
            return 0.0

        return inter_area / union

    @staticmethod
    def _center_distance(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        acx = (ax1 + ax2) / 2.0
        acy = (ay1 + ay2) / 2.0
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0

        dx = acx - bcx
        dy = acy - bcy
        return (dx * dx + dy * dy) ** 0.5

    def _allocate_stable_id(self):
        stable_id = self.next_stable_id
        self.next_stable_id += 1
        return stable_id

    def _find_reassociation(self, bbox, appearance_hist, used_stable_ids):
        best_id = None
        best_score = -10.0

        for stable_id, state in self.stable_memory.items():
            if stable_id in used_stable_ids:
                continue

            frame_gap = self.frame_index - state['last_seen']
            if frame_gap > self.reid_max_gap_frames:
                continue

            prev_bbox = state['bbox']
            prev_hist = state.get('hist')
            iou = self._iou(bbox, prev_bbox)
            center_dist = self._center_distance(bbox, prev_bbox)
            hist_sim = self._hist_similarity(appearance_hist, prev_hist)

            # Strong appearance mismatch: reject unless geometry is near-perfect.
            if hist_sim < 0.35 and iou < 0.40:
                continue

            if (
                iou < self.reid_iou_threshold
                and center_dist > self.reid_center_dist_threshold
                and hist_sim < self.reid_hist_min_similarity
            ):
                continue

            gap_penalty = min(1.0, frame_gap / float(self.reid_max_gap_frames))
            distance_score = max(0.0, 1.0 - (center_dist / (self.reid_center_dist_threshold * 2.0)))

            # Weighted score: appearance dominates for longer gaps.
            if frame_gap > 90:
                score = (0.60 * hist_sim) + (0.25 * iou) + (0.20 * distance_score) - (0.15 * gap_penalty)
            else:
                score = (0.40 * hist_sim) + (0.35 * iou) + (0.30 * distance_score) - (0.10 * gap_penalty)

            if score > best_score:
                best_score = score
                best_id = stable_id

        return best_id

    def update(self, detections, frame):

        self.frame_index += 1

        detection_list = []

        for det in detections:

            x1, y1, x2, y2, conf = det

            w = x2 - x1
            h = y2 - y1

            detection_list.append(
                (
                    [x1, y1, w, h],
                    conf,
                    'person'
                )
            )

        tracks = self.tracker.update_tracks(
            detection_list,
            frame=frame
        )

        active_raw_ids = set()
        used_stable_ids = set()
        results = []

        for track in tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id
            active_raw_ids.add(track_id)

            ltrb = track.to_ltrb()

            x1, y1, x2, y2 = map(int, ltrb)
            bbox = self._clip_bbox([x1, y1, x2, y2], frame)
            appearance_hist = self._compute_histogram(frame, bbox)

            # det_conf is available when the track was just matched to a detection
            det_conf = track.det_conf if track.det_conf is not None else 1.0

            if track_id in self.raw_to_stable:
                stable_id = self.raw_to_stable[track_id]
            else:
                stable_id = self._find_reassociation(bbox, appearance_hist, used_stable_ids)
                if stable_id is None:
                    stable_id = self._allocate_stable_id()
                self.raw_to_stable[track_id] = stable_id

            used_stable_ids.add(stable_id)

            prev_state = self.stable_memory.get(stable_id)
            if prev_state is not None and prev_state.get('hist') is not None and appearance_hist is not None:
                # EMA smooths appearance changes while preserving identity signature.
                appearance_hist = (0.80 * prev_state['hist']) + (0.20 * appearance_hist)

            self.stable_memory[stable_id] = {
                'bbox': bbox,
                'last_seen': self.frame_index,
                'hist': appearance_hist,
            }

            results.append({

                'id': int(stable_id),

                'bbox': bbox,

                'confidence': round(float(det_conf), 4)
            })

        # Prune very old stable tracks from memory.
        stale_stable_ids = []
        for stable_id, state in self.stable_memory.items():
            if self.frame_index - state['last_seen'] > self.reid_max_gap_frames * 2:
                stale_stable_ids.append(stable_id)

        for stable_id in stale_stable_ids:
            del self.stable_memory[stable_id]

        # Drop mappings for raw IDs that disappeared to keep map compact.
        stale_raw_ids = [rid for rid in self.raw_to_stable.keys() if rid not in active_raw_ids]
        for rid in stale_raw_ids:
            del self.raw_to_stable[rid]

        return results