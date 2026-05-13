"""Person tracking using Kalman Filter based approach."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
import logging

logger = logging.getLogger(__name__)


@dataclass
class KalmanState:
    """Kalman filter state for tracking."""
    # State: [x, y, w, h, vx, vy, vw, vh]
    x: np.ndarray = field(default_factory=lambda: np.zeros(8))  # State vector
    P: np.ndarray = field(default_factory=lambda: np.eye(8))    # Covariance matrix
    
    def copy(self) -> 'KalmanState':
        return KalmanState(x=self.x.copy(), P=self.P.copy())


class KalmanFilter:
    """Simple Kalman Filter for 2D bounding box tracking."""
    
    def __init__(self):
        # Transition matrix (F)
        dt = 1.0
        self.F = np.eye(8, 8)
        self.F[0, 4] = dt
        self.F[1, 5] = dt
        self.F[2, 6] = dt
        self.F[3, 7] = dt
        
        # Measurement matrix (H)
        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1
        
        # Measurement noise (R)
        self.R = np.eye(4) * 10
        
        # Process noise (Q)
        self.Q = np.eye(8) * 0.01
        self.Q[4:, 4:] *= 0.1
    
    def predict(self, state: KalmanState) -> KalmanState:
        """Predict next state."""
        new_state = state.copy()
        new_state.x = self.F @ state.x
        new_state.P = self.F @ state.P @ self.F.T + self.Q
        return new_state
    
    def update(self, state: KalmanState, measurement: np.ndarray) -> KalmanState:
        """Update state with measurement (x1, y1, x2, y2)."""
        new_state = state.copy()
        
        # Convert bbox to center + size
        meas = np.array([
            (measurement[0] + measurement[2]) / 2,  # cx
            (measurement[1] + measurement[3]) / 2,  # cy
            measurement[2] - measurement[0],        # w
            measurement[3] - measurement[1]         # h
        ])
        
        # Innovation
        y = meas - (self.H @ new_state.x)
        
        # Innovation covariance
        S = self.H @ new_state.P @ self.H.T + self.R
        
        # Kalman gain
        K = new_state.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        new_state.x = new_state.x + K @ y
        new_state.P = (np.eye(8) - K @ self.H) @ new_state.P
        
        return new_state


@dataclass
class Track:
    """A single track for a detected person."""
    track_id: int
    state: KalmanState
    last_detection: Tuple[float, float, float, float]
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    
    def get_bbox(self) -> Tuple[float, float, float, float]:
        """Get current bounding box from state."""
        x, y, w, h = self.state.x[:4]
        return (x - w/2, y - h/2, x + w/2, y + h/2)


class PersonTracker:
    """Track detected persons across frames."""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize tracker.
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum hits to start outputting track
            iou_threshold: IoU threshold for matching detections to tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: List[Track] = []
        self.next_id = 1
        self.kf = KalmanFilter()
    
    def update(self, detections: List[Tuple[float, float, float, float, float]]) -> List[Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of (x1, y1, x2, y2, confidence)
            
        Returns:
            List of active tracks with metadata
        """
        # Predict all tracks
        for track in self.tracks:
            track.state = self.kf.predict(track.state)
            track.time_since_update += 1
        
        # Match detections to tracks
        matched_pairs, unmatched_detections, unmatched_tracks = self._match_detections(detections)
        
        # Update matched tracks
        for track_idx, det_idx in matched_pairs:
            det = detections[det_idx]
            self.tracks[track_idx].state = self.kf.update(self.tracks[track_idx].state, 
                                                          np.array(det[:4]))
            self.tracks[track_idx].last_detection = det[:4]
            self.tracks[track_idx].hits += 1
            self.tracks[track_idx].time_since_update = 0
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            self._create_track(det)
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Get output tracks
        output_tracks = []
        for track in self.tracks:
            if track.hits >= self.min_hits or track.time_since_update == 0:
                bbox = track.get_bbox()
                output_tracks.append({
                    'track_id': track.track_id,
                    'bbox': bbox,
                    'confidence': track.last_detection[4] if len(track.last_detection) > 4 else 0.0,
                    'age': track.age,
                    'hits': track.hits
                })
        
        return output_tracks
    
    def _match_detections(self, detections: List[Tuple[float, float, float, float, float]]) -> \
            Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to existing tracks using IoU."""
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, track in enumerate(self.tracks):
            track_bbox = track.get_bbox()
            for j, det in enumerate(detections):
                det_bbox = det[:4]
                iou_matrix[i, j] = self._iou(track_bbox, det_bbox)
        
        # Hungarian algorithm
        track_indices, det_indices = linear_sum_assignment(-iou_matrix)
        
        matched_pairs = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        for track_idx, det_idx in zip(track_indices, det_indices):
            if iou_matrix[track_idx, det_idx] > self.iou_threshold:
                matched_pairs.append((track_idx, det_idx))
                unmatched_detections.remove(det_idx)
                unmatched_tracks.remove(track_idx)
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def _create_track(self, detection: Tuple[float, float, float, float, float]):
        """Create a new track from detection."""
        x1, y1, x2, y2, conf = detection
        
        # Initialize state
        state = KalmanState()
        state.x[0] = (x1 + x2) / 2  # cx
        state.x[1] = (y1 + y2) / 2  # cy
        state.x[2] = x2 - x1         # w
        state.x[3] = y2 - y1         # h
        
        track = Track(
            track_id=self.next_id,
            state=state,
            last_detection=(x1, y1, x2, y2, conf)
        )
        
        self.tracks.append(track)
        self.next_id += 1
    
    @staticmethod
    def _iou(box1: Tuple[float, float, float, float], 
             box2: Tuple[float, float, float, float]) -> float:
        """Calculate IoU between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
