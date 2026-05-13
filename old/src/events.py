"""Zone-based event detection (intrusion and loitering)."""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import logging

from .utils import point_in_polygon, box_center

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events that can be detected."""
    ZONE_INTRUSION = "zone_intrusion"
    LOITERING = "loitering"


@dataclass
class Event:
    """Detected event."""
    event_type: EventType
    zone_name: str
    track_id: int
    frame_number: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'event_type': self.event_type.value,
            'zone_name': self.zone_name,
            'track_id': self.track_id,
            'frame_number': self.frame_number,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'timestamp': self.timestamp
        }


class EventDetector:
    """Detect zone-based events."""
    
    def __init__(self, zones: List[Dict], loitering_threshold_frames: int = 30,
                 event_deduplication_frames: int = 10):
        """
        Initialize event detector.
        
        Args:
            zones: List of zone definitions with 'name' and 'points' keys
            loitering_threshold_frames: Frames to consider as loitering
            event_deduplication_frames: Frames to deduplicate repeated events
        """
        self.zones = zones
        self.loitering_threshold_frames = loitering_threshold_frames
        self.event_deduplication_frames = event_deduplication_frames
        
        # Track state for each person in each zone
        self.track_zone_state: Dict[Tuple[int, str], Dict] = defaultdict(dict)
        # Store last event for deduplication
        self.last_events: Dict[Tuple[int, str, EventType], int] = {}
    
    def detect(self, frame_number: int, tracks: List[Dict], 
               fps: float = 30.0) -> List[Event]:
        """
        Detect events based on current tracks.
        
        Args:
            frame_number: Current frame number
            tracks: List of tracked persons with bbox and track_id
            fps: Frames per second for timestamp calculation
            
        Returns:
            List of detected events
        """
        events = []
        timestamp = frame_number / fps
        
        # Update track positions in zones
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            
            for zone in self.zones:
                zone_name = zone['name']
                points = zone['points']
                
                # Check if center is in zone
                center = box_center(bbox)
                in_zone = point_in_polygon(center, points)
                
                key = (track_id, zone_name)
                
                if in_zone:
                    # Update state
                    if key not in self.track_zone_state:
                        self.track_zone_state[key] = {
                            'entry_frame': frame_number,
                            'last_frame': frame_number,
                            'frames_in_zone': 0
                        }
                    
                    self.track_zone_state[key]['last_frame'] = frame_number
                    self.track_zone_state[key]['frames_in_zone'] += 1
                    
                    # Check for intrusion (just entered)
                    if self.track_zone_state[key]['frames_in_zone'] == 1:
                        if self._should_emit_event(track_id, zone_name, EventType.ZONE_INTRUSION, frame_number):
                            events.append(Event(
                                event_type=EventType.ZONE_INTRUSION,
                                zone_name=zone_name,
                                track_id=track_id,
                                frame_number=frame_number,
                                bbox=bbox,
                                confidence=track.get('confidence', 0.0),
                                timestamp=timestamp
                            ))
                            self.last_events[(track_id, zone_name, EventType.ZONE_INTRUSION)] = frame_number
                    
                    # Check for loitering
                    frames_in_zone = self.track_zone_state[key]['frames_in_zone']
                    if frames_in_zone > 0 and frames_in_zone % self.loitering_threshold_frames == 0:
                        if self._should_emit_event(track_id, zone_name, EventType.LOITERING, frame_number):
                            events.append(Event(
                                event_type=EventType.LOITERING,
                                zone_name=zone_name,
                                track_id=track_id,
                                frame_number=frame_number,
                                bbox=bbox,
                                confidence=track.get('confidence', 0.0),
                                timestamp=timestamp
                            ))
                            self.last_events[(track_id, zone_name, EventType.LOITERING)] = frame_number
                else:
                    # Person left zone
                    if key in self.track_zone_state:
                        state = self.track_zone_state[key]
                        frames_in_zone = state.get('frames_in_zone', 0)
                        
                        # Log if person was in zone for significant time
                        if frames_in_zone > 5:
                            logger.debug(f"Track {track_id} left {zone_name} after {frames_in_zone} frames")
                        
                        del self.track_zone_state[key]
        
        # Cleanup old state entries (track might have disappeared)
        keys_to_remove = []
        for key in self.track_zone_state:
            track_id = key[0]
            if not any(t['track_id'] == track_id for t in tracks):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.track_zone_state[key]
        
        return events
    
    def _should_emit_event(self, track_id: int, zone_name: str, event_type: EventType, 
                          frame_number: int) -> bool:
        """Check if event should be emitted (deduplication)."""
        key = (track_id, zone_name, event_type)
        
        if key not in self.last_events:
            return True
        
        last_frame = self.last_events[key]
        return (frame_number - last_frame) > self.event_deduplication_frames
    
    def reset(self):
        """Reset detector state (e.g., between videos)."""
        self.track_zone_state.clear()
        self.last_events.clear()
