"""Configuration management for the video surveillance pipeline."""

import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Zone:
    """Zone configuration for event detection."""
    name: str
    points: List[Tuple[float, float]]
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Zone':
        return cls(
            name=data['name'],
            points=[tuple(p) for p in data['points']]
        )
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'points': self.points
        }


@dataclass
class EventDetectionConfig:
    """Configuration for event detection."""
    loitering_threshold_frames: int = 30  # Frames to consider as loitering
    zone_intrusion_confidence: float = 0.5
    event_deduplication_frames: int = 10  # Frames gap to deduplicate events
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EventDetectionConfig':
        return cls(
            loitering_threshold_frames=data.get('loitering_threshold_frames', 30),
            zone_intrusion_confidence=data.get('zone_intrusion_confidence', 0.5),
            event_deduplication_frames=data.get('event_deduplication_frames', 10)
        )


@dataclass
class DetectionConfig:
    """Configuration for object detection."""
    model: str = 'yolov11n'  # Model size: n (nano), s (small), m (medium), l (large)
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    gpu: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DetectionConfig':
        return cls(
            model=data.get('model', 'yolov11n'),
            confidence_threshold=data.get('confidence_threshold', 0.5),
            iou_threshold=data.get('iou_threshold', 0.45),
            gpu=data.get('gpu', True)
        )


@dataclass
class TrackingConfig:
    """Configuration for object tracking."""
    max_age: int = 30  # Maximum frames to track without detection
    min_hits: int = 3  # Minimum detections to start a track
    iou_threshold: float = 0.3
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrackingConfig':
        return cls(
            max_age=data.get('max_age', 30),
            min_hits=data.get('min_hits', 3),
            iou_threshold=data.get('iou_threshold', 0.3)
        )


class ConfigManager:
    """Load and manage pipeline configuration."""
    
    def __init__(self, config_file: str):
        """Initialize config manager from JSON file."""
        self.config_file = Path(config_file)
        self.raw_config = self._load_config()
        
        # Parse configurations
        self.zones = [Zone.from_dict(z) for z in self.raw_config.get('zones', [])]
        self.detection = DetectionConfig.from_dict(
            self.raw_config.get('detection', {})
        )
        self.tracking = TrackingConfig.from_dict(
            self.raw_config.get('tracking', {})
        )
        self.events = EventDetectionConfig.from_dict(
            self.raw_config.get('events', {})
        )
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def create_default(output_path: str) -> 'ConfigManager':
        """Create a default configuration file."""
        default_config = {
            "zones": [
                {
                    "name": "Restricted Zone",
                    "points": [[100, 100], [300, 100], [300, 300], [100, 300]]
                },
                {
                    "name": "Entrance",
                    "points": [[500, 200], [700, 200], [700, 400], [500, 400]]
                }
            ],
            "detection": {
                "model": "yolov11n",
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45,
                "gpu": True
            },
            "tracking": {
                "max_age": 30,
                "min_hits": 3,
                "iou_threshold": 0.3
            },
            "events": {
                "loitering_threshold_frames": 30,
                "zone_intrusion_confidence": 0.5,
                "event_deduplication_frames": 10
            }
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return ConfigManager(str(output_path))
