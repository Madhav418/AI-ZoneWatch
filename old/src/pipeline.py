"""Main video surveillance pipeline."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
import logging
from time import time
from tqdm import tqdm

from .detector import PersonDetector
from .tracker import PersonTracker
from .events import EventDetector, Event, EventType
from .config import ConfigManager
from .utils import draw_box, draw_polygon, draw_zone_filled, draw_text, ensure_directory_exists

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoSurveillancePipeline:
    """Complete video surveillance pipeline."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the pipeline.
        
        Args:
            config: ConfigManager instance with all configurations
        """
        self.config = config
        self.detector = PersonDetector(
            model_name=config.detection.model,
            confidence_threshold=config.detection.confidence_threshold,
            iou_threshold=config.detection.iou_threshold,
            use_gpu=config.detection.gpu
        )
        self.tracker = PersonTracker(
            max_age=config.tracking.max_age,
            min_hits=config.tracking.min_hits,
            iou_threshold=config.tracking.iou_threshold
        )
        self.event_detector = EventDetector(
            zones=[z.to_dict() for z in config.zones],
            loitering_threshold_frames=config.events.loitering_threshold_frames,
            event_deduplication_frames=config.events.event_deduplication_frames
        )
        
        self.events: List[Event] = []
    
    def process_video(self, video_path: str, output_dir: str = 'results',
                     skip_frames: int = 0, max_frames: Optional[int] = None) -> Dict:
        """
        Process a video file end-to-end.
        
        Args:
            video_path: Path to input video
            output_dir: Output directory for results
            skip_frames: Number of frames to skip for processing
            max_frames: Maximum frames to process (None for all)
            
        Returns:
            Dictionary with processing results and summary
        """
        video_path = Path(video_path)
        output_dir = ensure_directory_exists(output_dir)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps == 0:
            fps = 30.0
        
        logger.info(f"Video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
        
        # Warmup detector
        self.detector.warmup(height, width)
        
        # Prepare output video
        output_video_path = output_dir / f"{video_path.stem}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        # Process frames
        frame_number = 0
        processed_frames = 0
        start_time = time()
        self.events = []
        self.event_detector.reset()
        
        # Calculate frame range
        frames_to_process = max_frames if max_frames else total_frames
        
        with tqdm(total=frames_to_process, desc="Processing") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and processed_frames >= max_frames:
                    break
                
                # Skip frames if specified
                if skip_frames > 0 and frame_number % (skip_frames + 1) != 0:
                    frame_number += 1
                    continue
                
                # Run detection
                detections = self.detector.detect(frame)
                
                # Update tracking
                tracks = self.tracker.update(detections)
                
                # Detect events
                events = self.event_detector.detect(frame_number, tracks, fps)
                self.events.extend(events)
                
                # Annotate frame
                annotated_frame = self._annotate_frame(frame, tracks, events)
                
                # Write output
                out.write(annotated_frame)
                
                frame_number += 1
                processed_frames += 1
                pbar.update(1)
        
        cap.release()
        out.release()
        
        elapsed_time = time() - start_time
        fps_processed = processed_frames / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(f"Processed {processed_frames} frames in {elapsed_time:.2f}s ({fps_processed:.2f} FPS)")
        
        # Save event log
        event_log_path = output_dir / f"{video_path.stem}_events.json"
        self._save_event_log(event_log_path)
        
        results = {
            'video_path': str(video_path),
            'output_video': str(output_video_path),
            'event_log': str(event_log_path),
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'width': width,
            'height': height,
            'fps': fps,
            'processing_fps': fps_processed,
            'elapsed_time': elapsed_time,
            'total_events': len(self.events),
            'event_summary': self._summarize_events()
        }
        
        logger.info(f"Results saved to {output_dir}")
        logger.info(f"Events detected: {len(self.events)}")
        
        return results
    
    def _annotate_frame(self, frame: np.ndarray, tracks: List[Dict], 
                       events: List[Event]) -> np.ndarray:
        """Annotate frame with detections, tracks, and zones."""
        annotated = frame.copy()
        
        # Draw zones
        for zone in self.config.zones:
            points = zone.points
            # Draw filled zone with transparency
            annotated = draw_zone_filled(annotated, points, color=(0, 255, 255), alpha=0.2)
            # Draw zone boundary
            annotated = draw_polygon(annotated, points, color=(0, 255, 255), thickness=2)
            # Draw zone label
            if len(points) > 0:
                annotated = draw_text(annotated, zone.name, 
                                     (int(points[0][0]), int(points[0][1])-10),
                                     color=(0, 255, 255))
        
        # Draw tracks
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            confidence = track.get('confidence', 0.0)
            
            # Draw bbox
            annotated = draw_box(annotated, bbox, color=(0, 255, 0), thickness=2)
            
            # Draw track ID and confidence
            x1, y1 = int(bbox[0]), int(bbox[1])
            label = f"ID:{track_id} ({confidence:.2f})"
            annotated = draw_text(annotated, label, (x1, y1-5), color=(0, 255, 0))
        
        # Draw events
        for event in events:
            x1, y1, x2, y2 = map(int, event.bbox)
            
            # Highlight event box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Draw event label
            event_label = f"{event.event_type.value.upper()}: {event.zone_name}"
            annotated = draw_text(annotated, event_label, (x1, y1-25), 
                                 color=(0, 0, 255), font_scale=0.7)
        
        return annotated
    
    def _save_event_log(self, output_path: Path):
        """Save event log to JSON."""
        event_data = {
            'events': [e.to_dict() for e in self.events],
            'total_events': len(self.events),
            'event_summary': self._summarize_events()
        }
        
        with open(output_path, 'w') as f:
            json.dump(event_data, f, indent=2)
        
        logger.info(f"Event log saved to {output_path}")
    
    def _summarize_events(self) -> Dict:
        """Summarize events by type and zone."""
        summary = {}
        for event in self.events:
            key = f"{event.event_type.value}_{event.zone_name}"
            if key not in summary:
                summary[key] = 0
            summary[key] += 1
        return summary
