"""Person detection using YOLOv11."""

import torch
import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)


class PersonDetector:
    """Detect persons in video frames using YOLOv11."""
    
    PERSON_CLASS_ID = 0  # COCO class ID for person
    
    def __init__(self, model_name: str = 'yolov11n', confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45, use_gpu: bool = True):
        """
        Initialize the detector.
        
        Args:
            model_name: YOLOv11 model size (n/s/m/l/x)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: NMS IoU threshold
            use_gpu: Use GPU if available
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Set device
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Load model
        try:
            self.model = YOLO(f'{model_name}.pt')
            self.model.to(self.device)
            logger.info(f"Loaded YOLOv11 model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """
        Detect persons in a frame.
        
        Args:
            frame: Input frame (H, W, 3) in BGR format
            
        Returns:
            List of detections as (x1, y1, x2, y2, confidence)
        """
        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.PERSON_CLASS_ID,  # Only person class
            verbose=False
        )
        
        detections = []
        
        # Extract detections
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    
                    detections.append((float(x1), float(y1), float(x2), float(y2), conf))
        
        return detections
    
    def batch_detect(self, frames: List[np.ndarray]) -> List[List[Tuple[float, float, float, float, float]]]:
        """
        Detect persons in multiple frames (batch processing).
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection lists, one per frame
        """
        batch_detections = []
        
        for frame in frames:
            detections = self.detect(frame)
            batch_detections.append(detections)
        
        return batch_detections
    
    def warmup(self, frame_height: int = 480, frame_width: int = 640):
        """Warmup the model with a dummy frame."""
        try:
            dummy_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            self.detect(dummy_frame)
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
