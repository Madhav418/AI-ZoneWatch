#!/usr/bin/env python3
"""CLI entry point for the video surveillance pipeline."""

import argparse
import sys
import json
from pathlib import Path

from src.config import ConfigManager
from src.pipeline import VideoSurveillancePipeline


def main():
    parser = argparse.ArgumentParser(
        description='Video Surveillance: Detection, Tracking & Event Recognition'
    )
    
    parser.add_argument(
        '--video', type=str, required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '--zones', type=str, default='config/zones.json',
        help='Path to zones configuration file (default: config/zones.json)'
    )
    parser.add_argument(
        '--output', type=str, default='results',
        help='Output directory for results (default: results)'
    )
    parser.add_argument(
        '--skip-frames', type=int, default=0,
        help='Skip N frames between processing (0=process all). Higher values speed up processing.'
    )
    parser.add_argument(
        '--max-frames', type=int, default=None,
        help='Maximum number of frames to process (None=all frames)'
    )
    parser.add_argument(
        '--create-sample-config', action='store_true',
        help='Create a sample configuration file and exit'
    )
    parser.add_argument(
        '--model', type=str, default='yolov11n',
        choices=['yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x'],
        help='YOLOv11 model size (default: yolov11n for speed)'
    )
    parser.add_argument(
        '--confidence', type=float, default=0.5,
        help='Detection confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '--cpu', action='store_true',
        help='Force CPU processing (otherwise uses GPU if available)'
    )
    
    args = parser.parse_args()
    
    # Handle sample config creation
    if args.create_sample_config:
        config_path = args.zones
        config = ConfigManager.create_default(config_path)
        print(f"✓ Sample configuration created at: {config_path}")
        return 0
    
    # Validate inputs
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"✗ Video file not found: {args.video}", file=sys.stderr)
        return 1
    
    zones_path = Path(args.zones)
    if not zones_path.exists():
        print(f"✗ Zones configuration file not found: {args.zones}", file=sys.stderr)
        print(f"  Create one with: python run.py --video dummy.mp4 --create-sample-config", 
              file=sys.stderr)
        return 1
    
    try:
        # Load configuration
        config = ConfigManager(str(zones_path))
        
        # Override model settings if specified
        config.detection.model = args.model
        config.detection.confidence_threshold = args.confidence
        config.detection.gpu = not args.cpu
        
        print(f"Configuration loaded from: {zones_path}")
        print(f"  Model: {config.detection.model}")
        print(f"  Confidence threshold: {config.detection.confidence_threshold}")
        print(f"  GPU: {config.detection.gpu}")
        print(f"  Zones: {len(config.zones)}")
        print()
        
        # Create and run pipeline
        pipeline = VideoSurveillancePipeline(config)
        
        results = pipeline.process_video(
            video_path=str(video_path),
            output_dir=args.output,
            skip_frames=args.skip_frames,
            max_frames=args.max_frames
        )
        
        # Print results
        print()
        print("=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Video: {Path(results['video_path']).name}")
        print(f"Frames processed: {results['processed_frames']}/{results['total_frames']}")
        print(f"Processing FPS: {results['processing_fps']:.2f}")
        print(f"Elapsed time: {results['elapsed_time']:.2f}s")
        print()
        print(f"Output video: {results['output_video']}")
        print(f"Event log: {results['event_log']}")
        print()
        print("Event Summary:")
        for key, count in results['event_summary'].items():
            print(f"  {key}: {count}")
        print()
        
        return 0
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
