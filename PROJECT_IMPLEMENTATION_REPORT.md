# AI-ZoneWatch Project Implementation Report

## 1. Project Summary

This project implements a practical video surveillance pipeline for:
- Person detection
- Multi-object tracking with persistent IDs
- Zone-based event detection (intrusion and loitering)
- Annotated video generation
- Structured event logging (CSV and JSON)

The solution is built as a modular prototype suitable for take-home AI engineering evaluation and future production extension.

---

## 2. Problem Objective and Target Outcomes

The implementation was designed to satisfy the core assignment goals:
- Detect and track people with unique IDs across frames
- Handle temporary disappearance and re-entry with identity continuity
- Read configurable polygon zones from JSON
- Detect zone intrusion and loitering events
- Generate timestamped logs with frame number, bbox, event type, confidence
- Produce annotated output video
- Provide a simple command-line interface for end-to-end execution

---

## 3. Development Approach Used

### 3.1 Approach Strategy

The project followed a pipeline-first, modular design approach:
1. Build baseline end-to-end flow first
2. Separate concerns by module (detector, tracker, zones, events, rendering)
3. Add production-minded robustness (deduplication, error handling, output checks)
4. Tighten event semantics (stationary loitering logic)
5. Add documentation and sample outputs

This allowed continuous validation while keeping components independently maintainable.

### 3.2 Architecture Approach

The pipeline is organized as:
- Input video read frame by frame
- Detection model returns person bounding boxes
- Tracker assigns and maintains IDs over time
- Zone manager resolves track-in-zone relationships
- Event manager performs temporal reasoning and event generation
- Renderer overlays zones, tracks, FPS, and events
- Writer saves annotated video and event logs

---

## 4. Modules and Implementation Details

### 4.1 run.py (Pipeline Orchestration)

Responsibilities:
- Parse CLI arguments
- Load detector, tracker, zone manager, event manager
- Open input video and output writer
- Execute inference-tracking-event loop
- Draw overlays and save outputs
- Release resources cleanly

Key implementation choices:
- Output folder auto-creation before processing
- Fixed processing/output size for consistency
- FPS fallback if source metadata is invalid
- Writer-open validation to avoid silent video save failures

### 4.2 core/detector.py (Person Detection)

Approach used:
- YOLOv8-based detector
- Person class filtering (COCO class id 0)
- Confidence threshold parameterization

Why this approach:
- Good speed-accuracy trade-off for surveillance scenarios
- Easy model swap via CLI argument

### 4.3 core/tracker.py (Tracking + Re-identification)

Base tracker approach:
- DeepSORT with tuned parameters for occlusion tolerance

Identity continuity approach:
- Stable ID layer maintained on top of raw tracker IDs
- Re-association of new raw tracks to existing stable identities
- Combined scoring based on:
  - Geometry overlap (IoU)
  - Center distance
  - Appearance similarity (HSV histogram)
  - Time gap penalty

Why this approach:
- Raw tracking IDs can switch under occlusion/re-entry
- Stable ID layer improves identity consistency in real CCTV conditions

### 4.4 core/zone_manager.py (Zone Logic)

Approach used:
- Load zone definitions from JSON
- Build polygons using Shapely
- Determine zone occupancy via point-in-polygon on track center

Why this approach:
- Accurate and robust polygon math
- Flexible config-driven zone definitions

### 4.5 core/event_manager.py (Temporal Event Reasoning)

Implemented event types:
- Intrusion
- Loitering

Approach for intrusion:
- Zone-based trigger
- Frame-level deduplication cooldown to avoid event spam

Approach for loitering:
- Time-in-zone + stationary-motion requirement
- Per-zone movement state per track:
  - Anchor center
  - Last center
  - Stationary start time
- Stationary timer resets if movement exceeds threshold
- Threshold configurable using stationary_threshold_px (default 25)

Logging approach:
- CSV output for tabular auditing
- JSON output for downstream system integration
- Includes timestamp, frame, track id, event type, zone, bbox, confidence

### 4.6 core/utils.py (Visualization Utilities)

Approach used:
- Dedicated drawing helpers for readability and reuse
- Zone overlays with labels
- Track boxes with IDs and confidence
- Event text overlays
- FPS and frame counter overlays

### 4.7 core/video_processor.py (Reusable Video Utility)

Approach used:
- Added reusable video processor abstraction
- Supports frame iteration, metadata access, FPS tracking, output writing

---

## 5. Configuration Approach

Config source:
- config/zones.json

Config-driven behavior:
- Zone name
- Event type
- Polygon coordinates
- Loiter time
- Stationary movement threshold (stationary_threshold_px)

This design enables adapting to new camera scenes without code changes.

---

## 6. CLI and Usability

Execution is done from project root.

Basic command:

python run.py --video <video_path> --zones config/zones.json --output <output_folder>

With optional model and confidence:

python run.py --video <video_path> --zones config/zones.json --output <output_folder> --model yolov8n.pt --conf 0.5

CLI options available:
- --video (required)
- --zones (required)
- --output (required)
- --model (optional)
- --conf (optional)

---

## 7. Output Artifacts Produced

Primary outputs:
- Annotated video
- events.csv
- events.json

Sample generated artifacts are available in output folder, including:
- output/video1_annotatedOutput.mp4
- output/video1_events.csv
- output/video1_events.json
- output/video2_annotatedOutput.mp4
- output/video3_annotatedOutput.mp4
- output/sample_result_video1.jpg
- output/sample_result_video2.jpg
- output/sample_result_video3.jpg

---

## 8. Reliability and Edge-Case Handling

Implemented safeguards:
- End-of-video frame read handling
- Video writer initialization check
- Invalid FPS fallback
- Event deduplication for repeated intrusion alerts
- Track state reset on exit/loss
- Confidence propagation from tracker outputs

Edge cases addressed:
- Temporary occlusions
- Re-entry after leaving frame
- Repeated event flooding
- Incorrect output video writing dimensions

---

## 9. Trade-offs and Design Decisions

Key trade-offs:
- Chosen lightweight model defaults for practical speed
- Center-point zone checks are fast but can miss partial-person edge cases
- Appearance histogram ReID is simple and efficient, but not as strong as dedicated ReID embeddings
- Wall-clock loiter timing is pragmatic but can differ from strict video-time under heavy compute load

---

## 10. Known Limitations

Current limitations:
- Dense crowds can still cause occasional identity switches
- Heavy lighting changes can degrade appearance-based re-association
- Stationary threshold may need camera-specific tuning
- Zone coordinates are pixel-space and should be recalibrated when camera resolution/perspective changes

---

## 11. What Was Improved Iteratively During Development

Major iterative improvements made:
- Fixed frame-read and resize ordering issue
- Fixed output video save mismatch by aligning writer size with processed frame size
- Added output directory creation and writer validation
- Added event deduplication and richer event payload
- Added JSON event output alongside CSV
- Improved tracker confidence handling
- Added stationary-motion loitering semantics
- Added sample result images into README

---

## 12. Final Assessment

The project now provides a complete, structured, and assignment-aligned prototype with:
- End-to-end pipeline execution
- Strong modularity and readability
- Practical production-minded handling
- Configurable behavior for surveillance scenarios

It is ready for demonstration, evaluation submission, and further extension.
