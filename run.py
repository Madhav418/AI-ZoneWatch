import cv2
import argparse
import os
import time

from core.detector import PersonDetector
from core.tracker import Tracker
from core.zone_manager import ZoneManager
from core.event_manager import EventManager
from core.utils import draw_zones, draw_tracks, draw_fps, draw_events


def main(video_path, zone_file, output_dir, model_path='yolov8n.pt', conf_threshold=0.5):

    print("===================================")
    print("VIDEO SURVEILLANCE SYSTEM STARTED")
    print("===================================")

    os.makedirs(output_dir, exist_ok=True)

    # detector
    print("[INFO] Loading YOLO model...")
    detector = PersonDetector(model_path=model_path, conf_threshold=conf_threshold)

    # tracker
    print("[INFO] Initializing ByteTrack...")
    tracker = Tracker()

    # zones
    print("[INFO] Loading zones...")
    zone_manager = ZoneManager(zone_file)

    # event manager
    print("[INFO] Initializing event manager...")
    event_manager = EventManager(
        csv_path=f'{output_dir}/events.csv',
        json_path=f'{output_dir}/events.json'
    )

    # video
    print("[INFO] Opening video...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print("[ERROR] Unable to open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Width: {width}")
    print(f"[INFO] Height: {height}")
    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Total Frames: {total_frames}")

    # Process and save at a fixed size for consistent inference and output.
    output_size = (960, 540)

    # output writer
    writer = cv2.VideoWriter(
        f'{output_dir}/annotated.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        output_size
    )

    if not writer.isOpened():
        print("[ERROR] Unable to create output video writer.")
        cap.release()
        event_manager.close()
        return

    frame_num = 0

    prev_time = time.time()

    print("\n[INFO] Starting processing...\n")

    while True:

        ret, frame = cap.read()

        if not ret:

            print("\n[INFO] Video processing completed.")
            break

        frame = cv2.resize(frame, output_size)

        # detect persons
        detections = detector.detect(frame)

        # tracking
        tracks = tracker.update(detections, frame)

        # zone matching
        matched_zones = {}

        for track in tracks:

            zones = zone_manager.check_zones(track)

            matched_zones[track['id']] = zones

        # event processing
        events = event_manager.process(
            frame_num,
            tracks,
            matched_zones
        )

        # draw zones, tracks, events using utils helpers
        frame = draw_zones(frame, zone_manager.zones)
        frame = draw_tracks(frame, tracks)
        frame = draw_events(frame, events)

        for event in events:
            print(
                f"[EVENT] "
                f"Frame={event['frame']} | "
                f"TrackID={event['track_id']} | "
                f"Type={event['event']} | "
                f"Zone={event['zone']}"
            )

        # FPS calculation
        current_time = time.time()

        current_fps = 1 / (current_time - prev_time)

        prev_time = current_time

        # draw FPS
        frame = draw_fps(frame, current_fps, frame_num, total_frames)

        # minimal logs every 30 frames
        if frame_num % 30 == 0:

            print(
                f"[INFO] "
                f"Frame={frame_num}/{total_frames} | "
                f"Detections={len(detections)} | "
                f"Tracks={len(tracks)} | "
                f"FPS={current_fps:.2f}"
            )

        # write video
        writer.write(frame)

        # show output
        cv2.imshow("Video Surveillance", frame)

        # quit
        if cv2.waitKey(1) & 0xFF == ord('q'):

            print("[INFO] Interrupted by user.")
            break

        frame_num += 1

    print("\n[INFO] Saving outputs...")

    cap.release()

    writer.release()

    event_manager.close()

    cv2.destroyAllWindows()

    print("[INFO] Output video saved:")
    print(f"{output_dir}/annotated.mp4")

    print("[INFO] Event log saved:")
    print(f"  CSV:  {output_dir}/events.csv")
    print(f"  JSON: {output_dir}/events.json")

    print("\n===================================")
    print("PROCESS COMPLETED SUCCESSFULLY")
    print("===================================")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--video',
        required=True,
        help='Input video path'
    )

    parser.add_argument(
        '--zones',
        required=True,
        help='Zones JSON path'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Output folder'
    )

    parser.add_argument(
        '--model',
        default='yolov8n.pt',
        help='YOLOv8 model path or name (default: yolov8n.pt)'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='Detection confidence threshold (default: 0.5)'
    )

    args = parser.parse_args()

    main(
        args.video,
        args.zones,
        args.output,
        model_path=args.model,
        conf_threshold=args.conf
    )