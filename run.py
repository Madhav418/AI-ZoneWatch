import cv2
import argparse
import time

from core.detector import PersonDetector
from core.tracker import Tracker
from core.zone_manager import ZoneManager
from core.event_manager import EventManager


def draw_zones(frame, zones):

    for zone in zones:

        pts = list(zone['polygon'].exterior.coords)

        pts = [(int(x), int(y)) for x, y in pts]

        for i in range(len(pts) - 1):

            cv2.line(
                frame,
                pts[i],
                pts[i + 1],
                (0, 255, 255),
                2
            )

        # zone name
        x, y = pts[0]

        cv2.putText(
            frame,
            zone['name'],
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    return frame


def main(video_path, zone_file, output_dir):

    print("===================================")
    print("VIDEO SURVEILLANCE SYSTEM STARTED")
    print("===================================")

    # detector
    print("[INFO] Loading YOLO model...")
    detector = PersonDetector()

    # tracker
    print("[INFO] Initializing ByteTrack...")
    tracker = Tracker()

    # zones
    print("[INFO] Loading zones...")
    zone_manager = ZoneManager(zone_file)

    # event manager
    print("[INFO] Initializing event manager...")
    event_manager = EventManager(
        f'{output_dir}/events.csv'
    )

    # video
    print("[INFO] Opening video...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print("[ERROR] Unable to open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Width: {width}")
    print(f"[INFO] Height: {height}")
    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Total Frames: {total_frames}")

    # output writer
    writer = cv2.VideoWriter(
        f'{output_dir}/annotated.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    frame_num = 0

    prev_time = time.time()

    print("\n[INFO] Starting processing...\n")

    while True:

        ret, frame = cap.read()
        frame = cv2.resize(frame, (960, 540))
    
        if not ret:

            print("\n[INFO] Video processing completed.")
            break

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

        # draw zones
        frame = draw_zones(
            frame,
            zone_manager.zones
        )

        # draw tracks
        for track in tracks:

            x1, y1, x2, y2 = track['bbox']

            track_id = track['id']

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f'ID:{track_id}',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # draw events
        for event in events:

            cv2.putText(
                frame,
                event['event'],
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

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
        cv2.putText(
            frame,
            f'FPS: {current_fps:.2f}',
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

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

    cv2.destroyAllWindows()

    print("[INFO] Output video saved:")
    print(f"{output_dir}/annotated.mp4")

    print("[INFO] Event log saved:")
    print(f"{output_dir}/events.csv")

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

    args = parser.parse_args()

    main(
        args.video,
        args.zones,
        args.output
    )