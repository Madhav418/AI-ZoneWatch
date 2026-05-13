import cv2
import numpy as np


# ── Colour palette ────────────────────────────────────────────────────────────
ZONE_COLOUR = (0, 255, 255)       # cyan
TRACK_COLOUR = (0, 255, 0)        # green
ALERT_COLOUR = (0, 0, 255)        # red
FPS_COLOUR = (255, 80, 0)         # blue-orange


def draw_zones(frame: np.ndarray, zones: list) -> np.ndarray:
    """
    Draws every zone polygon and its name onto *frame* in-place.
    Zones are semi-transparent filled polygons so they are visible
    without obscuring detections.
    """
    overlay = frame.copy()

    for zone in zones:
        pts = list(zone['polygon'].exterior.coords[:-1])   # drop repeated last point
        pts_arr = np.array([(int(x), int(y)) for x, y in pts], dtype=np.int32)

        # fill
        cv2.fillPoly(overlay, [pts_arr], ZONE_COLOUR)

    # blend at 20 % opacity
    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

    # draw outlines and labels on top (fully opaque)
    for zone in zones:
        pts = list(zone['polygon'].exterior.coords[:-1])
        pts_arr = np.array([(int(x), int(y)) for x, y in pts], dtype=np.int32)

        cv2.polylines(frame, [pts_arr], isClosed=True, color=ZONE_COLOUR, thickness=2)

        lx, ly = pts_arr[0]
        cv2.putText(
            frame, zone['name'],
            (lx, ly - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, ZONE_COLOUR, 2
        )

    return frame


def draw_tracks(frame: np.ndarray, tracks: list) -> np.ndarray:
    """Draws bounding boxes and track IDs onto *frame* in-place."""
    for track in tracks:
        x1, y1, x2, y2 = track['bbox']
        tid = track['id']
        conf = track.get('confidence', 1.0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), TRACK_COLOUR, 2)
        cv2.putText(
            frame,
            f'ID:{tid} {conf:.2f}',
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, TRACK_COLOUR, 2
        )

    return frame


def draw_fps(frame: np.ndarray, fps: float, frame_num: int, total: int) -> np.ndarray:
    """Draws FPS and frame counter in the top-left corner."""
    cv2.putText(
        frame,
        f'FPS: {fps:.1f}  Frame: {frame_num}/{total}',
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, FPS_COLOUR, 2
    )
    return frame


def draw_events(frame: np.ndarray, events: list) -> np.ndarray:
    """Draws active event labels stacked vertically below the FPS counter."""
    for idx, event in enumerate(events):
        label = f"{event['event'].upper()} | Zone:{event['zone']} | ID:{event['track_id']}"
        cv2.putText(
            frame, label,
            (10, 60 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, ALERT_COLOUR, 2
        )
    return frame


def point_in_polygon(px: int, py: int, polygon_pts: list) -> bool:
    """
    Pure-NumPy ray-casting point-in-polygon test.
    polygon_pts is a list of (x, y) tuples.
    Kept here as a lightweight fallback — ZoneManager uses Shapely.
    """
    n = len(polygon_pts)
    inside = False
    xp, yp = float(px), float(py)
    x0, y0 = polygon_pts[-1]
    for x1, y1 in polygon_pts:
        if ((y1 > yp) != (y0 > yp)) and (xp < (x0 - x1) * (yp - y1) / (y0 - y1 + 1e-9) + x1):
            inside = not inside
        x0, y0 = x1, y1
    return inside
