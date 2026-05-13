import csv
import json
import time
import math


class EventManager:

    # Minimum frames between repeated alerts for the same track+zone+event
    INTRUSION_COOLDOWN_FRAMES = 30
    DEFAULT_STATIONARY_THRESHOLD_PX = 25

    def __init__(self, csv_path, json_path=None):

        # Per-track state: zone_entry_time, zone_membership, last_alert_frame
        self.track_state = {}

        # All events accumulated for JSON dump at end
        self.all_events = []

        self.json_path = json_path

        self.csv_file = open(csv_path, 'w', newline='')

        self.writer = csv.writer(self.csv_file)

        self.writer.writerow([
            'timestamp',
            'frame',
            'track_id',
            'event_type',
            'zone_name',
            'bbox_x1',
            'bbox_y1',
            'bbox_x2',
            'bbox_y2',
            'confidence'
        ])

    def _init_track(self, track_id):
        if track_id not in self.track_state:
            self.track_state[track_id] = {
                # zone_name -> entry timestamp (set when track enters zone)
                'zone_entry_time': {},
                # zone_name -> stationary movement state
                'zone_motion': {},
                # zone_name -> last intrusion alert frame (deduplication)
                'last_intrusion_frame': {},
                # zone_name -> loitering alert already fired (reset on exit)
                'loitering_alerted': set(),
            }

    @staticmethod
    def _center_from_bbox(bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @staticmethod
    def _distance(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def process(self, frame_num, tracks, matched_zones):

        current_time = time.time()
        events = []

        # Build a set of currently active track IDs for exit detection
        active_ids = {track['id'] for track in tracks}

        # Detect zone exits for tracks that are gone this frame
        for track_id, state in self.track_state.items():
            if track_id not in active_ids:
                # Track lost — clear its zone memberships so timers reset on re-entry
                state['zone_entry_time'].clear()
                state['zone_motion'].clear()
                state['loitering_alerted'].clear()

        for track in tracks:

            track_id = track['id']
            bbox = track['bbox']
            x1, y1, x2, y2 = bbox
            center = self._center_from_bbox(bbox)
            conf = track.get('confidence', 1.0)

            self._init_track(track_id)
            state = self.track_state[track_id]

            current_zone_names = {z['name'] for z in matched_zones.get(track_id, [])}

            # Detect zone exits (track was in zone last frame, no longer is)
            exited = set(state['zone_entry_time'].keys()) - current_zone_names
            for zone_name in exited:
                state['zone_entry_time'].pop(zone_name, None)
                state['zone_motion'].pop(zone_name, None)
                state['loitering_alerted'].discard(zone_name)

            for zone in matched_zones.get(track_id, []):

                zone_name = zone['name']
                event_type = zone['type']

                # Record entry time the first time we see this track in this zone
                if zone_name not in state['zone_entry_time']:
                    state['zone_entry_time'][zone_name] = current_time
                    state['zone_motion'][zone_name] = {
                        'anchor_center': center,
                        'last_center': center,
                        'stationary_start': current_time,
                    }

                # --- Intrusion event (deduplicated by cooldown) ---
                if event_type == 'intrusion':

                    last_frame = state['last_intrusion_frame'].get(zone_name, -self.INTRUSION_COOLDOWN_FRAMES)

                    if frame_num - last_frame >= self.INTRUSION_COOLDOWN_FRAMES:

                        state['last_intrusion_frame'][zone_name] = frame_num

                        event = {
                            'frame': frame_num,
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'track_id': track_id,
                            'event': 'zone_intrusion',
                            'zone': zone_name,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': round(conf, 4),
                        }

                        events.append(event)

                # --- Loitering event (fires once per zone-entry period) ---
                elif event_type == 'loitering':

                    if zone_name not in state['loitering_alerted']:

                        motion_state = state['zone_motion'].setdefault(
                            zone_name,
                            {
                                'anchor_center': center,
                                'last_center': center,
                                'stationary_start': current_time,
                            }
                        )

                        stationary_threshold = zone.get(
                            'stationary_threshold_px',
                            self.DEFAULT_STATIONARY_THRESHOLD_PX
                        )

                        move_step = self._distance(center, motion_state['last_center'])
                        move_from_anchor = self._distance(center, motion_state['anchor_center'])

                        if move_step > stationary_threshold or move_from_anchor > stationary_threshold:
                            motion_state['anchor_center'] = center
                            motion_state['stationary_start'] = current_time

                        motion_state['last_center'] = center

                        stationary_duration = current_time - motion_state['stationary_start']

                        if stationary_duration >= zone.get('loiter_time', 10):

                            state['loitering_alerted'].add(zone_name)

                            event = {
                                'frame': frame_num,
                                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                                'track_id': track_id,
                                'event': 'loitering',
                                'zone': zone_name,
                                'bbox': [x1, y1, x2, y2],
                                'confidence': round(conf, 4),
                            }

                            events.append(event)

        for e in events:

            x1, y1, x2, y2 = e['bbox']

            self.writer.writerow([
                e['timestamp'],
                e['frame'],
                e['track_id'],
                e['event'],
                e['zone'],
                x1, y1, x2, y2,
                e['confidence'],
            ])

            self.all_events.append(e)

        return events

    def close(self):
        """Flush CSV and write JSON summary."""
        self.csv_file.close()

        if self.json_path:
            with open(self.json_path, 'w') as f:
                json.dump(self.all_events, f, indent=2)