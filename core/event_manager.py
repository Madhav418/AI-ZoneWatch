import csv
import time


class EventManager:

    def __init__(self, output_csv):

        self.track_history = {}

        self.csv_file = open(
            output_csv,
            'w',
            newline=''
        )

        self.writer = csv.writer(self.csv_file)

        self.writer.writerow([
            'timestamp',
            'frame',
            'track_id',
            'event_type',
            'zone_name',
            'confidence'
        ])

    def process(
        self,
        frame_num,
        tracks,
        matched_zones
    ):

        current_time = time.time()

        events = []

        for track in tracks:

            track_id = track['id']

            if track_id not in self.track_history:

                self.track_history[track_id] = {
                    'first_seen': current_time,
                    'last_seen': current_time
                }

            self.track_history[track_id][
                'last_seen'
            ] = current_time

            zones = matched_zones.get(track_id, [])

            for zone in zones:

                event_type = zone['type']

                # intrusion
                if event_type == 'intrusion':

                    event = {
                        'frame': frame_num,
                        'track_id': track_id,
                        'event': 'zone_intrusion',
                        'zone': zone['name']
                    }

                    events.append(event)

                # loitering
                elif event_type == 'loitering':

                    duration = (
                        current_time -
                        self.track_history[track_id][
                            'first_seen'
                        ]
                    )

                    if duration >= zone['loiter_time']:

                        event = {
                            'frame': frame_num,
                            'track_id': track_id,
                            'event': 'loitering',
                            'zone': zone['name']
                        }

                        events.append(event)

        for e in events:

            self.writer.writerow([
                time.strftime('%Y-%m-%d %H:%M:%S'),
                e['frame'],
                e['track_id'],
                e['event'],
                e['zone'],
                1.0
            ])

        return events