from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracker:

    def __init__(self):

        self.tracker = DeepSort(

            max_age=30,

            n_init=2,

            max_cosine_distance=0.3,

            nn_budget=50,

            embedder="mobilenet",

            half=True,

            bgr=True
        )

    def update(self, detections, frame):

        detection_list = []

        for det in detections:

            x1, y1, x2, y2, conf = det

            w = x2 - x1
            h = y2 - y1

            detection_list.append(
                (
                    [x1, y1, w, h],
                    conf,
                    'person'
                )
            )

        tracks = self.tracker.update_tracks(
            detection_list,
            frame=frame
        )

        results = []

        for track in tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id

            ltrb = track.to_ltrb()

            x1, y1, x2, y2 = map(int, ltrb)

            results.append({

                'id': int(track_id),

                'bbox': [x1, y1, x2, y2],

                'confidence': 1.0
            })

        return results