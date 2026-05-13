import cv2
import time


class VideoProcessor:
    """
    Wraps an OpenCV VideoCapture and provides:
    - Frame iteration with optional stride (frame-skip for speed)
    - Real-time FPS measurement
    - Video metadata access
    - Annotated video writer setup
    """

    def __init__(self, video_path: str, output_path: str, frame_stride: int = 1, resize: tuple = (960, 540)):
        self.video_path = video_path
        self.frame_stride = max(1, frame_stride)
        self.resize = resize

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_w, out_h = resize if resize else (self.width, self.height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (out_w, out_h))

        self._prev_time = None
        self.current_fps = 0.0

    @property
    def metadata(self) -> dict:
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames,
        }

    def frames(self):
        """
        Generator that yields (frame_number, frame).
        Skips frames according to frame_stride for speed.
        Resizes to self.resize if set.
        Updates self.current_fps on each yield.
        """
        frame_num = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_num % self.frame_stride == 0:
                now = time.time()
                if self._prev_time is not None:
                    elapsed = now - self._prev_time
                    self.current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                self._prev_time = now

                if self.resize:
                    frame = cv2.resize(frame, self.resize)

                yield frame_num, frame

            frame_num += 1

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.cap.release()
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
