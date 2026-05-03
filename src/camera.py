import abc
import queue
import threading

import cv2
import numpy as np


class BaseCameraCapture(abc.ABC):
    def __init__(self, camera_id: int):
        self.camera_id = camera_id

    @abc.abstractmethod
    def open(self) -> bool: ...

    @abc.abstractmethod
    def read_frame(self) -> tuple[bool, np.ndarray | None]: ...

    @abc.abstractmethod
    def release(self) -> None: ...

    def __enter__(self) -> "BaseCameraCapture":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()


class LinuxCamera(BaseCameraCapture):
    """OpenCV VideoCapture for Linux (V4L2). Extend for Mac via MacCamera."""

    def __init__(self, camera_id: int):
        super().__init__(camera_id)
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_id)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self._cap.isOpened()

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ok, frame = self._cap.read()
        return (ok, frame) if ok else (False, None)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class CameraThread:
    """Reads frames in a daemon thread; always serves the freshest frame."""

    def __init__(self, camera: BaseCameraCapture, rotate_degrees: int = 0):
        self._camera = camera
        self._rotate_code = _ROTATE_CODES.get(rotate_degrees % 360)
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._camera.read_frame()
            if not ok or frame is None:
                continue
            if self._rotate_code is not None:
                frame = cv2.rotate(frame, self._rotate_code)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass
