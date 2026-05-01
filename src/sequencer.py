import threading
import time

from .audio import AudioPlayer
from .detector import EMPTY, GridState


class BeatSequencer:
    """
    Advances through beat columns at a fixed BPM in a daemon thread.

    Uses absolute next-tick timestamps (perf_counter) to prevent cumulative drift.
    BPM and paused state are writable from the main thread (GIL-safe primitives).
    """

    def __init__(
        self,
        grid_state: GridState,
        audio: AudioPlayer,
        cols: int,
        bpm: float = 120.0,
    ):
        self.grid_state = grid_state
        self.audio = audio
        self.cols = cols
        self.bpm: float = bpm
        self.paused: bool = False
        self.current_col: int = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _interval(self) -> float:
        return 60.0 / max(self.bpm, 1.0)

    def _run(self) -> None:
        next_tick = time.perf_counter() + self._interval()
        while not self._stop.is_set():
            now = time.perf_counter()
            wait = next_tick - now
            if wait > 0:
                time.sleep(wait)
            if self._stop.is_set():
                break
            if not self.paused:
                self._fire(self.current_col)
                self.current_col = (self.current_col + 1) % self.cols
            next_tick += self._interval()

    def _fire(self, col: int) -> None:
        snap = self.grid_state.snapshot()  # int8 array
        for row, orientation in enumerate(snap[:, col]):
            if orientation != EMPTY:
                self.audio.play(row, int(orientation))
