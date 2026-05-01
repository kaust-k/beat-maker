import threading

import cv2
import numpy as np


class GridState:
    """
    Thread-safe 2D boolean grid tracking which cells have a tile.

    Hysteresis prevents flicker during occlusion:
      coverage >= tile_threshold  -> tile present
      coverage <= empty_threshold -> tile absent
      between                     -> keep previous state
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        tile_threshold: float = 0.15,
        empty_threshold: float = 0.05,
    ):
        self.rows = rows
        self.cols = cols
        self.tile_threshold = tile_threshold
        self.empty_threshold = empty_threshold
        self._lock = threading.Lock()
        self._tiles = np.zeros((rows, cols), dtype=bool)

    def snapshot(self) -> np.ndarray:
        with self._lock:
            return self._tiles.copy()

    def update(self, coverage: np.ndarray) -> None:
        with self._lock:
            self._tiles[coverage >= self.tile_threshold] = True
            self._tiles[coverage <= self.empty_threshold] = False
            # Cells in the hysteresis band are unchanged (occlusion hold)


class ColorTileDetector:
    """
    Detects colored tile presence per grid cell using HSV segmentation.

    Each row/track maps to one named color. Multiple HSV ranges per color
    are supported (e.g., red wraps around H=0/179 in OpenCV HSV).
    """

    def __init__(
        self,
        tracks: list[dict],
        color_defs: dict,
        rows: int,
        cols: int,
        cell_size: int,
    ):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self._track_ranges = self._build_track_ranges(tracks, color_defs)

    def _build_track_ranges(
        self, tracks: list[dict], color_defs: dict
    ) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        result = []
        for track in tracks:
            color_name = track["color"]
            ranges = []
            for key, defn in color_defs.items():
                if key == color_name or key.startswith(color_name + "_"):
                    lo = np.array(defn["lower"], dtype=np.uint8)
                    hi = np.array(defn["upper"], dtype=np.uint8)
                    ranges.append((lo, hi))
            result.append(ranges)
        return result

    def detect(self, warped_frame: np.ndarray) -> np.ndarray:
        """
        Returns float array (rows, cols) with per-cell color coverage [0, 1].
        Row index = track/instrument, col index = beat position.
        """
        hsv = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2HSV)
        cs = self.cell_size
        cell_pixels = cs * cs
        coverage = np.zeros((self.rows, self.cols), dtype=float)

        for row_idx, ranges in enumerate(self._track_ranges):
            if not ranges:
                continue
            for col_idx in range(self.cols):
                cell_hsv = hsv[
                    row_idx * cs : (row_idx + 1) * cs,
                    col_idx * cs : (col_idx + 1) * cs,
                ]
                mask = np.zeros((cs, cs), dtype=np.uint8)
                for lo, hi in ranges:
                    mask |= cv2.inRange(cell_hsv, lo, hi)
                coverage[row_idx, col_idx] = mask.sum() / (255 * cell_pixels)

        return coverage
