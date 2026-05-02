import threading

import cv2
import numpy as np

# Tile state constants — used by GridState, AudioPlayer, and BeatSequencer
EMPTY      = 0
HORIZONTAL = 1  # tile wider than tall (landscape)
VERTICAL   = 2  # tile taller than wide (portrait)


class GridState:
    """
    Thread-safe 2D grid tracking tile orientation per cell (EMPTY/HORIZONTAL/VERTICAL).

    Uses temporal debouncing: a cell must be detected for min_present_frames consecutive
    frames before being accepted, and absent for min_absent_frames frames before being
    cleared. This filters fast-moving hands (~5–10 frames) while accepting stable tiles.
    The hysteresis band (empty_threshold < area < tile_threshold) freezes all counters
    to handle brief occlusions without resetting the accumulation.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        tile_threshold: float = 0.15,
        empty_threshold: float = 0.05,
        min_present_frames: int = 3,
        min_absent_frames: int = 2,
    ):
        self.rows = rows
        self.cols = cols
        self.tile_threshold = tile_threshold
        self.empty_threshold = empty_threshold
        self.min_present_frames = min_present_frames
        self.min_absent_frames = min_absent_frames
        self._lock = threading.Lock()
        self._tiles = np.zeros((rows, cols), dtype=np.int8)
        self._present_count = np.zeros((rows, cols), dtype=np.int16)
        self._absent_count  = np.zeros((rows, cols), dtype=np.int16)
        self._pending_orient = np.zeros((rows, cols), dtype=np.int8)

    def snapshot(self) -> np.ndarray:
        with self._lock:
            return self._tiles.copy()

    def update(self, orientations: np.ndarray, areas: np.ndarray) -> None:
        """
        orientations: int8 array (rows×cols) with EMPTY/HORIZONTAL/VERTICAL values
        areas:        float array (rows×cols) with normalized coverage [0, 1]
        """
        with self._lock:
            present = areas >= self.tile_threshold
            absent  = areas <= self.empty_threshold
            # hysteresis band: neither mask is True — counters and tiles are frozen

            # Present zone: accumulate present frames, reset absent counter
            self._present_count[present] += 1
            self._absent_count[present]   = 0
            self._pending_orient[present] = orientations[present]

            # Absent zone: accumulate absent frames, reset present counter
            self._absent_count[absent] += 1
            self._present_count[absent]  = 0

            # Accept tile after enough consecutive present frames
            accept = present & (self._present_count >= self.min_present_frames)
            self._tiles[accept] = self._pending_orient[accept]

            # Remove tile after enough consecutive absent frames
            remove = absent & (self._absent_count >= self.min_absent_frames)
            self._tiles[remove] = EMPTY


class ColorTileDetector:
    """
    Detects tile presence per cell using HSV color segmentation.
    Each row maps to a named color; all detected tiles are treated as HORIZONTAL
    (color mode has no orientation information).

    Implements the unified detect() interface: returns (orientations, areas).
    """

    def __init__(
        self,
        tracks: list[dict],
        color_defs: dict,
        rows: int,
        cols: int,
        cell_size: int,
        tile_threshold: float = 0.15,
    ):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.tile_threshold = tile_threshold
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

    def detect(self, warped_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (orientations, areas):
          orientations: int8 (rows×cols) — HORIZONTAL where tile detected, else EMPTY
          areas:        float (rows×cols) — HSV color coverage fraction [0, 1]
        """
        hsv = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2HSV)
        cs = self.cell_size
        cell_pixels = cs * cs
        areas = np.zeros((self.rows, self.cols), dtype=float)

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
                areas[row_idx, col_idx] = mask.sum() / (255 * cell_pixels)

        orientations = np.where(
            areas >= self.tile_threshold, HORIZONTAL, EMPTY
        ).astype(np.int8)
        return orientations, areas


class OrientationTileDetector:
    """
    Detects tile presence and orientation per cell using HSV saturation thresholding.
    Color is irrelevant — any colored tile against white paper is detected.
    Orientation (HORIZONTAL/VERTICAL) is determined from the bounding rect aspect ratio.

    Implements the unified detect() interface: returns (orientations, areas).
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        cell_size: int,
        sat_threshold: int = 40,
        min_area_ratio: float = 0.10,
        h_ratio_threshold: float = 1.25,
        v_ratio_threshold: float = 0.80,
        baseline_s: np.ndarray | None = None,
    ):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.sat_threshold = sat_threshold
        self.min_area_ratio = min_area_ratio
        self.h_ratio = h_ratio_threshold
        self.v_ratio = v_ratio_threshold
        self._baseline_s = baseline_s  # shape (rows, cols), float32 or None

    def detect(self, warped_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (orientations, areas):
          orientations: int8 (rows×cols) — EMPTY / HORIZONTAL / VERTICAL
          areas:        float (rows×cols) — contour area / cell area [0, 1]
        """
        hsv = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2HSV)
        cs = self.cell_size
        cell_area = float(cs * cs)
        min_contour_area = self.min_area_ratio * cell_area

        orientations = np.zeros((self.rows, self.cols), dtype=np.int8)
        areas = np.zeros((self.rows, self.cols), dtype=float)

        for r in range(self.rows):
            for c in range(self.cols):
                s_channel = hsv[r * cs : (r + 1) * cs, c * cs : (c + 1) * cs, 1]
                s_blurred = cv2.GaussianBlur(s_channel, (5, 5), 0)
                # Subtract per-cell baseline saturation when available so threshold
                # is relative to the background paper under current lighting.
                if self._baseline_s is not None and r < self._baseline_s.shape[0]:
                    base = np.uint8(min(int(self._baseline_s[r, c]), 255))
                    s_adj = cv2.subtract(s_blurred, np.full_like(s_blurred, base))
                else:
                    s_adj = s_blurred
                _, mask = cv2.threshold(
                    s_adj, self.sat_threshold, 255, cv2.THRESH_BINARY
                )
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                valid = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contour_area]
                if not valid:
                    continue

                best = max(valid, key=cv2.contourArea)
                contour_area = cv2.contourArea(best)
                areas[r, c] = contour_area / cell_area

                x, y, w, h = cv2.boundingRect(best)
                ratio = w / max(h, 1)
                if ratio > self.h_ratio:
                    orientations[r, c] = HORIZONTAL
                elif ratio < self.v_ratio:
                    orientations[r, c] = VERTICAL
                else:
                    orientations[r, c] = HORIZONTAL  # square tile → treat as horizontal

        return orientations, areas


class SelectorDetector:
    """
    Detects which cell in the profile selector strip (top row of the warped image)
    has a tile, using HSV saturation thresholding. Position (column) determines
    the profile index — any colored tile works.
    """

    def __init__(
        self,
        n_profiles: int,
        cols: int,
        cell_size: int,
        sat_threshold: int = 40,
        min_area_ratio: float = 0.10,
        baseline_s: np.ndarray | None = None,
    ):
        self.n_profiles = n_profiles
        self.cols = cols
        self.cell_size = cell_size
        self.sat_threshold = sat_threshold
        self.min_area_ratio = min_area_ratio
        self._baseline_s = baseline_s  # shape (selector_rows, cols), float32 or None

    def detect(self, warped_frame: np.ndarray) -> int | None:
        """
        Returns the 0-based profile index of the cell with the largest tile,
        or None if no cell meets the minimum area threshold.
        Scans columns 0..min(n_profiles, cols)-1 in the first row of the warped frame.
        """
        cs = self.cell_size
        cell_area = float(cs * cs)
        min_contour_area = self.min_area_ratio * cell_area
        hsv = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2HSV)

        best_col: int | None = None
        best_area: float = 0.0
        n = min(self.n_profiles, self.cols)
        for c in range(n):
            s_channel = hsv[0:cs, c * cs:(c + 1) * cs, 1]
            s_blurred = cv2.GaussianBlur(s_channel, (5, 5), 0)
            if self._baseline_s is not None and self._baseline_s.shape[0] > 0:
                base = np.uint8(min(int(self._baseline_s[0, c]), 255))
                s_adj = cv2.subtract(s_blurred, np.full_like(s_blurred, base))
            else:
                s_adj = s_blurred
            _, mask = cv2.threshold(
                s_adj, self.sat_threshold, 255, cv2.THRESH_BINARY
            )
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            valid = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contour_area]
            if valid:
                area = cv2.contourArea(max(valid, key=cv2.contourArea)) / cell_area
                if area > best_area:
                    best_area = area
                    best_col = c

        return best_col
