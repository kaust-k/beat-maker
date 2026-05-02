import json
from pathlib import Path

import cv2
import numpy as np

CORNER_ORDER = ["top-left", "top-right", "bottom-right", "bottom-left"]


class GridMapper:
    """
    Manages perspective calibration and warped cell extraction.

    Calibration: user clicks 4 corners (TL→TR→BR→BL) on the raw camera frame.
    After calibration, warp() produces a bird's-eye view divided into a regular grid.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        cell_size: int,
        calib_path: str = "calibration.json",
        selector_rows: int = 0,
    ):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.selector_rows = selector_rows
        self.calib_path = Path(calib_path)
        self._warp_matrix: np.ndarray | None = None
        self._corners: list[tuple[int, int]] = []
        self._click_buffer: list[tuple[int, int]] = []
        self.calibrating: bool = False

        # Destination corners for the full warped area (selector strip + beat grid)
        w = cols * cell_size
        h = (selector_rows + rows) * cell_size
        self._dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    @property
    def is_calibrated(self) -> bool:
        return self._warp_matrix is not None

    def load(self) -> bool:
        if not self.calib_path.exists():
            return False
        try:
            data = json.loads(self.calib_path.read_text())
            stored_rows = data.get("grid_rows")
            stored_sel = data.get("selector_rows", 0)
            stored_cols = data.get("cols")
            stored_cs = data.get("cell_size")
            if stored_rows is not None and (
                stored_rows != self.rows
                or stored_sel != self.selector_rows
                or stored_cols != self.cols
                or stored_cs != self.cell_size
            ):
                print(
                    "Calibration grid dimensions changed — recalibration required. "
                    "Press 'c' to recalibrate."
                )
                return False
            self._corners = [tuple(p) for p in data["corners"]]
            self._warp_matrix = np.array(data["warp_matrix"], dtype=np.float64)
            return True
        except Exception:
            return False

    def save(self) -> None:
        data = {
            "corners": [list(p) for p in self._corners],
            "warp_matrix": self._warp_matrix.tolist(),
            "grid_rows": self.rows,
            "selector_rows": self.selector_rows,
            "cols": self.cols,
            "cell_size": self.cell_size,
        }
        self.calib_path.write_text(json.dumps(data, indent=2))

    def start_calibration(self) -> None:
        self._click_buffer = []
        self.calibrating = True

    def add_click(self, x: int, y: int) -> bool:
        """Register one corner click. Returns True when all 4 are collected."""
        if not self.calibrating:
            return False
        self._click_buffer.append((x, y))
        if len(self._click_buffer) < 4:
            return False
        src = np.float32(self._click_buffer)
        self._warp_matrix = cv2.getPerspectiveTransform(src, self._dst)
        self._corners = list(self._click_buffer)
        self._click_buffer = []
        self.calibrating = False
        self.save()
        return True

    def next_corner_name(self) -> str:
        idx = len(self._click_buffer)
        return CORNER_ORDER[idx] if idx < 4 else ""

    def warp(self, frame: np.ndarray) -> np.ndarray:
        h = (self.selector_rows + self.rows) * self.cell_size
        w = self.cols * self.cell_size
        return cv2.warpPerspective(frame, self._warp_matrix, (w, h))

    def cell_roi(self, warped: np.ndarray, row: int, col: int) -> np.ndarray:
        cs = self.cell_size
        return warped[row * cs : (row + 1) * cs, col * cs : (col + 1) * cs]

    @property
    def click_buffer(self) -> list[tuple[int, int]]:
        return self._click_buffer
