import os
import queue
import sys
import warnings
from pathlib import Path

# Wayland guard: force X11 backend for OpenCV windows on Ubuntu
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# pygame prints an AVX2 RuntimeWarning on systems where it wasn't compiled with AVX2
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*avx2.*")

import cv2
import numpy as np
import pygame
import yaml

from .audio import AudioPlayer
from .camera import CameraThread, LinuxCamera
from .detector import ColorTileDetector, GridState
from .grid import CORNER_ORDER, GridMapper
from .sequencer import BeatSequencer

CONFIG_PATH = "config.yaml"
CALIB_PATH = "calibration.json"
WINDOW_NAME = "Tembo Beat Maker"

# BGR colors for each track overlay (matches default color order: red/blue/green/yellow/orange/purple)
TRACK_COLORS_BGR = [
    (40, 40, 220),    # kick:   red
    (200, 80, 40),    # snare:  blue
    (40, 180, 40),    # hihat:  green
    (40, 220, 220),   # clap:   yellow
    (40, 140, 220),   # tom:    orange
    (200, 40, 200),   # cymbal: purple
]


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _draw_grid_overlay(
    warped: np.ndarray,
    grid_state: GridState,
    sequencer: BeatSequencer,
    cell_size: int,
) -> np.ndarray:
    overlay = warped.copy()
    rows, cols = grid_state.rows, grid_state.cols
    snap = grid_state.snapshot()
    cs = cell_size

    # Semi-transparent fill for active cells
    cell_layer = warped.copy()
    for r in range(rows):
        color = TRACK_COLORS_BGR[r % len(TRACK_COLORS_BGR)]
        for c in range(cols):
            if snap[r, c]:
                cv2.rectangle(
                    cell_layer, (c * cs, r * cs), ((c + 1) * cs, (r + 1) * cs),
                    color, -1,
                )
    cv2.addWeighted(cell_layer, 0.5, overlay, 0.5, 0, overlay)

    # Current beat column highlight (yellow, translucent)
    beat_col = sequencer.current_col
    col_layer = overlay.copy()
    cv2.rectangle(
        col_layer, (beat_col * cs, 0), ((beat_col + 1) * cs, rows * cs),
        (0, 255, 255), -1,
    )
    cv2.addWeighted(col_layer, 0.35, overlay, 0.65, 0, overlay)

    # Grid lines
    for c in range(cols + 1):
        cv2.line(overlay, (c * cs, 0), (c * cs, rows * cs), (80, 80, 80), 1)
    for r in range(rows + 1):
        cv2.line(overlay, (0, r * cs), (cols * cs, r * cs), (80, 80, 80), 1)

    # Row labels (instrument names — shown as row index for now)
    for r in range(rows):
        cv2.putText(
            overlay, str(r), (4, r * cs + cs // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
        )

    return overlay


def _draw_calibration_overlay(frame: np.ndarray, mapper: GridMapper) -> np.ndarray:
    out = frame.copy()
    for i, (cx, cy) in enumerate(mapper.click_buffer):
        cv2.drawMarker(out, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(
            out, CORNER_ORDER[i], (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )
    next_name = mapper.next_corner_name()
    cv2.putText(
        out, f"Click: {next_name}  ({len(mapper.click_buffer)}/4)",
        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
    )
    return out


def _draw_hud(frame: np.ndarray, sequencer: BeatSequencer) -> None:
    if sequencer.paused:
        label = "PAUSED"
        color = (100, 100, 255)
    else:
        label = f"{sequencer.bpm:.0f} BPM"
        color = (255, 255, 0)
    cv2.putText(
        frame, label, (10, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
    )
    # Controls hint
    cv2.putText(
        frame, "q:quit  c:calibrate  +/-:tempo  space:pause  r:reset",
        (10, frame.shape[0] - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1,
    )


def run() -> None:
    cfg_path = Path(CONFIG_PATH)
    if not cfg_path.exists():
        print(f"ERROR: {CONFIG_PATH} not found. Run from the project root directory.")
        sys.exit(1)

    cfg = _load_config(CONFIG_PATH)
    rows = cfg["grid"]["rows"]
    cols = cfg["grid"]["cols"]
    cell_size = cfg["display"]["cell_size"]
    tracks = cfg["sounds"]["tracks"]
    color_defs = cfg["colors"]

    # Audio — must pre-init mixer before pygame.init()
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    pygame.mixer.set_num_channels(16)

    audio = AudioPlayer(tracks, cfg["sounds"]["dir"])
    audio.load_all()
    print(f"Loaded {len(tracks)} instrument sounds.")

    grid_state = GridState(
        rows, cols,
        tile_threshold=cfg["detection"]["tile_threshold"],
        empty_threshold=cfg["detection"]["empty_threshold"],
    )
    detector = ColorTileDetector(tracks, color_defs, rows, cols, cell_size)

    camera = LinuxCamera(cfg["camera_id"])
    if not camera.open():
        print(f"ERROR: Cannot open camera {cfg['camera_id']}.")
        pygame.quit()
        sys.exit(1)
    cam_thread = CameraThread(camera)
    cam_thread.start()

    mapper = GridMapper(rows, cols, cell_size, CALIB_PATH)
    if mapper.load():
        print("Calibration loaded from calibration.json.")
    else:
        print("No calibration found. Press 'c' to calibrate.")

    sequencer = BeatSequencer(grid_state, audio, cols, cfg["tempo_bpm"])
    sequencer.start()

    cv2.namedWindow(WINDOW_NAME)

    # State shared with mouse callback
    cb_state = {"mapper": mapper}

    def mouse_callback(event, x, y, flags, param):
        m: GridMapper = param["mapper"]
        if event == cv2.EVENT_LBUTTONDOWN and m.calibrating:
            # Only accept clicks on the LEFT (raw camera) panel
            if x < param.get("raw_panel_w", 99999):
                done = m.add_click(x, y)
                if done:
                    print("Calibration complete and saved.")

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, cb_state)

    last_frame: np.ndarray | None = None
    grid_h = rows * cell_size
    grid_w = cols * cell_size

    try:
        while True:
            # Get freshest frame (non-blocking; fall back to last known)
            try:
                frame = cam_thread.frame_queue.get_nowait()
                last_frame = frame
            except queue.Empty:
                frame = last_frame

            if frame is None:
                key = cv2.waitKey(10) & 0xFF
            else:
                # Scale raw frame to match grid height for side-by-side display
                scale = grid_h / frame.shape[0]
                raw_w = int(frame.shape[1] * scale)
                raw_resized = cv2.resize(frame, (raw_w, grid_h))
                cb_state["raw_panel_w"] = raw_w

                if mapper.calibrating:
                    left_panel = _draw_calibration_overlay(raw_resized, mapper)
                    right_panel = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
                    cv2.putText(
                        right_panel, "Calibrating...",
                        (10, grid_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 1,
                    )
                elif mapper.is_calibrated:
                    warped = mapper.warp(frame)
                    coverage = detector.detect(warped)
                    grid_state.update(coverage)
                    right_panel = _draw_grid_overlay(warped, grid_state, sequencer, cell_size)
                    left_panel = raw_resized.copy()
                else:
                    left_panel = raw_resized.copy()
                    cv2.putText(
                        left_panel, "Press 'c' to calibrate",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                    )
                    right_panel = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

                _draw_hud(left_panel, sequencer)
                combined = np.hstack([left_panel, right_panel])
                cv2.imshow(WINDOW_NAME, combined)
                key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c"):
                mapper.start_calibration()
                print("Calibration started. Click top-left corner first.")
            elif key in (ord("+"), ord("=")):
                sequencer.bpm = min(sequencer.bpm + 5, 300)
                print(f"Tempo: {sequencer.bpm:.0f} BPM")
            elif key == ord("-"):
                sequencer.bpm = max(sequencer.bpm - 5, 20)
                print(f"Tempo: {sequencer.bpm:.0f} BPM")
            elif key == ord(" "):
                sequencer.paused = not sequencer.paused
                print("Paused." if sequencer.paused else "Playing.")
            elif key == ord("r"):
                sequencer.current_col = 0
                print("Beat reset to column 0.")

    finally:
        sequencer.stop()
        cam_thread.stop()
        camera.release()
        cv2.destroyAllWindows()
        pygame.quit()
        print("Goodbye.")


if __name__ == "__main__":
    run()
