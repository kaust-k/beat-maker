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
from .detector import (
    EMPTY, HORIZONTAL, VERTICAL,
    ColorTileDetector, GridState, OrientationTileDetector,
)
from .grid import CORNER_ORDER, GridMapper
from .sequencer import BeatSequencer

CONFIG_PATH  = "config.yaml"
CALIB_PATH   = "calibration.json"
WINDOW_NAME  = "Tembo Beat Maker"

# Per-row track colours for color-detector mode (BGR)
TRACK_COLORS_BGR = [
    (40,  40,  220),   # kick:   red
    (200, 80,  40),    # snare:  blue
    (40,  180, 40),    # hihat:  green
    (40,  220, 220),   # clap:   yellow
    (40,  140, 220),   # tom:    orange
    (200, 40,  200),   # cymbal: purple
]
HORIZONTAL_COLOR_BGR = (40, 200, 40)   # orientation mode: green for H tiles
VERTICAL_COLOR_BGR   = (200, 40, 200)  # orientation mode: purple for V tiles


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _draw_grid_overlay(
    warped: np.ndarray,
    grid_state: GridState,
    sequencer: BeatSequencer,
    cell_size: int,
    track_names: list[str],
    detector_mode: str,
) -> np.ndarray:
    overlay = warped.copy()
    rows, cols = grid_state.rows, grid_state.cols
    snap = grid_state.snapshot()
    cs = cell_size

    # Semi-transparent tile fills
    cell_layer = warped.copy()
    for r in range(rows):
        for c in range(cols):
            orientation = snap[r, c]
            if orientation == EMPTY:
                continue
            if detector_mode == "color":
                color = TRACK_COLORS_BGR[r % len(TRACK_COLORS_BGR)]
            elif orientation == HORIZONTAL:
                color = HORIZONTAL_COLOR_BGR
            else:
                color = VERTICAL_COLOR_BGR
            cv2.rectangle(
                cell_layer,
                (c * cs, r * cs), ((c + 1) * cs, (r + 1) * cs),
                color, -1,
            )
    cv2.addWeighted(cell_layer, 0.5, overlay, 0.5, 0, overlay)

    # Current beat column highlight (yellow, translucent)
    beat_col = sequencer.current_col
    col_layer = overlay.copy()
    cv2.rectangle(
        col_layer,
        (beat_col * cs, 0), ((beat_col + 1) * cs, rows * cs),
        (0, 255, 255), -1,
    )
    cv2.addWeighted(col_layer, 0.35, overlay, 0.65, 0, overlay)

    # Grid lines
    for c in range(cols + 1):
        cv2.line(overlay, (c * cs, 0), (c * cs, rows * cs), (80, 80, 80), 1)
    for r in range(rows + 1):
        cv2.line(overlay, (0, r * cs), (cols * cs, r * cs), (80, 80, 80), 1)

    # Row track labels
    for r, name in enumerate(track_names):
        cv2.putText(
            overlay, name[:4], (4, r * cs + cs // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1,
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
    cv2.putText(
        out, f"Click: {mapper.next_corner_name()}  ({len(mapper.click_buffer)}/4)",
        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
    )
    return out


def _draw_hud(frame: np.ndarray, sequencer: BeatSequencer, det_mode: str) -> None:
    h = frame.shape[0]
    if sequencer.paused:
        status = "PAUSED"
        color  = (100, 100, 255)
    else:
        status = f"{sequencer.bpm:.0f} BPM"
        color  = (255, 255, 0)
    cv2.putText(frame, f"[{det_mode}]  {status}",
                (10, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, "q:quit  c:calibrate  +/-:tempo  space:pause  r:reset",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)


def run() -> None:
    cfg_path = Path(CONFIG_PATH)
    if not cfg_path.exists():
        print(f"ERROR: {CONFIG_PATH} not found. Run from the project root.")
        sys.exit(1)

    cfg = _load_config(CONFIG_PATH)
    rows      = cfg["grid"]["rows"]
    cols      = cfg["grid"]["cols"]
    cell_size = cfg["display"]["cell_size"]
    tracks    = cfg["sounds"]["tracks"]
    det_mode  = cfg.get("detector", "orientation")
    det_cfg   = cfg["detection"]
    track_names = [t["name"] for t in tracks]

    grid_h = rows * cell_size
    grid_w = cols * cell_size

    # Audio
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    pygame.mixer.set_num_channels(16)
    audio = AudioPlayer(tracks, cfg["sounds"]["dir"])
    audio.load_all()
    print(f"Loaded sounds for {len(tracks)} tracks ({det_mode} detector mode).")

    # Grid state and detector
    grid_state = GridState(
        rows, cols,
        tile_threshold=det_cfg["tile_threshold"],
        empty_threshold=det_cfg["empty_threshold"],
    )
    if det_mode == "color":
        detector = ColorTileDetector(
            tracks, cfg["colors"], rows, cols, cell_size,
            tile_threshold=det_cfg["tile_threshold"],
        )
    else:
        detector = OrientationTileDetector(
            rows, cols, cell_size,
            sat_threshold=det_cfg.get("sat_threshold", 40),
            min_area_ratio=det_cfg.get("min_area_ratio", 0.10),
            h_ratio_threshold=det_cfg.get("h_ratio_threshold", 1.25),
            v_ratio_threshold=det_cfg.get("v_ratio_threshold", 0.80),
        )

    # Camera
    camera = LinuxCamera(cfg["camera_id"])
    if not camera.open():
        print(f"ERROR: Cannot open camera {cfg['camera_id']}.")
        pygame.quit()
        sys.exit(1)
    cam_thread = CameraThread(camera)
    cam_thread.start()

    # Grid mapper / calibration
    mapper = GridMapper(rows, cols, cell_size, CALIB_PATH)
    if mapper.load():
        print("Calibration loaded from calibration.json.")
    else:
        print("No calibration found. Press 'c' to calibrate.")

    # Sequencer
    sequencer = BeatSequencer(grid_state, audio, cols, cfg["tempo_bpm"])
    sequencer.start()

    # OpenCV window + mouse callback
    cv2.namedWindow(WINDOW_NAME)
    cb_state = {"mapper": mapper, "raw_panel_w": 9999}

    def mouse_callback(event, x, y, flags, param):
        m: GridMapper = param["mapper"]
        if event == cv2.EVENT_LBUTTONDOWN and m.calibrating:
            if x < param["raw_panel_w"]:
                done = m.add_click(x, y)
                if done:
                    print("Calibration complete and saved.")

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, cb_state)

    last_frame: np.ndarray | None = None

    try:
        while True:
            try:
                frame = cam_thread.frame_queue.get_nowait()
                last_frame = frame
            except queue.Empty:
                frame = last_frame

            if frame is None:
                key = cv2.waitKey(10) & 0xFF
            else:
                scale = grid_h / frame.shape[0]
                raw_w = int(frame.shape[1] * scale)
                raw_resized = cv2.resize(frame, (raw_w, grid_h))
                cb_state["raw_panel_w"] = raw_w

                if mapper.calibrating:
                    left  = _draw_calibration_overlay(raw_resized, mapper)
                    right = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
                    cv2.putText(right, "Calibrating...",
                                (10, grid_h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 1)
                elif mapper.is_calibrated:
                    warped = mapper.warp(frame)
                    orientations, areas = detector.detect(warped)
                    grid_state.update(orientations, areas)
                    right = _draw_grid_overlay(
                        warped, grid_state, sequencer, cell_size,
                        track_names, det_mode,
                    )
                    left = raw_resized.copy()
                else:
                    left = raw_resized.copy()
                    cv2.putText(left, "Press 'c' to calibrate",
                                (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 255), 2)
                    right = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

                _draw_hud(left, sequencer, det_mode)
                cv2.imshow(WINDOW_NAME, np.hstack([left, right]))
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
                print("Beat reset.")

    finally:
        sequencer.stop()
        cam_thread.stop()
        camera.release()
        cv2.destroyAllWindows()
        pygame.quit()
        print("Goodbye.")


if __name__ == "__main__":
    run()
