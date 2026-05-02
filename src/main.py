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
    ColorTileDetector, GridState, OrientationTileDetector, SelectorDetector,
)
from .grid import CORNER_ORDER, GridMapper
from .profiles import ProfileManager
from .sequencer import BeatSequencer

CONFIG_PATH = "config.yaml"
CALIB_PATH  = "calibration.json"
WINDOW_NAME = "Tembo Beat Maker"

# Per-profile and per-row overlay colors (BGR)
PROFILE_COLORS_BGR = [
    (40,  40,  220),
    (40,  220, 40),
    (220, 40,  40),
    (40,  220, 220),
    (220, 220, 40),
    (200, 40,  200),
]
TRACK_COLORS_BGR = [
    (40,  40,  220),
    (200, 80,  40),
    (40,  180, 40),
    (40,  220, 220),
    (40,  140, 220),
    (200, 40,  200),
]
HORIZONTAL_COLOR_BGR = (40, 200, 40)
VERTICAL_COLOR_BGR   = (200, 40, 200)


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_color_detector(
    tracks: list[dict],
    color_defs: dict,
    rows: int,
    cols: int,
    cell_size: int,
    tile_threshold: float,
) -> ColorTileDetector:
    return ColorTileDetector(tracks, color_defs, rows, cols, cell_size,
                             tile_threshold=tile_threshold)


def _draw_selector_overlay(
    warped: np.ndarray,
    profile_manager: ProfileManager,
    profiles: list[dict],
    cols: int,
    cell_size: int,
) -> np.ndarray:
    """Draw profile selector strip in the top row of the warped image."""
    overlay = warped.copy()
    cs = cell_size

    cell_layer = warped.copy()
    for i, profile in enumerate(profiles):
        if i >= cols:
            break
        raw_color = profile.get("color_bgr", PROFILE_COLORS_BGR[i % len(PROFILE_COLORS_BGR)])
        color = tuple(int(c) for c in raw_color)
        cv2.rectangle(cell_layer, (i * cs, 0), ((i + 1) * cs, cs), color, -1)
    cv2.addWeighted(cell_layer, 0.45, overlay, 0.55, 0, overlay)

    for i, profile in enumerate(profiles):
        if i >= cols:
            break
        x0, x1, y0, y1 = i * cs, (i + 1) * cs, 0, cs

        if i == profile_manager.active_index:
            border_color = (255, 255, 255)
            thickness = 2
        elif i == profile_manager.pending_index:
            border_color = (0, 255, 255)
            thickness = 2
        else:
            border_color = (80, 80, 80)
            thickness = 1

        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), border_color, thickness)
        label = profile["name"][:7]
        cv2.putText(overlay, label, (x0 + 4, y0 + cs // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (230, 230, 230), 1)

    # Label the strip area beyond the profile slots
    label_x = min(len(profiles), cols) * cs + 4
    if label_x < cols * cs:
        cv2.putText(overlay, "PROFILE", (label_x, cs // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (130, 130, 130), 1)

    return overlay


def _draw_grid_overlay(
    warped: np.ndarray,
    grid_state: GridState,
    sequencer: BeatSequencer,
    cell_size: int,
    track_names: list[str],
    detector_mode: str,
    y_offset: int = 0,
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
                (c * cs, r * cs + y_offset),
                ((c + 1) * cs, (r + 1) * cs + y_offset),
                color, -1,
            )
    cv2.addWeighted(cell_layer, 0.5, overlay, 0.5, 0, overlay)

    # Current beat column highlight (yellow, translucent)
    beat_col = sequencer.current_col
    col_layer = overlay.copy()
    cv2.rectangle(
        col_layer,
        (beat_col * cs, y_offset),
        ((beat_col + 1) * cs, rows * cs + y_offset),
        (0, 255, 255), -1,
    )
    cv2.addWeighted(col_layer, 0.35, overlay, 0.65, 0, overlay)

    # Grid lines
    for c in range(cols + 1):
        cv2.line(overlay,
                 (c * cs, y_offset), (c * cs, rows * cs + y_offset),
                 (80, 80, 80), 1)
    for r in range(rows + 1):
        cv2.line(overlay,
                 (0, r * cs + y_offset), (cols * cs, r * cs + y_offset),
                 (80, 80, 80), 1)

    # Row track labels
    for r, name in enumerate(track_names):
        cv2.putText(
            overlay, name[:4], (4, r * cs + y_offset + cs // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1,
        )

    return overlay


def _draw_calibration_overlay(
    frame: np.ndarray, mapper: GridMapper, display_scale: float = 1.0
) -> np.ndarray:
    out = frame.copy()
    for i, (cx, cy) in enumerate(mapper.click_buffer):
        # click_buffer stores original-frame coords; scale to display frame for drawing
        dcx, dcy = int(cx * display_scale), int(cy * display_scale)
        cv2.drawMarker(out, (dcx, dcy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(
            out, CORNER_ORDER[i], (dcx + 8, dcy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )
    cv2.putText(
        out, f"Click: {mapper.next_corner_name()}  ({len(mapper.click_buffer)}/4)",
        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
    )
    return out


def _draw_hud(
    frame: np.ndarray,
    sequencer: BeatSequencer,
    det_mode: str,
    profile_name: str,
    pending_name: str | None,
    bg_ok: bool = False,
) -> None:
    h = frame.shape[0]
    if sequencer.paused:
        status = "PAUSED"
        color  = (100, 100, 255)
    else:
        status = f"{sequencer.bpm:.0f} BPM"
        color  = (255, 255, 0)
    pending_str = f" →{pending_name}" if pending_name else ""
    bg_str = "BG:ok" if bg_ok else "BG:--"
    cv2.putText(frame, f"[{det_mode}] {profile_name}{pending_str}  {status}  {bg_str}",
                (10, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, "q:quit  c:calibrate  b:background  +/-:tempo  space:pause  r:reset",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)


def run() -> None:
    cfg_path = Path(CONFIG_PATH)
    if not cfg_path.exists():
        print(f"ERROR: {CONFIG_PATH} not found. Run from the project root.")
        sys.exit(1)

    cfg = _load_config(CONFIG_PATH)
    rows         = cfg["grid"]["rows"]
    cols         = cfg["grid"]["cols"]
    cell_size    = cfg["display"]["cell_size"]
    det_mode     = cfg.get("detector", "orientation")
    det_cfg      = cfg["detection"]
    profiles     = cfg["profiles"]
    selector_rows = cfg.get("selector", {}).get("rows", 1)
    sound_dir    = cfg["sounds"]["dir"]

    grid_h = (selector_rows + rows) * cell_size  # total warped height
    grid_w = cols * cell_size
    y_off  = selector_rows * cell_size            # beat grid y-offset inside warped frame

    # Audio — preload all profiles at startup
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    pygame.mixer.set_num_channels(16)
    snd_cfg = cfg["sounds"]
    audio = AudioPlayer(
        sound_dir,
        normalize=snd_cfg.get("normalize", True),
        target_rms_db=snd_cfg.get("target_rms_db", -18.0),
    )
    audio.preload_profiles(profiles)
    n_tracks = len(profiles[0]["tracks"])
    print(f"Loaded {len(profiles)} profile(s) ({n_tracks} tracks each, {det_mode} detector).")

    # Profile manager
    profile_manager = ProfileManager(profiles)

    # Grid state and beat detector (rebuilt on profile switch in color mode)
    grid_state = GridState(
        rows, cols,
        tile_threshold=det_cfg["tile_threshold"],
        empty_threshold=det_cfg["empty_threshold"],
        min_present_frames=det_cfg.get("min_present_frames", 3),
        min_absent_frames=det_cfg.get("min_absent_frames", 2),
    )

    def _beat_baseline() -> np.ndarray | None:
        """Return the beat-grid rows of the background baseline, or None."""
        bs = mapper.baseline_s
        return bs[selector_rows:] if bs is not None else None

    def _selector_baseline() -> np.ndarray | None:
        bs = mapper.baseline_s
        return bs[:selector_rows] if bs is not None and selector_rows > 0 else None

    def _make_beat_detector(active_tracks: list[dict]):
        if det_mode == "color":
            return _build_color_detector(
                active_tracks, cfg["colors"], rows, cols, cell_size,
                tile_threshold=det_cfg["tile_threshold"],
            )
        return OrientationTileDetector(
            rows, cols, cell_size,
            sat_threshold=det_cfg.get("sat_threshold", 40),
            min_area_ratio=det_cfg.get("min_area_ratio", 0.10),
            h_ratio_threshold=det_cfg.get("h_ratio_threshold", 1.25),
            v_ratio_threshold=det_cfg.get("v_ratio_threshold", 0.80),
            baseline_s=_beat_baseline(),
        )

    def _make_selector_detector():
        return SelectorDetector(
            len(profiles), cols, cell_size,
            sat_threshold=det_cfg.get("sat_threshold", 40),
            min_area_ratio=det_cfg.get("min_area_ratio", 0.10),
            baseline_s=_selector_baseline(),
        )

    detector = _make_beat_detector(profiles[0]["tracks"])
    track_names = [t["name"] for t in profiles[0]["tracks"]]
    last_profile_idx = 0

    # Selector strip detector
    selector_detector = _make_selector_detector()

    # Camera
    camera = LinuxCamera(cfg["camera_id"])
    if not camera.open():
        print(f"ERROR: Cannot open camera {cfg['camera_id']}.")
        pygame.quit()
        sys.exit(1)
    cam_thread = CameraThread(camera)
    cam_thread.start()

    # Grid mapper / calibration
    mapper = GridMapper(rows, cols, cell_size, CALIB_PATH, selector_rows=selector_rows)
    if mapper.load():
        print("Calibration loaded from calibration.json.")
    else:
        print("No calibration found (or dimensions changed). Press 'c' to calibrate.")

    # Cycle-end callback: apply pending profile switch (runs in sequencer thread)
    def on_cycle_end() -> None:
        if profile_manager.apply_pending():
            audio.switch_profile(profile_manager.active_index)

    # Sequencer
    sequencer = BeatSequencer(
        grid_state, audio, cols, cfg["tempo_bpm"], on_cycle_end=on_cycle_end
    )
    sequencer.start()

    # OpenCV window + mouse callback
    cv2.namedWindow(WINDOW_NAME)
    cb_state = {"mapper": mapper, "raw_panel_w": 9999, "scale": 1.0}

    def mouse_callback(event, x, y, flags, param):
        m: GridMapper = param["mapper"]
        if event == cv2.EVENT_LBUTTONDOWN and m.calibrating:
            if x < param["raw_panel_w"]:
                # Clicks land on the scaled display frame; convert back to original-frame
                # coordinates so the warp matrix is correct for mapper.warp(original_frame).
                s = param["scale"]
                done = m.add_click(int(x / s), int(y / s))
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
                cb_state["scale"] = scale

                if mapper.calibrating:
                    left  = _draw_calibration_overlay(raw_resized, mapper, display_scale=scale)
                    right = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
                    cv2.putText(right, "Calibrating...",
                                (10, grid_h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 1)
                elif mapper.is_calibrated:
                    warped = mapper.warp(frame)

                    # Selector detection (always on top strip)
                    if selector_rows > 0:
                        sel_idx = selector_detector.detect(warped)
                        if sel_idx is not None:
                            profile_manager.request_switch(sel_idx)

                    # Beat grid detection (slice below selector strip)
                    beat_warped = warped[y_off:] if y_off > 0 else warped
                    orientations, areas = detector.detect(beat_warped)
                    grid_state.update(orientations, areas)

                    # Rebuild detector / names if profile switched
                    current_idx = profile_manager.active_index
                    if current_idx != last_profile_idx:
                        last_profile_idx = current_idx
                        track_names = [t["name"] for t in profiles[current_idx]["tracks"]]
                        detector = _make_beat_detector(profiles[current_idx]["tracks"])

                    right = _draw_grid_overlay(
                        warped, grid_state, sequencer, cell_size,
                        track_names, det_mode, y_offset=y_off,
                    )
                    if selector_rows > 0:
                        right = _draw_selector_overlay(
                            right, profile_manager, profiles, cols, cell_size,
                        )
                    left = raw_resized.copy()
                else:
                    left = raw_resized.copy()
                    cv2.putText(left, "Press 'c' to calibrate",
                                (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 255), 2)
                    right = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

                active_profile = profiles[profile_manager.active_index]
                pending_idx = profile_manager.pending_index
                pending_name = profiles[pending_idx]["name"] if pending_idx is not None else None
                _draw_hud(left, sequencer, det_mode, active_profile["name"], pending_name,
                          bg_ok=mapper.baseline_s is not None)
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
                sequencer._fire_next = 0
                print("Beat reset.")
            elif key == ord("b"):
                if mapper.is_calibrated:
                    if last_frame is not None:
                        mapper.capture_background(last_frame)
                        detector = _make_beat_detector(profiles[profile_manager.active_index]["tracks"])
                        selector_detector = _make_selector_detector()
                        print("Background captured. Detection is now lighting-adaptive.")
                    else:
                        print("No frame available yet.")
                else:
                    print("Calibrate first ('c'), then capture background ('b').")

    finally:
        sequencer.stop()
        cam_thread.stop()
        camera.release()
        cv2.destroyAllWindows()
        pygame.quit()
        print("Goodbye.")


if __name__ == "__main__":
    run()
