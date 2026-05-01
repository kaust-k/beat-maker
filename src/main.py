import base64
import queue
import sys
import threading
import warnings
from pathlib import Path

# pygame prints an AVX2 RuntimeWarning on systems where it wasn't compiled with AVX2
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*avx2.*")

import cv2
import numpy as np
import pygame
import yaml
import flet as ft

from .audio import AudioPlayer
from .camera import CameraThread, LinuxCamera
from .detector import (
    EMPTY, HORIZONTAL, VERTICAL,
    ColorTileDetector, GridState, OrientationTileDetector,
)
from .grid import CORNER_ORDER, GridMapper
from .sequencer import BeatSequencer

CONFIG_PATH = "config.yaml"
CALIB_PATH  = "calibration.json"

# Per-row track colours for color-detector mode (BGR)
TRACK_COLORS_BGR = [
    (40,  40,  220),  # kick:   red
    (200, 80,  40),   # snare:  blue
    (40,  180, 40),   # hihat:  green
    (40,  220, 220),  # clap:   yellow
    (40,  140, 220),  # tom:    orange
    (200, 40,  200),  # cymbal: purple
]
HORIZONTAL_COLOR_BGR = (40, 200, 40)   # orientation mode: green for H
VERTICAL_COLOR_BGR   = (200, 40, 200)  # orientation mode: purple for V


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _encode_jpeg_b64(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


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


def _processing_loop(
    cam_thread: CameraThread,
    mapper: GridMapper,
    detector,
    grid_state: GridState,
    sequencer: BeatSequencer,
    cell_size: int,
    grid_h: int,
    grid_w: int,
    img_raw: ft.Image,
    img_overlay: ft.Image,
    page: ft.Page,
    stop_event: threading.Event,
    current_scale: list,  # [float] mutable container for calibration coord mapping
    detector_mode: str,
    track_names: list[str],
) -> None:
    last_frame = None

    while not stop_event.is_set():
        try:
            frame = cam_thread.frame_queue.get(timeout=0.033)
            last_frame = frame
        except queue.Empty:
            frame = last_frame

        if frame is None:
            continue

        # Scale raw frame to grid height for display
        scale = grid_h / frame.shape[0]
        current_scale[0] = scale
        raw_w_px = int(frame.shape[1] * scale)
        raw_resized = cv2.resize(frame, (raw_w_px, grid_h))

        if mapper.calibrating:
            left = _draw_calibration_overlay(raw_resized, mapper)
            right = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            cv2.putText(right, "Calibrating...", (10, grid_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        elif mapper.is_calibrated:
            warped = mapper.warp(frame)
            orientations, areas = detector.detect(warped)
            grid_state.update(orientations, areas)
            right = _draw_grid_overlay(
                warped, grid_state, sequencer, cell_size,
                track_names, detector_mode,
            )
            left = raw_resized.copy()
        else:
            left = raw_resized.copy()
            cv2.putText(left, "Click 'Calibrate' to start",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            right = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        img_raw.src = _encode_jpeg_b64(left)
        img_overlay.src = _encode_jpeg_b64(right)
        try:
            page.update()
        except Exception:
            break  # page closed


def main(page: ft.Page) -> None:
    cfg_path = Path(CONFIG_PATH)
    if not cfg_path.exists():
        print(f"ERROR: {CONFIG_PATH} not found. Run from the project root.")
        return

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
        return
    cam_thread = CameraThread(camera)
    cam_thread.start()

    # Grid mapper / calibration
    mapper = GridMapper(rows, cols, cell_size, CALIB_PATH)
    if mapper.load():
        print("Calibration loaded.")
    else:
        print("No calibration. Click 'Calibrate'.")

    # Sequencer
    sequencer = BeatSequencer(grid_state, audio, cols, cfg["tempo_bpm"])
    sequencer.start()

    # Shared state for processing thread ↔ calibration click handler
    current_scale: list = [1.0]
    stop_event = threading.Event()

    # --- Flet UI controls ---
    page.title = f"Tembo Beat Maker [{det_mode} mode]"
    page.padding = 10
    page.spacing = 8

    img_raw     = ft.Image(src="", width=int(grid_h * 4 / 3), height=grid_h,
                           fit=ft.BoxFit.FILL, border_radius=4)
    img_overlay = ft.Image(src="", width=grid_w, height=grid_h,
                           fit=ft.BoxFit.FILL, border_radius=4)

    # Text nodes for buttons whose labels change at runtime
    calibrate_lbl  = ft.Text("Calibrate")
    play_pause_lbl = ft.Text("Pause")

    calibrate_btn = ft.FilledButton(
        content=calibrate_lbl, icon=ft.Icons.CENTER_FOCUS_STRONG,
    )

    def _on_camera_tap(e: ft.TapEvent) -> None:
        if not mapper.calibrating or e.local_position is None:
            return
        # Map rendered image coords → original camera frame coords
        camera_x = int(e.local_position.x / current_scale[0])
        camera_y = int(e.local_position.y / current_scale[0])
        done = mapper.add_click(camera_x, camera_y)
        if done:
            calibrate_lbl.value = "Calibrate"
            calibrate_btn.icon = ft.Icons.CENTER_FOCUS_STRONG
            page.update()

    def _on_calibrate(_) -> None:
        mapper.start_calibration()
        calibrate_lbl.value = "Calibrating… (click 4 corners)"
        calibrate_btn.icon = ft.Icons.TOUCH_APP
        page.update()

    calibrate_btn.on_click = _on_calibrate

    # File picker — async: pick_files() returns selected files directly
    file_picker = ft.FilePicker()
    page.overlay.append(file_picker)

    file_labels: list[dict] = [{"h": None, "v": None} for _ in tracks]

    async def _pick_sound(row: int, slot: str) -> None:
        files = await file_picker.pick_files(
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["wav", "mp3"],
            allow_multiple=False,
        )
        if not files or not files[0].path:
            return
        path = files[0].path
        try:
            audio.load_sounds_for_row(
                row,
                path_h=path if slot == "h" else None,
                path_v=path if slot == "v" else None,
            )
            lbl: ft.Text = file_labels[row][slot]
            if lbl:
                lbl.value = Path(path).name
                page.update()
        except RuntimeError as exc:
            page.snack_bar = ft.SnackBar(ft.Text(str(exc)), open=True)
            page.update()

    def _make_pick_handler(row: int, slot: str):
        async def handler(_):
            await _pick_sound(row, slot)
        return handler

    # Build per-track rows
    track_rows = []
    for r, track in enumerate(tracks):
        lbl_h = ft.Text("(synth)", size=11, color=ft.Colors.GREY_400, width=130,
                         overflow=ft.TextOverflow.ELLIPSIS)
        lbl_v = ft.Text("(accent)", size=11, color=ft.Colors.GREY_400, width=130,
                         overflow=ft.TextOverflow.ELLIPSIS)
        file_labels[r]["h"] = lbl_h
        file_labels[r]["v"] = lbl_v

        track_color = (
            ft.Colors.GREY_700 if det_mode == "orientation"
            else ["red", "blue", "green", "yellow", "orange", "purple"][r % 6]
        )
        name_chip = ft.Container(
            content=ft.Text(track["name"].upper(), size=11, weight=ft.FontWeight.BOLD),
            bgcolor=track_color, border_radius=4,
            padding=ft.padding.symmetric(horizontal=6, vertical=2),
            width=70,
        )

        track_rows.append(
            ft.Row([
                name_chip,
                ft.FilledButton(
                    content=ft.Text("H ♪"), on_click=_make_pick_handler(r, "h"),
                    color=ft.Colors.GREEN_300, height=30,
                ),
                lbl_h,
                ft.FilledButton(
                    content=ft.Text("V ♪"), on_click=_make_pick_handler(r, "v"),
                    color=ft.Colors.PURPLE_300, height=30,
                ),
                lbl_v,
            ], spacing=6, height=36)
        )

    # Tempo slider
    tempo_slider = ft.Slider(
        min=20, max=300, value=cfg["tempo_bpm"],
        divisions=280, label="{value} BPM",
        active_color=ft.Colors.AMBER,
        expand=True,
    )
    bpm_label = ft.Text(f"{cfg['tempo_bpm']:.0f} BPM", width=80)

    def _on_tempo_change(e):
        sequencer.bpm = float(e.control.value)
        bpm_label.value = f"{sequencer.bpm:.0f} BPM"
        page.update()
    tempo_slider.on_change = _on_tempo_change

    # Playback controls
    def _toggle_pause(_=None):
        sequencer.paused = not sequencer.paused
        play_pause_lbl.value = "Play" if sequencer.paused else "Pause"
        play_pause_btn.icon = ft.Icons.PLAY_ARROW if sequencer.paused else ft.Icons.PAUSE
        page.update()

    play_pause_btn = ft.FilledButton(
        content=play_pause_lbl, icon=ft.Icons.PAUSE, on_click=_toggle_pause,
    )
    reset_btn = ft.FilledButton(
        content=ft.Text("Reset"), icon=ft.Icons.SKIP_PREVIOUS,
        on_click=lambda _: setattr(sequencer, "current_col", 0),
    )

    mode_badge = ft.Text(
        f"Mode: {det_mode}", size=11,
        color=ft.Colors.GREEN_300 if det_mode == "orientation" else ft.Colors.ORANGE_300,
    )

    # Page layout
    page.add(
        ft.Row([
            ft.GestureDetector(
                content=img_raw,
                on_tap_down=_on_camera_tap,
            ),
            img_overlay,
        ], spacing=8),
        ft.Divider(height=6),
        ft.Column(track_rows, spacing=2),
        ft.Divider(height=6),
        ft.Row([
            ft.Text("Tempo:", size=13),
            tempo_slider,
            bpm_label,
        ], spacing=10),
        ft.Row([
            play_pause_btn,
            reset_btn,
            calibrate_btn,
            mode_badge,
        ], spacing=10),
    )

    # Cleanup on window close
    def _on_close(_):
        stop_event.set()
        sequencer.stop()
        cam_thread.stop()
        camera.release()
        pygame.quit()

    page.on_close = _on_close

    # Start processing thread
    proc_thread = threading.Thread(
        target=_processing_loop,
        args=(
            cam_thread, mapper, detector, grid_state, sequencer,
            cell_size, grid_h, grid_w,
            img_raw, img_overlay, page,
            stop_event, current_scale, det_mode, track_names,
        ),
        daemon=True,
    )
    proc_thread.start()


if __name__ == "__main__":
    ft.app(target=main)
