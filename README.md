# Beat Maker

A camera-based hardware step sequencer. Place colored tiles on a physical grid, point a USB camera at it, and the app detects tile positions in real time and triggers configured sounds — turning the grid into a playable beat machine.

## How It Works

The physical grid acts as an 8-beat × 6-track step sequencer. Each column is a beat, each row is an instrument (kick, snare, hi-hat, clap, tom, cymbal). Placing a tile in a cell activates that sound on that beat. A profile selector strip at the top of the grid lets you switch between drum kits by placing a tile in the corresponding slot.

Detection uses HSV saturation thresholding (color-agnostic) with per-cell background calibration and temporal debouncing for reliable, flicker-free detection under varying lighting.

> Currently only tested on Ubuntu.

## Requirements

- Python 3.11+
- USB camera (V4L2 on Linux)
- Physical grid + colored tiles

**Tips:**
- Use bright-colored tiles in low-light environments and dark-colored tiles under bright lighting for more reliable detection.
- Place WAV or MP3 sample files in `assets/sounds/` and reference them in `config.yaml` under each track's `file` key. [99sounds.org](https://99sounds.org/) is a good free source for drum samples.

```bash
pip install -r requirements.txt
```

## Running

```bash
python -m src.main
```

Run from the project root. On first run, calibrate the camera to the grid corners.

## Setup

1. **Calibrate**: press `c`, then click the 4 grid corners (top-left → top-right → bottom-right → bottom-left).
2. **Set background**: remove all tiles, press `b` to capture the ambient lighting baseline.
3. Place tiles and hit `space` to start the sequencer.

## Controls

| Key | Action |
|-----|--------|
| `c` | Calibrate (click 4 corners) |
| `b` | Capture background baseline |
| `space` | Pause / play |
| `+` / `-` | Tempo ±5 BPM |
| `r` | Reset to beat 1 |
| `q` | Quit |

## Configuration

Edit `config.yaml` to change grid dimensions, tempo, camera ID, detection thresholds, and sound profiles. For per-machine overrides (e.g. a different camera index), create `config.local.yaml` — it deep-merges over the defaults and is gitignored.

```yaml
# config.local.yaml example
camera_id: 0
tempo_bpm: 120.0
```

Sound files (WAV/MP3) go in `assets/sounds/`. If a file is missing, the app synthesizes a fallback sound.
