import wave
from pathlib import Path

import numpy as np
import pygame

from .detector import EMPTY, HORIZONTAL, VERTICAL

SR = 44100


def _to_stereo_int16(mono: np.ndarray) -> np.ndarray:
    clipped = np.clip(mono, -1.0, 1.0)
    i16 = (clipped * 32767).astype(np.int16)
    return np.column_stack([i16, i16])


def _save_wav(path: Path, stereo_i16: np.ndarray) -> None:
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(stereo_i16.tobytes())


def _load_sound_file(path: Path) -> "pygame.mixer.Sound":
    """Load a WAV or MP3 file into a pygame Sound. Uses miniaudio for MP3."""
    if path.suffix.lower() == ".mp3":
        try:
            import miniaudio  # type: ignore
        except ImportError:
            raise RuntimeError(
                "miniaudio is not installed. Install it with: pip install miniaudio"
            )
        decoded = miniaudio.decode_file(
            str(path),
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=2,
            sample_rate=SR,
        )
        arr = np.frombuffer(decoded.samples, dtype=np.int16).reshape(-1, 2)
        return pygame.sndarray.make_sound(arr)
    return pygame.mixer.Sound(str(path))


# --- Synthesis functions: each returns float64 mono in [-1, 1] ---

def _synth_kick(sr: int = SR) -> np.ndarray:
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    freq_env = 60.0 * np.exp(-t * 20.0)
    phase = np.cumsum(2 * np.pi * freq_env / sr)
    return np.sin(phase) * np.exp(-t * 8.0)


def _synth_snare(sr: int = SR) -> np.ndarray:
    t = np.linspace(0, 0.3, int(sr * 0.3), endpoint=False)
    rng = np.random.default_rng(42)
    noise = rng.uniform(-1.0, 1.0, len(t))
    tone = np.sin(2 * np.pi * 200.0 * t)
    return (0.7 * noise + 0.3 * tone) * np.exp(-t * 15.0)


def _synth_hihat(sr: int = SR) -> np.ndarray:
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    rng = np.random.default_rng(7)
    return rng.uniform(-1.0, 1.0, len(t)) * np.exp(-t * 30.0)


def _synth_clap(sr: int = SR) -> np.ndarray:
    total = int(sr * 0.2)
    result = np.zeros(total, dtype=float)
    rng = np.random.default_rng(13)
    for offset_ms in [0, 10, 20]:
        start = int(offset_ms / 1000 * sr)
        burst_len = min(int(0.05 * sr), total - start)
        if burst_len > 0:
            t_burst = np.linspace(0, 5, burst_len)
            result[start : start + burst_len] += (
                rng.uniform(-1.0, 1.0, burst_len) * np.exp(-t_burst)
            )
    t = np.linspace(0, 0.2, total, endpoint=False)
    result *= np.exp(-t * 10.0)
    peak = np.abs(result).max()
    return result / (peak + 1e-9)


def _synth_tom(sr: int = SR) -> np.ndarray:
    t = np.linspace(0, 0.4, int(sr * 0.4), endpoint=False)
    freq_env = 120.0 * np.exp(-t * 8.0)
    phase = np.cumsum(2 * np.pi * freq_env / sr)
    return np.sin(phase) * np.exp(-t * 6.0)


def _synth_cymbal(sr: int = SR) -> np.ndarray:
    t = np.linspace(0, 0.8, int(sr * 0.8), endpoint=False)
    rng = np.random.default_rng(99)
    noise = rng.uniform(-1.0, 1.0, len(t))
    mod = np.sin(2 * np.pi * 8000.0 * t)
    return noise * mod * np.exp(-t * 3.0)


_SYNTH_FUNCS = {
    "kick":   _synth_kick,
    "snare":  _synth_snare,
    "hihat":  _synth_hihat,
    "clap":   _synth_clap,
    "tom":    _synth_tom,
    "cymbal": _synth_cymbal,
}


class AudioPlayer:
    """
    Loads or synthesizes two sounds per track row:
      - H sound (file_h): primary hit, played when tile is HORIZONTAL
      - V sound (file_v): accent hit, played when tile is VERTICAL
                          Defaults to 85% volume variant of H if not specified.
    """

    def __init__(self, tracks: list[dict], sound_dir: str):
        self._tracks = tracks
        self._sound_dir = Path(sound_dir)
        self._h_sounds: list[pygame.mixer.Sound | None] = []
        self._v_sounds: list[pygame.mixer.Sound | None] = []

    def load_all(self) -> None:
        self._sound_dir.mkdir(parents=True, exist_ok=True)
        self._h_sounds = []
        self._v_sounds = []
        for track in self._tracks:
            h, v = self._load_track_pair(track)
            self._h_sounds.append(h)
            self._v_sounds.append(v)

    def _load_track_pair(
        self, track: dict
    ) -> tuple["pygame.mixer.Sound", "pygame.mixer.Sound"]:
        name = track["name"]

        # --- H sound ---
        file_h = track.get("file_h")
        if file_h:
            path_h = self._sound_dir / file_h
            if path_h.exists():
                h_sound = _load_sound_file(path_h)
            else:
                h_sound = self._synthesize(name)
        else:
            h_sound = self._synthesize(name)

        # --- V sound ---
        file_v = track.get("file_v")
        if file_v:
            path_v = self._sound_dir / file_v
            if path_v.exists():
                v_sound = _load_sound_file(path_v)
            else:
                v_sound = self._make_accent(h_sound)
        else:
            v_sound = self._make_accent(h_sound)

        return h_sound, v_sound

    def _synthesize(self, name: str) -> "pygame.mixer.Sound":
        cached = self._sound_dir / f"{name}.wav"
        if not cached.exists():
            synth_fn = _SYNTH_FUNCS.get(name, _synth_kick)
            _save_wav(cached, _to_stereo_int16(synth_fn()))
        return pygame.mixer.Sound(str(cached))

    def _make_accent(self, h_sound: "pygame.mixer.Sound") -> "pygame.mixer.Sound":
        """85% volume variant of h_sound for the vertical/accent hit."""
        arr = pygame.sndarray.array(h_sound)
        arr_accent = (arr * 0.85).astype(arr.dtype)
        return pygame.sndarray.make_sound(arr_accent)

    def play(self, row: int, orientation: int) -> None:
        """Play H or V sound for the given row based on tile orientation."""
        if orientation == HORIZONTAL:
            sounds = self._h_sounds
        elif orientation == VERTICAL:
            sounds = self._v_sounds
        else:
            return
        if 0 <= row < len(sounds) and sounds[row] is not None:
            sounds[row].play()

    def load_sounds_for_row(
        self, row: int, path_h: str | None, path_v: str | None
    ) -> None:
        """
        Replace H and/or V sounds for a row at runtime (called from file picker).
        Thread-safe under CPython GIL for single list-index assignment.
        """
        if not (0 <= row < len(self._tracks)):
            return
        track = self._tracks[row]
        if path_h is not None:
            p = Path(path_h)
            track["file_h"] = path_h
            self._h_sounds[row] = _load_sound_file(p)
            # Regenerate accent from new h sound if v is not separately specified
            if not track.get("file_v"):
                self._v_sounds[row] = self._make_accent(self._h_sounds[row])
        if path_v is not None:
            p = Path(path_v)
            track["file_v"] = path_v
            self._v_sounds[row] = _load_sound_file(p)

    def stop_all(self) -> None:
        pygame.mixer.stop()
