import wave
from pathlib import Path

import numpy as np
import pygame

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
    "kick": _synth_kick,
    "snare": _synth_snare,
    "hihat": _synth_hihat,
    "clap": _synth_clap,
    "tom": _synth_tom,
    "cymbal": _synth_cymbal,
}


class AudioPlayer:
    def __init__(self, tracks: list[dict], sound_dir: str):
        self._tracks = tracks
        self._sound_dir = Path(sound_dir)
        self._sounds: list[pygame.mixer.Sound] = []

    def load_all(self) -> None:
        self._sound_dir.mkdir(parents=True, exist_ok=True)
        self._sounds = [self._load_track(t) for t in self._tracks]

    def _load_track(self, track: dict) -> "pygame.mixer.Sound":
        custom = track.get("file")
        if custom:
            path = self._sound_dir / custom
            if path.exists():
                return pygame.mixer.Sound(str(path))

        name = track["name"]
        cached = self._sound_dir / f"{name}.wav"
        if not cached.exists():
            synth_fn = _SYNTH_FUNCS.get(name, _synth_kick)
            stereo = _to_stereo_int16(synth_fn())
            _save_wav(cached, stereo)
        return pygame.mixer.Sound(str(cached))

    def play(self, row: int) -> None:
        if 0 <= row < len(self._sounds):
            self._sounds[row].play()

    def stop_all(self) -> None:
        pygame.mixer.stop()
