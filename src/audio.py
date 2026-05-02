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


def _normalize_rms(arr: np.ndarray, target_rms_db: float) -> np.ndarray:
    """
    Scale an int16 stereo array so its RMS hits target_rms_db dBFS.
    Gain is capped so the peak stays at or below -1 dBFS to avoid hard clipping.
    Returns int16.
    """
    f = arr.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(f ** 2)))
    if rms < 1e-9:
        return arr  # silent sample — leave untouched
    target_rms = 10.0 ** (target_rms_db / 20.0)
    gain = target_rms / rms
    # Cap gain so peak never exceeds -1 dBFS headroom
    peak = float(np.abs(f).max())
    headroom = 10.0 ** (-1.0 / 20.0)  # ≈ 0.891
    if peak * gain > headroom:
        gain = headroom / peak
    return (np.clip(f * gain, -1.0, 1.0) * 32767).astype(np.int16)


def _load_sound_file(
    path: Path,
    normalize: bool = False,
    target_rms_db: float = -18.0,
) -> "pygame.mixer.Sound":
    """
    Load any audio file (WAV or MP3) into a pygame Sound.
    Always decodes via miniaudio so that 24-bit / 32-bit / mono / non-44100 Hz files
    are converted to int16 stereo 44100 Hz before handing off to pygame.
    If normalize=True, RMS-normalizes the decoded array to target_rms_db dBFS.
    """
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
    if normalize:
        arr = _normalize_rms(arr, target_rms_db)
    return pygame.sndarray.make_sound(arr)


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
    Loads or synthesizes two sounds per track row across multiple profiles.

    Resolution order per track:
      H sound: file_h → file → synthesize(name)
      V sound: file_v → file (as accent source) → _make_accent(h_sound)

    All profiles are preloaded at startup via preload_profiles(); switching is O(1).
    """

    def __init__(self, sound_dir: str, normalize: bool = True, target_rms_db: float = -18.0):
        self._sound_dir = Path(sound_dir)
        self._normalize = normalize
        self._target_rms_db = target_rms_db
        self._h_sounds: list[pygame.mixer.Sound | None] = []
        self._v_sounds: list[pygame.mixer.Sound | None] = []
        self._profile_sounds: list[tuple[list, list]] = []
        self._profile_tracks: list[list[dict]] = []   # for per-track volume on switch
        self._channels: list[pygame.mixer.Channel] = []

    def preload_profiles(self, profiles: list[dict]) -> None:
        """Load sounds for every profile at startup. Activates profile 0."""
        self._sound_dir.mkdir(parents=True, exist_ok=True)
        self._profile_sounds = []
        self._profile_tracks = []
        for profile in profiles:
            h_list: list[pygame.mixer.Sound | None] = []
            v_list: list[pygame.mixer.Sound | None] = []
            for track in profile["tracks"]:
                h, v = self._load_track_pair(track)
                h_list.append(h)
                v_list.append(v)
            self._profile_sounds.append((h_list, v_list))
            self._profile_tracks.append(profile["tracks"])

        if self._profile_sounds:
            self._h_sounds, self._v_sounds = self._profile_sounds[0]

        # Allocate one dedicated channel per track row so each instrument is isolated.
        # channel.play() cuts any previous hit on that channel — no cross-track interference.
        n_tracks = max(len(p["tracks"]) for p in profiles)
        pygame.mixer.set_num_channels(max(pygame.mixer.get_num_channels(), n_tracks))
        self._channels = [pygame.mixer.Channel(i) for i in range(n_tracks)]
        for i, ch in enumerate(self._channels):
            if profiles and i < len(profiles[0]["tracks"]):
                ch.set_volume(float(profiles[0]["tracks"][i].get("volume", 1.0)))

    def switch_profile(self, idx: int) -> None:
        """O(1) profile switch — no I/O. Safe to call from the sequencer thread."""
        if 0 <= idx < len(self._profile_sounds):
            self._h_sounds, self._v_sounds = self._profile_sounds[idx]
            if idx < len(self._profile_tracks):
                for i, ch in enumerate(self._channels):
                    tracks = self._profile_tracks[idx]
                    if i < len(tracks):
                        ch.set_volume(float(tracks[i].get("volume", 1.0)))

    def _load_track_pair(
        self, track: dict
    ) -> tuple["pygame.mixer.Sound", "pygame.mixer.Sound"]:
        name = track["name"]
        file_h = track.get("file_h")
        file_v = track.get("file_v")
        file_base = track.get("file")

        # --- H sound: file_h → file → synthesize ---
        if file_h:
            path = self._sound_dir / file_h
            h_sound = self._load(path) if path.exists() else self._synthesize(name)
        elif file_base:
            path = self._sound_dir / file_base
            h_sound = self._load(path) if path.exists() else self._synthesize(name)
        else:
            h_sound = self._synthesize(name)

        # --- V sound: file_v → file (accent) → _make_accent(h_sound) ---
        if file_v:
            path = self._sound_dir / file_v
            v_sound = self._load(path) if path.exists() else self._make_accent(h_sound)
        elif file_base:
            path = self._sound_dir / file_base
            base_sound = self._load(path) if path.exists() else h_sound
            v_sound = self._make_accent(base_sound)
        else:
            v_sound = self._make_accent(h_sound)

        return h_sound, v_sound

    def _load(self, path: Path) -> "pygame.mixer.Sound":
        """Load a file through miniaudio with optional RMS normalization."""
        return _load_sound_file(path, normalize=self._normalize, target_rms_db=self._target_rms_db)

    def _synthesize(self, name: str) -> "pygame.mixer.Sound":
        cached = self._sound_dir / f"{name}.wav"
        if not cached.exists():
            synth_fn = _SYNTH_FUNCS.get(name, _synth_kick)
            _save_wav(cached, _to_stereo_int16(synth_fn()))
        return self._load(cached)

    def _make_accent(self, h_sound: "pygame.mixer.Sound") -> "pygame.mixer.Sound":
        """85% volume variant of h_sound for the vertical/accent hit."""
        arr = pygame.sndarray.array(h_sound)
        arr_accent = (arr * 0.85).astype(arr.dtype)
        return pygame.sndarray.make_sound(arr_accent)

    def play(self, row: int, orientation: int) -> None:
        """Play H or V sound for the given row on its dedicated channel."""
        if orientation == HORIZONTAL:
            sounds = self._h_sounds
        elif orientation == VERTICAL:
            sounds = self._v_sounds
        else:
            return
        if (0 <= row < len(sounds) and sounds[row] is not None
                and row < len(self._channels)):
            self._channels[row].play(sounds[row])

    def stop_all(self) -> None:
        pygame.mixer.stop()
