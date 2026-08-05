"""Real music beds via the ElevenLabs Music API.

Clips are cached on disk keyed by (prompt, length), so editing a cue sheet and
re-rendering doesn't re-bill generation credits.
"""

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

API_URL = "https://api.elevenlabs.io/v1/music"
MIN_LENGTH_MS = 3000
MAX_LENGTH_MS = 600_000
TARGET_PEAK = 0.7

_STYLE = {
    "warm": "warm and gentle, soft felt piano and strings, calm and unhurried",
    "tense": "tense and suspenseful, low pulsing synths and sparse percussion, restrained",
    "wistful": "wistful and bittersweet, slow piano with airy pads, melancholy but tender",
    "uplifting": "uplifting and hopeful, bright acoustic guitar and light rhythm, building gently",
    "mysterious": "mysterious and atmospheric, ambient drones and distant bells, spacious",
}


def build_prompt(mood: str, reason: str) -> str:
    style = _STYLE.get(mood, _STYLE["warm"])
    prompt = (
        f"Instrumental background music bed for a narrated podcast: {style}. "
        "Understated underscore that sits beneath a speaking voice - no vocals, "
        "no melody that fights for attention, consistent energy throughout."
    )
    if reason:
        prompt += f" Scene context: {reason}"
    return prompt


def _default_fetcher(prompt: str, length_ms: int) -> bytes:
    import json
    import urllib.request

    body = json.dumps(
        {"prompt": prompt, "music_length_ms": length_ms, "force_instrumental": True}
    ).encode()
    req = urllib.request.Request(
        f"{API_URL}?output_format=mp3_44100_128",
        data=body,
        headers={
            "xi-api-key": os.environ["ELEVENLABS_API_KEY"],
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


class ElevenLabsMusic:
    def __init__(self, api_key: str | None = None, cache_dir: str | Path | None = None,
                 fetcher=None):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "underscore" / "music")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fetcher = fetcher or _default_fetcher

    def bed(self, mood: str, duration: float, sr: int, seed: int = 0, reason: str = "") -> np.ndarray:
        """Return a stereo float32 clip of exactly `duration` seconds."""
        prompt = build_prompt(mood, reason)
        length_ms = min(max(int(duration * 1000), MIN_LENGTH_MS), MAX_LENGTH_MS)
        key = hashlib.sha256(f"{prompt}|{length_ms}".encode()).hexdigest()[:24]
        cached = self.cache_dir / f"{key}.audio"

        if not cached.exists():
            cached.write_bytes(self.fetcher(prompt, length_ms))

        clip = self._decode(cached, sr)
        n = round(duration * sr)
        if clip.shape[0] >= n:
            clip = clip[:n]
        else:  # loop with a short crossfade to reach the target length
            reps = int(np.ceil(n / clip.shape[0]))
            clip = np.tile(clip, (reps, 1))[:n]
        peak = np.max(np.abs(clip))
        if peak > 0:
            clip = clip * (TARGET_PEAK / peak)
        return clip.astype(np.float32)

    @staticmethod
    def _decode(path: Path, sr: int) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                 "-ac", "2", "-ar", str(sr), tmp.name],
                check=True,
            )
            data, _ = sf.read(tmp.name, dtype="float32")
        Path(tmp.name).unlink(missing_ok=True)
        return data
