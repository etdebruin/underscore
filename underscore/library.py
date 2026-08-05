"""Local music library: every generated clip is kept, described, and reusable.

Each clip is an audio file plus a JSON sidecar (mood, the scene description it
was composed for, length). The catalog is shown to the analyzer so it can reuse
a clip that fits — free, and sonically consistent across episodes — instead of
generating a new one.
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_ROOT = Path.home() / ".underscore" / "library"
TARGET_PEAK = 0.7


@dataclass
class ClipEntry:
    clip_id: str
    mood: str
    description: str
    length_ms: int


class Library:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or DEFAULT_ROOT)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, clip_id: str) -> Path:
        return self.root / f"{clip_id}.audio"

    def register(self, clip_id: str, mood: str, description: str, length_ms: int,
                 prompt: str = "") -> None:
        meta = {"clip_id": clip_id, "mood": mood, "description": description,
                "length_ms": length_ms, "prompt": prompt}
        (self.root / f"{clip_id}.json").write_text(json.dumps(meta, indent=2))

    def catalog(self) -> list[ClipEntry]:
        entries = []
        for meta_path in sorted(self.root.glob("*.json")):
            meta = json.loads(meta_path.read_text())
            if self.path_for(meta["clip_id"]).exists():
                entries.append(ClipEntry(
                    clip_id=meta["clip_id"], mood=meta.get("mood", ""),
                    description=meta.get("description", ""),
                    length_ms=int(meta.get("length_ms", 0)),
                ))
        return entries

    def has(self, clip_id: str) -> bool:
        return self.path_for(clip_id).exists()

    def delete(self, clip_id: str) -> None:
        self.path_for(clip_id).unlink(missing_ok=True)
        (self.root / f"{clip_id}.json").unlink(missing_ok=True)

    def load(self, clip_id: str, duration: float, sr: int) -> np.ndarray:
        """Stereo float32 clip fitted to exactly `duration` seconds."""
        clip = _decode(self.path_for(clip_id), sr)
        return fit_clip(clip, round(duration * sr))


def fit_clip(clip: np.ndarray, n: int) -> np.ndarray:
    """Trim or loop a stereo clip to n samples, peak-normalized."""
    if clip.shape[0] >= n:
        clip = clip[:n]
    else:
        reps = int(np.ceil(n / clip.shape[0]))
        clip = np.tile(clip, (reps, 1))[:n]
    peak = np.max(np.abs(clip))
    if peak > 0:
        clip = clip * (TARGET_PEAK / peak)
    return clip.astype(np.float32)


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


def catalog_text(entries: list[ClipEntry]) -> str:
    """Render the catalog for the analyzer prompt."""
    if not entries:
        return ""
    lines = ["Existing music clips available for reuse (prefer reusing when one fits):"]
    for e in entries:
        lines.append(
            f"- clip_id={e.clip_id}  mood={e.mood}  length={e.length_ms / 1000:.0f}s  "
            f"made for: {e.description or 'n/a'}"
        )
    return "\n".join(lines)
