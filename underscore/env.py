"""Environment setup shared by the CLI and the web server."""

import os
import shutil
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=value lines from a .env file; existing env vars win."""
    env = Path(path)
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def preflight() -> list[str]:
    """Human-readable notes about anything a fresh setup is missing."""
    notes = []
    if shutil.which("ffmpeg") is None:
        notes.append(
            "ffmpeg not found - audio decode/master will fail. Install it "
            "(macOS: `brew install ffmpeg`)."
        )
    try:
        import mlx_whisper  # noqa: F401
    except ImportError:
        notes.append(
            "mlx-whisper unavailable - transcription requires macOS on Apple "
            "Silicon. Tracks cannot be ingested on this machine."
        )
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        notes.append(
            "No Anthropic credentials - cue analysis will fail. Set "
            "ANTHROPIC_API_KEY (env or .env), or run `ant auth login`."
        )
    if not os.environ.get("ELEVENLABS_API_KEY"):
        notes.append(
            "No ELEVENLABS_API_KEY - music renders use the built-in synth "
            "pads. Add the key to .env for generated music."
        )
    return notes
