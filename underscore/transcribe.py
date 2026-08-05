"""Transcription with word-level timestamps via mlx-whisper (Apple Silicon)."""

from dataclasses import dataclass

WHISPER_MODEL = "mlx-community/whisper-base-mlx"


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Transcript:
    text: str
    words: list[Word]
    duration: float

    def speech_spans(self, join_gap: float = 0.35) -> list[tuple[float, float]]:
        """Merge word timings into continuous speech spans."""
        spans: list[list[float]] = []
        for w in self.words:
            if spans and w.start - spans[-1][1] <= join_gap:
                spans[-1][1] = w.end
            else:
                spans.append([w.start, w.end])
        return [(s, e) for s, e in spans]


def transcribe(audio_path: str) -> Transcript:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio_path, path_or_hf_repo=WHISPER_MODEL, word_timestamps=True
    )
    words = [
        Word(text=w["word"].strip(), start=float(w["start"]), end=float(w["end"]))
        for seg in result["segments"]
        for w in seg.get("words", [])
    ]
    duration = words[-1].end if words else 0.0
    return Transcript(text=result["text"].strip(), words=words, duration=duration)
