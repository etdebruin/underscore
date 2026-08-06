"""Local web UI: a two-lane DAW-style view over the scoring pipeline.

Projects live in ~/.underscore/projects/<id>/ as plain files (source audio,
transcript.json, cues.json, scored.mp3) so everything stays inspectable and
editable outside the UI too.
"""

import dataclasses
import hashlib
import json
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .analyze import analyze
from .cues import CueSheet, sheet_from_dict
from .elevenlabs_music import MIN_LENGTH_MS, build_prompt
from .library import Library, catalog_text
from .mix import POST_OVERLAP, PRE_OVERLAP, assemble, find_gaps, snap_to_gap
from .transcribe import Transcript, Word, transcribe
from .voicefx import process_voice

SR = 44100
PEAK_BUCKETS = 12000
DEFAULT_PROJECTS_ROOT = Path.home() / ".underscore" / "projects"
STATIC_DIR = Path(__file__).parent / "static"


def _compute_peaks(voice: np.ndarray, buckets: int = PEAK_BUCKETS) -> list[list[float]]:
    n = len(voice)
    edges = np.linspace(0, n, buckets + 1, dtype=int)
    peaks = []
    import itertools

    for a, b in itertools.pairwise(edges):
        chunk = voice[a:b] if b > a else np.zeros(1, dtype=np.float32)
        peaks.append([round(float(chunk.min()), 3), round(float(chunk.max()), 3)])
    return peaks


def _cue_length_s(kind: str, cue: dict) -> float:
    if kind == "insert":
        return PRE_OVERLAP + float(cue["duration"]) + POST_OVERLAP
    return float(cue["end"]) - float(cue["start"])


def _snap_inserts(sheet: CueSheet, transcript: Transcript, max_dist: float) -> CueSheet:
    """Move insert points into real silences so the UI, preview, and render agree.

    Claude proposes sentence-boundary times, which land on the onset of the
    next sentence; the pause belongs in the silence just before it.
    """
    gaps = find_gaps(transcript.speech_spans(), transcript.duration)
    for ins in sheet.inserts:
        ins.time = round(snap_to_gap(ins.time, gaps, max_dist=max_dist), 2)
    for clip in sheet.clips:
        clip.time = round(snap_to_gap(clip.time, gaps, max_dist=max_dist), 2)
    return sheet


def _clip_key(mood: str, reason: str, seconds: float) -> str:
    prompt = build_prompt(mood, reason)
    length_ms = max(int(seconds * 1000), MIN_LENGTH_MS)
    return hashlib.sha256(f"{prompt}|{length_ms}".encode()).hexdigest()[:24]


class _Project:
    def __init__(self, root: Path):
        self.root = root

    @property
    def id(self) -> str:
        return self.root.name

    def meta(self) -> dict:
        return json.loads((self.root / "meta.json").read_text())

    def update(self, **fields) -> None:
        meta = self.meta() if (self.root / "meta.json").exists() else {}
        meta.update(fields)
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2))

    def cues(self) -> dict:
        return json.loads((self.root / "cues.json").read_text())

    def transcript(self) -> Transcript:
        data = json.loads((self.root / "transcript.json").read_text())
        return Transcript(
            text=data["text"],
            words=[Word(**w) for w in data["words"]],
            duration=data["duration"],
        )

    def source_path(self) -> Path:
        return next(self.root.glob("source.*"))


def create_app(
    projects_root: str | Path | None = None,
    transcribe_fn=None,
    analyze_fn=None,
    library: Library | None = None,
    sync_jobs: bool = False,
) -> FastAPI:
    root = Path(projects_root or DEFAULT_PROJECTS_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    lib = library or Library()
    transcribe_fn = transcribe_fn or transcribe
    analyze_fn = analyze_fn or (
        lambda transcript, scoring, catalog: analyze(transcript, scoring=scoring, catalog=catalog)
    )
    app = FastAPI(title="underscore")

    # Jobs die with the process: anything still marked busy was interrupted.
    for meta_path in root.glob("*/meta.json"):
        meta = json.loads(meta_path.read_text())
        if meta.get("status") in ("transcribing", "analyzing", "rendering"):
            meta.update(status="error", error="interrupted by a server restart - re-add the track")
            meta_path.write_text(json.dumps(meta, indent=2))

    def project(pid: str) -> _Project:
        p = _Project(root / pid)
        if not (p.root / "meta.json").exists():
            raise HTTPException(404, "unknown project")
        return p

    def run_job(p: _Project, fn) -> None:
        def wrapped():
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - job boundary; surfaced in the UI
                p.update(status="error", error=str(exc))

        if sync_jobs:
            wrapped()
        else:
            threading.Thread(target=wrapped, daemon=True).start()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC_DIR / "index.html").read_text()

    @app.get("/api/projects")
    def list_projects():
        out = []
        for meta_path in sorted(root.glob("*/meta.json"), key=lambda p: p.stat().st_mtime):
            meta = json.loads(meta_path.read_text())
            out.append({"id": meta_path.parent.name, "name": meta.get("name"),
                        "status": meta.get("status")})
        return out

    @app.post("/api/projects")
    async def upload(file: UploadFile, scoring: str = "standard"):
        pid = uuid.uuid4().hex[:10]
        p = _Project(root / pid)
        p.root.mkdir(parents=True)
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        source = p.root / f"source{suffix}"
        partial = p.root / "source.partial"
        with partial.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        if partial.stat().st_size == 0:
            shutil.rmtree(p.root)
            raise HTTPException(
                400,
                f"'{file.filename}' arrived empty. If it lives in iCloud/OneDrive/"
                "Dropbox, download it locally first, then drop it again.",
            )
        partial.rename(source)
        p.update(name=file.filename, status="transcribing", error=None,
                 scoring=scoring, duration=0.0)

        def job():
            try:
                voice = process_voice(str(source), SR, level=False, warm=False)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"couldn't decode '{file.filename}' - it doesn't look like a "
                    "readable audio file"
                ) from exc
            sf.write(p.root / "voice.wav", voice, SR)
            (p.root / "peaks.json").write_text(json.dumps(_compute_peaks(voice)))
            transcript = transcribe_fn(str(p.root / "voice.wav"))
            (p.root / "transcript.json").write_text(json.dumps({
                "text": transcript.text,
                "words": [dataclasses.asdict(w) for w in transcript.words],
                "duration": transcript.duration,
            }))
            p.update(status="analyzing", duration=len(voice) / SR)
            sheet = analyze_fn(transcript, p.meta()["scoring"], catalog_text(lib.catalog()))
            sheet = _snap_inserts(sheet, transcript, max_dist=6.0)
            (p.root / "cues.json").write_text(json.dumps(dataclasses.asdict(sheet), indent=2))
            p.update(status="ready")

        run_job(p, job)
        return {"id": pid}

    @app.delete("/api/projects/{pid}")
    def delete_project(pid: str):
        p = project(pid)
        shutil.rmtree(p.root)
        return {"ok": True}

    @app.get("/api/projects/{pid}")
    def detail(pid: str):
        p = project(pid)
        meta = p.meta()
        cues = p.cues() if (p.root / "cues.json").exists() else None
        return {**meta, "id": pid, "cues": cues,
                "has_scored": (p.root / "scored.mp3").exists(),
                "has_peaks": (p.root / "peaks.json").exists()}

    @app.get("/api/projects/{pid}/transcript")
    def transcript(pid: str):
        p = project(pid)
        if not (p.root / "transcript.json").exists():
            raise HTTPException(404, "transcript not ready")
        return p.transcript().sentences()

    @app.get("/api/projects/{pid}/peaks")
    def peaks(pid: str):
        p = project(pid)
        path = p.root / "peaks.json"
        if not path.exists():
            raise HTTPException(404, "peaks not ready")
        data = json.loads(path.read_text())
        voice_path = p.root / "voice.wav"
        if len(data) < PEAK_BUCKETS and voice_path.exists():
            voice, _ = sf.read(voice_path, dtype="float32")
            data = _compute_peaks(voice)
            path.write_text(json.dumps(data))
        return data

    @app.put("/api/projects/{pid}/cues")
    def save_cues(pid: str, cues: dict):
        p = project(pid)
        sheet = sheet_from_dict(cues)
        if (p.root / "transcript.json").exists():
            # gentler tolerance for hand-placed cues than for Claude's proposals
            sheet = _snap_inserts(sheet, p.transcript(), max_dist=2.0)
        (p.root / "cues.json").write_text(json.dumps(dataclasses.asdict(sheet), indent=2))
        return {"ok": True, "cues": dataclasses.asdict(sheet)}

    @app.post("/api/projects/{pid}/analyze")
    def reanalyze(pid: str, body: dict):
        p = project(pid)
        p.update(status="analyzing", scoring=body.get("scoring", "standard"))

        def job():
            transcript = p.transcript()
            sheet = analyze_fn(transcript, p.meta()["scoring"], catalog_text(lib.catalog()))
            sheet = _snap_inserts(sheet, transcript, max_dist=6.0)
            (p.root / "cues.json").write_text(json.dumps(dataclasses.asdict(sheet), indent=2))
            p.update(status="ready")

        run_job(p, job)
        return {"ok": True}

    @app.post("/api/projects/{pid}/render")
    def render(pid: str, body: dict):
        p = project(pid)
        p.update(status="rendering")

        def job():
            sheet = sheet_from_dict(p.cues())
            bed_fn = None
            if body.get("music", "elevenlabs") == "elevenlabs":
                from .elevenlabs_music import ElevenLabsMusic

                bed_fn = ElevenLabsMusic(library=lib).bed
            voice = process_voice(str(p.source_path()), SR,
                                  level=body.get("level", True), warm=body.get("warm", True))
            master = assemble(voice, SR, sheet, p.transcript().speech_spans(), bed_fn=bed_fn,
                              clip_fn=lib.load_raw if sheet.clips else None)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, master, SR)
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name,
                     "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", str(p.root / "scored.mp3")],
                    check=True,
                )
            Path(tmp.name).unlink(missing_ok=True)
            p.update(status="ready")

        run_job(p, job)
        return {"ok": True}

    @app.get("/api/projects/{pid}/audio/{which}")
    def audio(pid: str, which: str):
        p = project(pid)
        path = {"voice": p.root / "voice.wav", "scored": p.root / "scored.mp3"}.get(which)
        if path is None or not path.exists():
            raise HTTPException(404, "no such audio")
        media = "audio/mpeg" if path.suffix == ".mp3" else "audio/wav"
        return FileResponse(path, media_type=media)

    @app.get("/api/projects/{pid}/cue-audio/{kind}/{index}")
    def cue_audio(pid: str, kind: str, index: int):
        """The music clip a cue resolves to, if it exists in the library yet."""
        p = project(pid)
        cues = p.cues()
        if kind == "clip":
            group = cues.get("clips", [])
            if index >= len(group):
                raise HTTPException(404, "no such cue")
            clip_id = group[index]["clip_id"]
            if not lib.has(clip_id):
                raise HTTPException(404, "clip not in library")
            return FileResponse(lib.path_for(clip_id), media_type="audio/mpeg",
                                headers={"X-Clip-Source": "library"})
        group = cues["inserts"] if kind == "insert" else cues["underlays"]
        if index >= len(group):
            raise HTTPException(404, "no such cue")
        cue = group[index]
        seconds = _cue_length_s(kind, cue)
        clip_id = cue.get("clip_id")
        if not (clip_id and lib.has(clip_id)):
            clip_id = _clip_key(cue["mood"], cue.get("reason", ""), seconds)
        if lib.has(clip_id):
            return FileResponse(lib.path_for(clip_id), media_type="audio/mpeg",
                                headers={"X-Clip-Source": "library"})
        # Free synth stand-in so the preview always has music before first render.
        from .music import generate_bed

        cache = p.root / "previews"
        cache.mkdir(exist_ok=True)
        key = hashlib.sha256(f"synth|{cue['mood']}|{seconds:.1f}".encode()).hexdigest()[:16]
        path = cache / f"{key}.wav"
        if not path.exists():
            sf.write(path, generate_bed(cue["mood"], seconds, SR, seed=index), SR)
        return FileResponse(path, media_type="audio/wav",
                            headers={"X-Clip-Source": "synth"})

    @app.get("/api/library")
    def library_catalog():
        return [dataclasses.asdict(e) for e in lib.catalog()]

    @app.delete("/api/library/{clip_id}")
    def delete_clip(clip_id: str):
        if not lib.has(clip_id):
            raise HTTPException(404, "unknown clip")
        lib.delete(clip_id)
        return {"ok": True}

    @app.get("/api/library/{clip_id}/audio")
    def library_audio(clip_id: str):
        if not lib.has(clip_id):
            raise HTTPException(404, "unknown clip")
        return FileResponse(lib.path_for(clip_id), media_type="audio/mpeg")

    return app


def main() -> None:
    import webbrowser

    import uvicorn

    from .env import load_dotenv, preflight

    load_dotenv()
    for note in preflight():
        print(f"note: {note}")
    app = create_app()
    threading.Timer(0.8, lambda: webbrowser.open("http://127.0.0.1:8765")).start()
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")


if __name__ == "__main__":
    main()
