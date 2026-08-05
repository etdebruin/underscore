import io
import json

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from underscore import server
from underscore.cues import CueSheet, Insert, Underlay
from underscore.transcribe import Transcript, Word

SR = 44100


def _wav_bytes(seconds: float = 2.0) -> bytes:
    t = np.linspace(0, seconds, int(seconds * SR), endpoint=False)
    tone = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, tone, SR, format="WAV")
    return buf.getvalue()


def _fake_transcript() -> Transcript:
    return Transcript(
        text="hello world.",
        words=[Word("hello", 0.1, 0.5), Word("world.", 0.6, 1.0)],
        duration=2.0,
    )


def _fake_sheet() -> CueSheet:
    return CueSheet(
        inserts=[Insert(time=1.0, duration=3.0, mood="warm", reason="test")],
        underlays=[Underlay(start=0.0, end=2.0, mood="wistful", reason="bed")],
    )


def _client(tmp_path):
    from underscore.library import Library

    app = server.create_app(
        projects_root=tmp_path / "projects",
        transcribe_fn=lambda path: _fake_transcript(),
        analyze_fn=lambda transcript, scoring, catalog: _fake_sheet(),
        library=Library(tmp_path / "lib"),
        sync_jobs=True,
    )
    return TestClient(app)


def test_upload_creates_ready_project(tmp_path):
    client = _client(tmp_path)
    resp = client.post(
        "/api/projects",
        files={"file": ("ep1.wav", _wav_bytes(), "audio/wav")},
        data={"scoring": "standard"},
    )
    assert resp.status_code == 200
    pid = resp.json()["id"]

    detail = client.get(f"/api/projects/{pid}").json()
    assert detail["status"] == "ready"
    assert detail["name"] == "ep1.wav"
    assert len(detail["cues"]["inserts"]) == 1
    assert detail["duration"] > 0


def test_peaks_available_after_upload(tmp_path):
    client = _client(tmp_path)
    pid = client.post(
        "/api/projects", files={"file": ("e.wav", _wav_bytes(), "audio/wav")}
    ).json()["id"]
    peaks = client.get(f"/api/projects/{pid}/peaks").json()
    assert len(peaks) > 100
    assert all(lo <= hi for lo, hi in peaks)


def test_low_res_peaks_upgraded_on_read(tmp_path):
    client = _client(tmp_path)
    pid = client.post(
        "/api/projects", files={"file": ("e.wav", _wav_bytes(), "audio/wav")}
    ).json()["id"]
    stale = tmp_path / "projects" / pid / "peaks.json"
    stale.write_text(json.dumps([[-0.1, 0.1]] * 10))
    peaks = client.get(f"/api/projects/{pid}/peaks").json()
    assert len(peaks) == server.PEAK_BUCKETS


def test_cue_roundtrip(tmp_path):
    client = _client(tmp_path)
    pid = client.post(
        "/api/projects", files={"file": ("e.wav", _wav_bytes(), "audio/wav")}
    ).json()["id"]

    cues = client.get(f"/api/projects/{pid}").json()["cues"]
    cues["inserts"][0]["mood"] = "tense"
    cues["inserts"][0]["time"] = 1.4
    resp = client.put(f"/api/projects/{pid}/cues", json=cues)
    assert resp.status_code == 200

    fresh = client.get(f"/api/projects/{pid}").json()["cues"]
    assert fresh["inserts"][0]["mood"] == "tense"
    assert fresh["inserts"][0]["time"] == 1.4


def test_render_produces_output(tmp_path):
    client = _client(tmp_path)
    pid = client.post(
        "/api/projects", files={"file": ("e.wav", _wav_bytes(), "audio/wav")}
    ).json()["id"]

    resp = client.post(f"/api/projects/{pid}/render",
                       json={"warm": False, "level": False, "music": "synth"})
    assert resp.status_code == 200
    assert client.get(f"/api/projects/{pid}").json()["status"] == "ready"

    audio = client.get(f"/api/projects/{pid}/audio/scored")
    assert audio.status_code == 200
    assert len(audio.content) > 1000


def test_projects_listed(tmp_path):
    client = _client(tmp_path)
    client.post("/api/projects", files={"file": ("a.wav", _wav_bytes(), "audio/wav")})
    client.post("/api/projects", files={"file": ("b.wav", _wav_bytes(), "audio/wav")})
    listing = client.get("/api/projects").json()
    assert {p["name"] for p in listing} == {"a.wav", "b.wav"}


def test_transcript_endpoint_returns_sentences(tmp_path):
    client = _client(tmp_path)
    pid = client.post(
        "/api/projects", files={"file": ("e.wav", _wav_bytes(), "audio/wav")}
    ).json()["id"]
    sentences = client.get(f"/api/projects/{pid}/transcript").json()
    assert sentences == [{"start": 0.1, "end": 1.0, "text": "hello world."}]


def test_library_endpoint(tmp_path):
    client = _client(tmp_path)
    assert client.get("/api/library").json() == []


def test_cue_audio_falls_back_to_synth_standin(tmp_path):
    client = _client(tmp_path)
    pid = client.post(
        "/api/projects", files={"file": ("e.wav", _wav_bytes(), "audio/wav")}
    ).json()["id"]
    resp = client.get(f"/api/projects/{pid}/cue-audio/insert/0")
    assert resp.status_code == 200
    assert resp.headers["x-clip-source"] == "synth"
    assert len(resp.content) > 1000


def test_index_served(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "underscore" in resp.text


def test_unknown_project_404(tmp_path):
    client = _client(tmp_path)
    assert client.get("/api/projects/nope").status_code == 404


def test_saved_cues_survive_json_file(tmp_path):
    client = _client(tmp_path)
    pid = client.post(
        "/api/projects", files={"file": ("e.wav", _wav_bytes(), "audio/wav")}
    ).json()["id"]
    cues_path = tmp_path / "projects" / pid / "cues.json"
    assert cues_path.exists()
    data = json.loads(cues_path.read_text())
    assert "inserts" in data and "underlays" in data
