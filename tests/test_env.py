import os

from underscore.env import load_dotenv, preflight


def test_load_dotenv_sets_and_respects_existing(tmp_path, monkeypatch):
    envfile = tmp_path / ".env"
    envfile.write_text("# comment\nFOO_KEY=abc\nBAR_KEY = spaced \n")
    monkeypatch.delenv("FOO_KEY", raising=False)
    monkeypatch.setenv("BAR_KEY", "already-set")

    load_dotenv(envfile)
    assert os.environ["FOO_KEY"] == "abc"
    assert os.environ["BAR_KEY"] == "already-set"  # existing env wins


def test_load_dotenv_missing_file_is_noop(tmp_path):
    load_dotenv(tmp_path / "nope.env")  # must not raise


def test_preflight_flags_missing_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    notes = " ".join(preflight())
    assert "Anthropic" in notes
    assert "ELEVENLABS_API_KEY" in notes


def test_preflight_quiet_when_configured(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "y")
    notes = preflight()
    assert not any("API_KEY" in n or "Anthropic" in n for n in notes)
