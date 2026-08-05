# underscore

Score a podcast voice track with music, automatically.

*Underscore* (n.): music played beneath dialogue or narration. Give this tool a
voice recording and it finds the moments that deserve music, then renders a new
mixed track — the way a radio producer would.

Two techniques, both chosen and placed by an LLM acting as producer:

1. **Inserts** — the narration pauses for a short musical sting at act breaks
   and emotional beats, with the music bleeding slightly under the surrounding
   speech so the edit sounds produced rather than pasted.
2. **Underlays** — a quiet mood bed plays under scene-setting or emotionally
   charged passages, automatically ducked while speech is present and swelling
   in the pauses.

## How it works

```
voice.wav ──▶ transcribe ──▶ analyze ──▶ mix ──▶ loudnorm ──▶ scored.mp3
              (mlx-whisper)   (Claude)    (numpy)  (ffmpeg)
```

1. **Transcribe** — [mlx-whisper](https://github.com/ml-explore/mlx-examples)
   produces word-level timestamps locally (Apple Silicon).
2. **Analyze** — Claude reads the timestamped transcript as a podcast producer
   and returns a structured cue sheet: where to insert stings, where to lay
   beds, and what mood each should be. The cue sheet is plain JSON you can
   review and edit by hand before rendering.
3. **Mix** — pure-numpy assembly: pauses are spliced into the voice track for
   inserts, beds are gain-enveloped (ducking driven by the word timestamps),
   and everything is soft-limited.
4. **Master** — ffmpeg `loudnorm` to −16 LUFS, the podcast standard.

Music beds are procedurally synthesized pads (five moods: warm, tense, wistful,
uplifting, mysterious) so the tool is fully self-contained. The `music.py`
module is deliberately swappable for a licensed track library or a music
generation API.

## Usage

Requires macOS on Apple Silicon, `ffmpeg`, [`uv`](https://docs.astral.sh/uv/),
and an `ANTHROPIC_API_KEY`.

```sh
uv run python -m underscore.cli voice.wav -o scored.mp3 --save-cues cues.json
```

Edit `cues.json` to taste, then re-render without another API call:

```sh
uv run python -m underscore.cli voice.wav -o scored.mp3 --cues cues.json
```

## Example

`examples/the-lighthouse.txt` is a short NPR-style story. Synthesize a
narration and score it:

```sh
say -v Samantha -r 165 -o voice.aiff -f examples/the-lighthouse.txt
ffmpeg -i voice.aiff -ar 44100 voice.wav
uv run python -m underscore.cli voice.wav -o scored.mp3
```

## Development

```sh
uv sync
uv run pytest
uv run ruff check .
```

## License

MIT
