# underscore

register: product

## Product Purpose

Score a podcast voice track with music, automatically. Whisper transcribes,
Claude (as a radio producer) writes a cue sheet, ElevenLabs generates music
beds, a numpy mixer assembles and masters the episode. The web UI exists for
one job: **reviewing and editing the cue sheet** — see the waveform, audition
Claude's choices, nudge or delete cues, re-render cheaply from the clip library.

## Users

Currently one person: the author, a technical podcast producer working on a
Mac, editing episodes in focused sessions. Comfortable with CLIs; wants eyes
on the timeline, not JSON. May become a tool for other podcasters later.

## Tone

Calm, editorial, radio-craft. The aesthetic ancestor is public-radio
production: tape, waveforms, cue sheets, studio restraint. Confidence through
precision, not decoration.

## Anti-references

- Generic SaaS dashboard (cards, KPI tiles, purple gradients)
- DAW maximalism (hundreds of knobs; this is a review tool, not a DAW)
- Toy-like "AI magic" styling (sparkle emoji, gradient buttons)

## Strategic principles

- The waveform is the interface; everything else supports it.
- Claude proposes, the human disposes: editing must be faster than regenerating.
- Every render should feel cheap (library reuse), every edit reversible (cue
  sheet is plain JSON on disk).
