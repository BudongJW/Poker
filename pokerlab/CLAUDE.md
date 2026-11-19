# pokerlab: notes for Claude Code

`pokerlab/` is an independent package in this repository; `poker/` next to it is a separate older project, leave it alone. Start from `HANDOFF.md` (state, open items) and `README.md` (what everything does, with measured results).

## Working with the user

- Reply in Korean. The user reads on a phone a lot.
- No AI-typical punctuation in anything the user sees (replies, UI text, CLI output, README): no middle dots (·), arrows (→), long dashes (—), angle brackets or "…". Use commas, parentheses, words, plain hyphens.
- Tables in the UI: headers centred, cells vertically centred, short columns on one line.
- When asked to check what needs improving, measure (tests, `tools/misread_audit.py`, replays, screenshots) rather than guess, then give a numbered list; the user picks numbers.
- Report results faithfully, with the numbers, including what did not work.

## API keys and money

- Never ask for an API key in chat. The key is the environment variable `ANTHROPIC_API_KEY` (`python -m pokerlab check` shows where it comes from without showing it).
- Every frame read costs money (Claude Opus 5.5 about $0.018). Say the expected cost and get a yes before measuring with real reads; replay saved reads (`tools/replay_reads.py`, `POKERLAB_VISION_REPLAY`) whenever a check does not need new reads.

## Checks before committing

- `python -m pytest pokerlab/tests -q` (about 2.5 minutes, all must pass).
- After changes to `follow.py` or `table.py`: `python pokerlab/tools/misread_audit.py` (and `--stacks 60,150,300,1000`), and `tools/replay_reads.py` on both fixtures; results must not get worse.
- Commit only on the working branch; push when asked or when a piece of work is done.

## Map

| File | What it holds |
|---|---|
| `vision.py` | the reader: prompt and JSON schema, Claude (default) and Gemini calls, prices, the optional short answer form, reading the hero's cards or the board again (the board from an enlarged crop) |
| `table.py` | a frame checked and put in order (`normalize`), a hand put back together from its frames (`reconstruct`): misread smoothing (`_steady`, `_rising`, `_answered`), whole-hand amounts (`_street_bets`), the betting replay (`_Street.play`, guessed order) |
| `follow.py` | `Follower`, one per window: new hands, frames set aside as passing moments or one-off misreads, the hero's seat kept per source, hero cards and board by majority with their checks, doubts for the saving reader, records, quality and spending |
| `web.py`, `static/index.html` | the local page and its API (`/api/capture`, `/api/analyze`) |
| `records.py` | learning the `live` model from recorded hands, quality per program |
| `check.py` | `python -m pokerlab check` |
| `advisor.py`, `equity.py`, `preflop.py`, `inference.py` | equity, ranges and the opponent action model |
| `engine.py`, `bots.py`, `selfplay.py`, `daily.py`, `formats.py` | the engine, bots, self-play and the daily learning cycle |
| `tools/` | development tools and saved reads (see `tools/README.md`) |
