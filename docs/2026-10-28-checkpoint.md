# Checkpoint 2026-10-28

Pick-up point for the next session. Judgements below are **by ear from the user**; objective metrics (chroma, tempo) did not predict quality and are not a gate.

## Best recipe so far

```bash
# 1. sample a long reference (spread windows keep the set's order)
python scripts/fetch_audio.py URL set1 --spread 12 --seconds 30
# 2. ACE-Step server:  cd D:/dev/ACE-Step-1.5 && uv run acestep-api
# 3. one command
TIMELINE=data/samples/set1.wav scripts/chain.sh myset "electronic DJ set, club mix, driving groove" 180
```

Magenta `--timeline` + `--reset-every 60` + temperature 1.0 + cfg 6, then ACE cover strength **0.6**.

## Findings

| Question | Result |
|---|---|
| Browser `<audio>` vs Web Audio for live | Web Audio fixed stutters (0 underruns over 30 s) |
| Magenta alone on long runs | Quality drops after ~1:20 (carried state) |
| `--reset-every 40/60` | Fixed the drop; user preferred these sketches. Noise floor ~20% higher (denser) but not hiss |
| Temperature 1.3 vs 1.0 | No measurable noise-floor change; 1.0 kept (sounded better) |
| Timeline vs pooled style | Timeline better overall |
| ACE cover strength on a weak sketch | 0.3 won |
| ACE cover strength on a good sketch | 0.3 made it worse; **0.6 narrowly won**, 0.8 ~ tied with raw sketch; inconsistent per section |
| 3-way blend (ref audio + 2 text styles) | Incoherent; 2 parts or text-dominant works better |
| ACE `MODE=ref` (style-reference, new structure) | Generated but not preferred over cover 0.3 |
| 600 s ACE cover | Works in one pass (~30 s on the 4080 Super, turbo model) |

## State of the experiments

- Sets: `data/samples/set1.wav` (12 windows, 6 min), `set2.wav` (20 windows, 10 min) from the 119-minute DJ set.
- 10-minute run with the full recipe: `data/chain/mag_set10.wav`, `ace_set10.wav` (9.9 min). **Not yet listened to.**
- Lyrics experiment on the vocal track (`data/samples/heat_full.wav`, 3:50), lyrics in `data/lyrics/heat.txt` (gitignored, never commit third-party lyrics):
  - `data/lyrics_run/ace_direct.wav`: ACE cover of the original with lyrics (baseline)
  - `data/chain/ace_heat.wav`: Magenta timeline sketch + ACE cover with lyrics
  - **Not yet listened to.** Caption used: "indie electropop, male vocals, dreamy synths, melancholic, mid-tempo groove".
- Everything is committed through `2395d4c`; only `data/` outputs are local.

## Open questions for tomorrow

1. Listen to `ace_set10` (seams at each 60 s reset? late drift?) and the two lyrics clips (are words intelligible and in time?).
2. If words are mushy: try other cover strengths for lyrics, or a lyrics-first run keeping less original audio.
3. Per-section inconsistency of the 0.6 cover: consider choosing raw-vs-ACE per segment (needs a judge other than ears) or crossfading 0.4/0.6/0.8.
4. Beat-matched resets instead of a plain crossfade, if seams are audible.
5. Productise cover mode in `moonstar-music` (its ACE provider is text2music only).
6. Optional: "send to ACE" from the live page; crossfade between live prompts; `docker-compose.yml` for the long `docker run`.

## Gotchas learned

- `TEMP` is a Windows env var: the temperature override is `MAG_TEMP`.
- ACE rejects JSON `src_audio_path` outside the system temp dir: upload as multipart (`ace_cover.sh` does).
- Heredocs with Windows paths/backslashes in Git Bash and CRLF line endings broke scripted edits twice; use the file tools for edits.
- `fetch_audio.py` downloads the whole video each run (117 MB for the set); the temp copy is deleted.
- ACE-Step server and the Magenta container compete for VRAM; Docker Desktop may need starting.
- Magenta has no sampling seed; `--seed` only affects text-prompt embedding, so runs are not reproducible.
