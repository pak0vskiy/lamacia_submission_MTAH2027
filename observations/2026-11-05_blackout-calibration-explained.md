# 2026-11-05 · The blackout calibration, in plain words

Author: Maxim. This is a plain-language companion to
[2026-11-05_blackout-calibration.md](2026-11-05_blackout-calibration.md), which has the numbers
and caveats. Aggregates only, no labels from our match.

## TL;DR

Our practice exam was harder than the real exam, and hard in the wrong way. We made the practice
data lose the ball the way our match does. The scores we expect went up (0.374 → 0.438), and the
model trained on the new practice data got slightly better too.

## Why we need "practice data" at all

- We can't label our match (Barcelona vs Athletic), not even to check ourselves. So we test every
  idea on **7 German matches (IDSSE)**, where the right answers come with the data.
- But those German matches have **perfect tracking**: the ball is always visible. Our match's
  tracking **loses the ball 41 % of the time**.
- So before testing, we deliberately **hide the ball** in the German matches to make them look
  like ours. We call this "degrading" them.
- If we hide the ball the wrong way, we're practising for the wrong exam. Every choice we make
  (which method, which settings) could then be wrong for our match.

## What was wrong with the old way

The old version only copied one fact from our match: "hide the ball 41 % of the time". It
didn't copy **when** the ball goes missing. So it:

1. **Hid the ball far too often during live play**: 31 % of live play, against **11 %** in our
   match.
2. **Used gaps that were far too long** in live play: a typical gap of 14 s, against **5.5 s** in
   our match.
3. **Put the gaps in random places.** In our match the ball tends to vanish **right after a
   duel** (players from both teams close to the ball), which is exactly when possession is
   likely to change.

In our match, most of the missing ball is **during stoppages**: the ball goes out, someone fetches
it, and the tracker drops it. It usually comes back **just before the restart**. Live play is
mostly visible.

## What we changed

We measured how the ball goes missing in our match, using only the tracking, never any answers.
Then we built a new way of hiding the ball in the German matches that copies it:

- **Stoppages:** about 3 in 4 stoppages get a gap. It starts around the moment play stops and
  ends just before the restart, with the same spread of timings as in our match.
- **Live play:** a few short gaps, at the same rate as our match (about one every 1.5 minutes of
  play). About a third are tiny blips of 1–3 frames, the rest 5–11 s.
- **Duels:** live gaps are made more likely right after a duel, as in our match.

Then we checked it worked by measuring the new German data the same way as our match:

| | our match | old practice data | new practice data |
|---|---|---|---|
| ball missing in live play | 10.6 % | 31.4 % | 12.3 % |
| ball missing in stoppages | 76.7 % | 53.5 % | 70.7 % |
| typical live gap | 5.5 s | 14.2 s | 5.7 s |
| how much likelier a gap is right after a duel | 1.7× | 0.8× | 1.7× |

Close enough everywhere that matters. Stoppages are still slightly under-hidden; that's noted.

## What it changed in our results

- **The scores we expect went up.** The old practice data was too hard in live play, where
  possession is decided. Our team-possession rule (Vidal-Codina) now scores **0.438** instead of
  0.374 on spellF1 (the score the ranking uses).
- **The dead-ball model got a bit better** when retrained on the realistic data: 0.900 against
  0.886.
- **One setting moved:** the "possession zone" (how close a player must be to count as having the
  ball) is now **2.0 m** instead of 2.5 m. All 7 test rounds picked it.
- **The ranking of methods didn't change.** Vidal-Codina still clearly beats the other two team
  methods we tried (both 0.347).
- **The submission now uses all of this**, and it rebuilds identically with one command:
  `uv run python scripts/make_submission.py`.

## What it tells us about next steps

- **The dead-ball side is in good shape.** A perfect dead-ball detector would only add about 0.04.
- **The big remaining gap is deciding which team has the ball:** 0.44 for us, against 0.90 when
  we cheat with the German matches' real possession data. That's where to put effort next.
- **One small leftover:** the ~2 minutes before the second-half kickoff should all be "dead
  ball", but we only label about 58 % of it that way. We'll fix that with a kickoff detector.

## Rules check

- Nothing from our match's answers was used, public or private. We only measured how the
  tracking behaves (when the ball is missing), the same as we did for the ball smoothing.
- The "live" vs "dead" split on our match comes from our own model's guesses. So the targets are
  approximate, but we measured the German data the same way to keep the comparison fair.
