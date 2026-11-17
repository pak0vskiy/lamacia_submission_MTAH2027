# 2026-11-04 · Do DFL and Metrica mark dead ball the same way?

Author: Maxim. Method: `analysis/04_dead_ball_conventions.py` (aggregates only). Other games only:
IDSSE / DFL (7 matches, alive/dead flag + events) and Metrica sample games 1–3 (events only).
Nothing from BAR-ATH is read. Follows [2026-11-04_idsse-dead-ball-labels.md](2026-11-04_idsse-dead-ball-labels.md).

## TL;DR for the team

1. **The DFL flag goes alive at the restart.** 98 % of the 673 set pieces after a stoppage have a
   flagged dead-spell end within 5 s; median offset **+0.5 s**, IQR −0.5…+1.3 s. That spread is about
   what manually logged event times give, so: D ends on the restart touch.
2. *(Superseded for out-of-play starts by [2026-11-04_idsse-all-matches.md](2026-11-04_idsse-all-matches.md):
   measured against the tracked ball exit instead of event times, the flag goes dead a median
   0.16 s after the ball leaves the pitch. The fuzziness below is event-timing noise.)*
   **The start of D is fuzzier.** The flag goes dead a median **0.5 s before** the event-derived
   stoppage start (ball out / foul / offside / goal, else last play event), with a long early tail
   (q25 −2.5 s, q10 −5 s). Whistle vs ball-over-line, plus event timing noise.
3. **The flag covers more than set-piece stoppages.** 7 % of flagged dead spells have no set piece
   (injuries, drop balls, …). Flag D share 40.7–51.7 % per match; the event recipe gets 38.6 %, and
   frame IoU flag vs recipe is 79 %.
4. **Metrica vs DFL, same event recipe on both: same picture.** Stoppage durations by restart type
   agree to within a few seconds (table below). No sign of a gross convention difference (for
   example, Metrica not counting throw-ins as dead, or only counting whistle-to-restart).
5. **What we can't measure from other games:** a sub-second systematic offset between DFL's flag
   and Metrica's corrected-tracking flag. Event data is too noisy for that, and the only Metrica
   flag we have is BAR-ATH's public 5 min (~4 stoppages). → ask the organisers (draft below).

## Numbers

DFL flag dead share per match: 41.5, 51.7, 47.6, 46.5, 44.3, 40.7, 41.9 %.

| offset, s (q10 / q25 / q50 / q75 / q90) | value |
|---|---|
| flag dead-end − set-piece event | −1.78 / −0.46 / +0.53 / +1.33 / +2.03 (49.5 % within ±1 s) |
| flag dead-start − event stoppage start | −5.03 / −2.49 / −0.53 / +0.73 / +1.84 (61.2 % within ±2 s) |

Stoppage duration by restart type, s, median [q25–q75] (n):

| type | DFL flag | DFL event recipe | Metrica event recipe |
|---|---|---|---|
| throw-in | 13.4 [7.9–19.2] (304) | 12.0 [6.4–17.9] (305) | 10.1 [7.0–15.8] (109) |
| corner | 32.2 [28.4–35.6] (57) | 26.8 [21.1–32.0] (57) | 30.4 [24.2–35.0] (26) |
| free kick | 31.3 [18.6–42.6] (163) | 29.5 [18.1–41.2] (172) | 22.4 [11.8–38.0] (62) |
| goal kick | 22.1 [15.8–29.5] (119) | 20.1 [13.6–25.8] (119) | 26.8 [19.6–31.9] (47) |
| kick-off | 65.1 [58.8–75.8] (16) | 59.4 [55.8–73.9] (17) | 59.3 [54.5–65.6] (12) |

Metrica event-recipe dead share per game: 43.9, 42.2, 34.7 %.

## Caveats

- Event timestamps are logged by people and synced to tracking afterwards; ±1 s noise is expected,
  so these offsets bound the convention difference, they don't pin it down.
- The recipe misses stoppages that end without a set piece, and Metrica's sample games are 3
  anonymised matches from an unknown league. League style (quick free kicks, time-wasting) and
  convention are mixed together in the duration table.
- The Metrica column is event-derived, not Metrica's tracking flag; the GT comes from the flag.

## Implications

- **Train the D detector on the DFL flag as-is.** It already means "dead from ball out / whistle
  until the restart touch", which is what CHALLENGE.md describes. No shift applied.
- Expect ~0.5–1 s of irreducible disagreement at D starts. That pushes towards decoding that
  places boundaries well rather than towards squeezing frame accuracy.
- Looking at the public 5 min to understand label meaning is allowed (CHALLENGE.md: "the public
  ground truth is where you can see it"); measuring an offset there and shifting by it is tuning. Not
  doing that.

## Draft question for the organisers (GitHub issue, not posted yet)

> **Dead-ball boundaries in the ground truth**
>
> CHALLENGE.md says a frame is `D` where the corrected tracking marks the ball dead. To build a
> dead-ball detector on other games' data (DFL/IDSSE, which has its own alive/dead flag), we'd like
> to match your convention rather than guess it from the 5 public minutes:
>
> 1. When does the corrected tracking switch to dead: when the ball fully crosses the line, or at
>    the referee's whistle (fouls, offside)?
> 2. When does it switch back to alive: on the restart touch (throw-in release, kick of a free
>    kick / corner / goal kick / kick-off)?
> 3. Are there dead stretches without a restart event (injury stoppages, drop balls), and how are
>    drop balls labelled?
> 4. We watched the public 5 minutes to understand the D convention and kept general,
>    non-frame-linked notes. Is that acceptable?
>
> We are not asking about any specific moment of BAR-ATH, only the general rule.
