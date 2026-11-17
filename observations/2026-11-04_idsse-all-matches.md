# 2026-11-04 · IDSSE on all 7 matches + start of D by cause

Author: Maxim. Method: `analysis/05_idsse_all_matches.py` (aggregates only). Other games only
(IDSSE / DFL). Nothing from BAR-ATH is read. Follows
[2026-11-04_idsse-dead-ball-labels.md](2026-11-04_idsse-dead-ball-labels.md) (one match) and
[2026-11-04_dead-ball-conventions.md](2026-11-04_dead-ball-conventions.md) (event-based offsets).

## TL;DR for the team

1. **Going from 1 match to all 7 changes nothing material.** Pooled dead share is 44.8 % (per match
   40.7–51.7 %; J03WPY 44.3 %), with 713 dead spells, median 19.6 s (J03WPY 22.8 s). Data is clean
   in every match: ball present 100 %, 22 players, ball z non-zero ~55 %. Dead spells under 1 s:
   3 in total.
2. **The DFL flag follows our convention for D starts.** The team's general rule is: ball out =
   dead as soon as the ball's ground projection is off the pitch; foul = dead at the whistle.
   - **Out of play (68 % of dead spells): the flag goes dead a median 0.16 s (4 frames) after the
     ball centre crosses the line.** IQR +0.08…+0.24 s, 95.5 % within 0.5 s, the same in every
     match (medians +0.12…+0.16 s). Goal line is slightly tighter than touchline. This anchor is
     geometric (tracking, not human-logged), so the earlier "start is fuzzy, −0.5 s" result was
     event-timing noise, not the flag.
   - Only 1.0 % of ball-outside-pitch frames are still flagged alive.
   - **Fouls / offside (17.5 %): offset vs the DFL foul event is centred near the whistle**
     (median +0.58 s) but wide (IQR −0.37…+1.47 s). That spread is what logged event times give;
     there's no audio or whistle in IDSSE to pin it tighter. Consistent with "from the whistle",
     but it doesn't prove it.
   - Other (14.4 %): no exit or foul within −6…+3 s. Injuries, drop balls, goals, etc.
3. **Implication: train on the DFL flag as-is (no shift).** For out-of-play D starts, we can also
   derive a precise rule from our own ball track: D starts when the ball leaves the pitch, plus
   ~0.15 s. That only works where the ball is tracked through the exit. In IDSSE it always is; in
   our ATD the ball is missing ~41 % of the time, often around stoppages.

## Numbers

| match | dead share | dead spells | dead spell s, q50 | out-caused spells | median offset vs exit, s |
|---|---|---|---|---|---|
| J03WMX | 41.5 % | 103 | 15.5 | 82 | +0.16 |
| J03WN1 | 51.7 % | 107 | 23.8 | 75 | +0.16 |
| J03WOH | 47.6 % | 102 | 19.1 | 69 | +0.16 |
| J03WOY | 46.5 % | 113 | 19.2 | 79 | +0.12 |
| J03WPY | 44.3 % | 97 | 22.8 | 72 | +0.12 |
| J03WQQ | 40.7 % | 99 | 19.6 | 41 | +0.12 |
| J03WR9 | 41.9 % | 92 | 20.0 | 66 | +0.16 |
| **all** | **44.8 %** | **713** | **19.6** | **484** | **+0.16** |

Start-of-D offset (flag dead-start minus anchor, s, q10 / 25 / 50 / 75 / 90):

| anchor | n | quantiles | ≤ 0.2 s | ≤ 0.5 s | ≤ 1 s |
|---|---|---|---|---|---|
| ball centre crosses line | 484 | +0.04 / +0.08 / +0.16 / +0.24 / +0.32 | 68.6 % | 95.5 % | 98.1 % |
| ball fully over line (+0.11 m) | 457 | +0.04 / +0.08 / +0.12 / +0.20 / +0.32 | 71.6 % | 95.4 % | 97.8 % |
| touchline only | 304 | +0.08 / +0.08 / +0.16 / +0.25 / +0.36 | 61.5 % | 93.4 % | 97.0 % |
| goal line only | 180 | +0.04 / +0.08 / +0.12 / +0.16 / +0.32 | 80.6 % | 98.9 % | 100 % |
| foul / offside event | 124 | −1.46 / −0.37 / +0.58 / +1.47 / +2.34 | 10.5 % | 24.2 % | 41.1 % |

Pooled alive-spell duration q10…q90: 4.3 / 9.1 / 20.3 / 41.1 / 71.3 s; ~383 owner changes per
alive hour.

## Caveats

- Cause is assigned greedily: ball exit within −6…+3 s of the dead start wins; otherwise a foul or
  offside event in the same window. A foul right next to the touchline could be counted as "out".
- DFL pitch lines come from the match metadata pitch size. Line calibration error of a few cm would
  shift offsets by about a frame.
- The ball-in-air case (projection out but ball still above the line) is included in "out", as the
  team's rule says. We can't separate it here, since IDSSE z is noisy and our ATD has no z anyway.
- The +0.16 s is a DFL lag. We assume it's similar in Metrica's corrected tracking, but that
  isn't verified. Either way it's about 4 frames.
