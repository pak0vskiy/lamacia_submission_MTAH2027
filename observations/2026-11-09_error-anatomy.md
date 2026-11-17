# 2026-11-09 · Where VC loses spellF1: error anatomy and boundary timing (held-out IDSSE)

Author: Maxim. Method: `analysis/13_error_anatomy.py`. It uses the same held-out copies, D fold
models, calibrated blackouts and 0.25 s ball smoothing as `analysis/09`, with team logic VC
`TUNED`. Scored on 7 matches × 3 copies. Aggregates only. Other games only; nothing from BAR-ATH
is read.

## TL;DR for the team

1. **Boundary timing is not the problem.** Moving boundaries buys about +0.01 at most. Team
   switches 6 frames later gives +0.006 (chosen in 7/7 folds, better in 7/7 held-out matches).
   D ends 12 frames earlier gives +0.003, but not consistently.
2. **Short possessions are the problem.** 60 % of true possessions last under 5 s, and we catch
   only 20 % of them. Of the possessions of 5 s or more we catch 70 %.
3. **VC misses real changes of possession and invents false ones at the same time.**
   - Missed: 129 of 437 true possessions per match are short opponent spells that VC never sees
     (median 2.1 s). Another 61 are true possessions merged across such a missed switch.
   - Invented: 85 false flips per match (median 1.8 s), where VC hands the ball to the wrong team
     for a moment.

   That pattern means the gain trigger itself can't tell real gains from fake ones. No threshold
   change fixes both at once, which supports per-frame evidence plus a decoder with a switching
   cost (improvement #3).
4. **Ball blackouts explain only a small part.** Only 15 % of true possessions start with the ball
   hidden. Their recall is 0.18 against 0.44 when the ball is visible, which accounts for about 17
   missed possessions per match out of about 260.
5. **D costs about 0.04, as before.** With the true (oracle) D, recall rises from 40 % to 44 %, and
   the error breakdown hardly changes.
6. **Convention risk is now concrete.** Under DFL labels, 30 % of true possessions last under 2 s.
   Metrica's "controlled possession" rule (deflections and touches without control don't count)
   probably removes many of them. Chasing them could help on IDSSE and hurt on BAR-ATH, so we
   should check on Metrica's sample games (improvement #4) before optimising for them.

## Numbers (D = our model, mean per match)

True possessions by outcome at IoU 0.5 (437 per match; we predict 350):

| outcome | per match | share | median length |
|---|---|---|---|
| matched | 175 | 40 % | 8.8 s |
| other team holds (missed opponent gain) | 129 | 29 % | 2.1 s |
| merged across a missed switch | 61 | 14 % | 3.0 s |
| boundary (one same-team spell, IoU < 0.5) | 33 | 7 % | 2.0 s |
| predicted dead | 31 | 7 % | 1.7 s |
| split | 9 | 2 % | 8.4 s |
| merged across D | 1 | 0.3 % | 6.0 s |

Spurious predicted possessions (175 per match):

| cause | per match | median length |
|---|---|---|
| false flip (truth is the other team) | 85 | 1.8 s |
| boundary | 40 | 7.2 s |
| fragment of a longer true spell | 39 | 1.2 s |
| inside a true stoppage | 11 | 5.9 s |

Recall by true possession length:

| length | < 2 s | 2–5 s | 5–10 s | 10–20 s | ≥ 20 s |
|---|---|---|---|---|---|
| share of true | 30 % | 31 % | 19 % | 12 % | 9 % |
| recall @0.5 | 0.12 | 0.29 | 0.54 | 0.79 | 0.91 |

Boundary offsets (pred − truth, frames; matched within ± 2 s):

| boundary | median | q10 / q90 | within ± 5 fr | recall @2 s | pred / true count |
|---|---|---|---|---|---|
| D start | 0 | −15 / +28 | 47 % | 0.70 | 0.88 |
| D end | +4 | −15 / +26 | 35 % | 0.74 | 0.88 |
| team switch | −3 | −24 / +26 | 42 % | 0.47 | 0.77 |

- Timing is spread out but not biased. The median offsets are within 4 frames, which costs
  almost nothing (3 frames late still scores 0.98).
- The spread is where the cost is, and a constant shift can't remove it.

Shift sweeps (mean held-out spellF1; baseline 0.441). Each held-out match uses the shift chosen on
the other 6:

| moved boundary | best shift | held-out gain | chosen in |
|---|---|---|---|
| team switch | +6 fr (later) | +0.006 (min +0.001) | 7/7 |
| D end | −12 fr (earlier) | +0.003 (min −0.003) | 7/7 |
| D start | 0 | 0 | 7/7 |

With DFL's own owner (setting A, baseline 0.896), the D-edge shifts are worth ≤ 0.002. The D
model's timing is already fine.

## Caveats

- The causes come from a simple rule: whichever single label covers ≥ 50 % of the spell, checked
  in a fixed order. They're approximate but add up correctly.
- The +6 frame switch shift is a constant chosen leave-one-match-out on other games, so it's
  allowed. But it was fitted to DFL's boundary convention, and Metrica may timestamp a gain
  differently. It's too small to adopt before the convention check.
- Everything uses the DFL possession convention. The ranking of causes is what carries over to our
  match; the absolute numbers don't.

## Next

1. Per-frame P(home has the ball) from a learned model (Sam's `owner_model.py`, retrained per fold
   on calibrated + smoothed copies, with VC's triggers as features), then decoding with a
   switching cost inside each live stretch. Target: fewer false flips and fewer missed opponent
   gains.
2. In parallel, check the convention on Metrica's sample games: how many sub-2 s possessions does
   a "controlled" rule keep? That decides how hard to chase them.
