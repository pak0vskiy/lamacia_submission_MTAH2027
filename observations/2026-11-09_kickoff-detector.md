# 2026-11-09 · Kickoff detector: the 2nd-half pre-kickoff stretch is now all D

Author: Maxim. Code: `dead_ball/kickoff.py` (detector), `scripts/make_submission.py`
(`opening_kick`, on by default; `--no-kickoff` turns it off), `analysis/17_kickoff_detector.py`
(validation). Other games: IDSSE / DFL (7) and Metrica sample games 1–3. BAR-ATH: label-free only.
Aggregates only.

## TL;DR for the team

1. **The problem:** the 2nd half is graded from about 117 s before its kickoff (FAQ), and all of
   that stretch is D. Our dead-ball model called only about 58 % of it D, so ~49 s of the
   pre-kickoff came out as made-up possessions.
2. **The detector uses Law 8 and positions only.** At a kickoff, every player stands in his own
   half; in live play that never lasts.
   - **Formation:** the two teams are in opposite halves (2 m tolerance, 1 outlier player, gaps up
     to 1 s closed) for ≥ 3 s.
   - **Kick:** placed where the formation breaks and stays broken for 20 s, then moved earlier by
     a 0.6 s lag. The lag is the pooled median measured on other games.
3. **Validation on other games:**
   - Every testable kickoff was found: IDSSE 17 / 17, Metrica 15 / 15.
   - Every detection is within 1 s of the true kick. Median error is −0.08 s on IDSSE (truth: the
     DFL flag goes alive) and +0.24 s on Metrica (truth: the KICK OFF event).
   - **No false formations** in 369 minutes of IDSSE live play.
4. **Gate:** the detector would also find kickoffs after goals. In the 1st half, the graded window
   starts on the kick itself, so the first kickoff found there is after a goal. So the submission
   only accepts a detection inside Smart Tagging's automatic `Kick Off (Start)` interval for that
   half (±5 s). Smart Tagging is a provided algorithmic input, so this uses no frame constant.
5. **BAR-ATH, label-free:**
   - The 2nd-half kick is detected 117.5 s into the graded window, inside Smart Tagging's
     interval.
   - **Sanity only, not used for any choice:** that's +0.12 s from the FAQ's kick frame.
   - The 1st half has no in-window opening formation, as expected, so nothing is forced there.
6. **Submission:** 49.1 s of predicted live play before the kick is now D. Graded frames are H 33.2
   % · A 21.7 % · D 45.1 %, with 362 possessions (was 365). A from-scratch rebuild gives the same
   sha256 (`0add77a7…`). Public 5 min, **sanity only**: 0.674, unchanged, because that window
   doesn't contain the 2nd-half kick.

## How it was developed

- **First version:** 1 m tolerance, no outliers, no gap closing. Its timing was good, but it found
  only about half of the kickoffs. Players stand right on the line, so the runs kept getting
  chopped.
- **2 m formation tolerance plus 1 s gap closing:** Metrica went from 10 / 17 to 15 / 17 found.
- **One outlier allowed:** BAR-ATH's 2nd half always has at least one track on the wrong side of
  halfway, even at a 5 m tolerance. That's probably a mis-assigned track, not a player off the
  pitch. This is a generic data-quality allowance. On the other games it still gives zero false
  formations, and every testable kickoff is found.
- **Lag:** the strict 1 m test gives the timing. It still breaks a little after the kick: pooled
  median +0.64 s, so LAG_S = 0.6.

## Caveats

- **"Testable" means at least 5 s of stoppage tracked before the kick.** That leaves out 14 IDSSE
  cases where the tracking starts on the kick, and the 2 Metrica cases with a 0 s or 2 s
  stoppage before the kick (both match openers).
- **The gate depends on Smart Tagging tagging the opening kickoff.** If it misses one, nothing is
  forced and we fall back to the dead-ball model, which is the previous behaviour.
- **Kickoffs after goals are left to the dead-ball model.** Forcing D on formation runs everywhere
  could be checked on IDSSE later, but the D model already covers those long stoppages well.
