# 2026-11-09 · Smart Tagging beyond the kickoff gate: checked, not adopted

Author: Maxim. Code: `analysis/18_restart_hints.py` (other games), `analysis/19_smart_tagging_on_atd.py`
(BAR-ATH, label-free), `possession/learned.py` (`decode(starts=...)`, `restart_starts`; off by
default). Aggregates only. Submission unchanged (sha256 `0add77a7…`).

## TL;DR for the team

1. **Restart-team hints don't help.** On held-out IDSSE (fold models, same setting as analysis/14) we
   gave the model the *true* restart team at the start of each predicted live stretch. Even perfect
   hints add nothing (0.601 → 0.601). The model already starts 95 % of those stretches with the right
   team, and the possessions it gets wrong are mid-play ones, not the ones after restarts. With tag
   errors (10 % / 25 % of hints flipped) the score drops in 7/7 matches.
2. **Rule fixed before the run:** adopt if corners + goal kicks + kickoffs at 25 % flips gave
   ≥ +0.003 and won ≥ 5/7. Result −0.005, 0/7. Not adopted.
3. **Smart Tagging on BAR-ATH agrees with what we have** (label-free):
   - set-piece team tags vs geometry (corner = team attacking that end, goal kick = team defending
     it; end from the ball): 19/22 agree (86 %), so roughly 1 tag in 7 may be wrong;
   - a predicted live stretch starts inside 96–100 % of corner, free-kick, kickoff and throw-in tags
     (80 % of goal kicks);
   - our team at that start = the tag on 21/25 team-tagged set pieces;
   - phases: our team = tagged team on 93 % of BUILD UP and 89 % of PROGRESSION live time. The
     DEFENDING codes mirror the attacking ones (same counts, same seconds), so they add nothing.
4. **Phase tags as a team prior were not tried:** they exist only for BAR-ATH, so their weight could
   be chosen only on BAR-ATH labels, which we don't use.

## Simulation (held-out IDSSE, switch cost 6, 7 matches × 3 copies)

| hints | flipped | stretches hinted | spellF1 | Δ | better |
|---|---|---|---|---|---|
| none | – | – | 0.601 | – | – |
| corner + goal kick + kickoff | 0 / 10 / 25 % | 24 % | 0.601 / 0.598 / 0.596 | −0.000 / −0.003 / −0.005 | 0/7 |
| + free kicks | 0 / 10 / 25 % | 43 % | 0.601 / 0.597 / 0.593 | −0.000 / −0.004 / −0.008 | ≤ 1/7 |
| + throw-ins (all) | 0 / 10 / 25 % | 81 % | 0.599 / 0.593 / 0.584 | −0.002 / −0.008 / −0.017 | 0/7 |

A hint is a true restart (DFL flag dead → alive, team on the first alive frame) spread over ±2.5 s,
matched to a predicted stretch start within 2 s of it.

## What Smart Tagging is used for

Only the opening-kickoff gate (`observations/2026-11-09_kickoff-detector.md`). Everything else is
measured here and left out.
