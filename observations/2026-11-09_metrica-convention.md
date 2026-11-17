# 2026-11-09 · Metrica's possession convention vs DFL's: switch cost 6 holds

Author: Maxim. Code: `dead_ball/metrica.py` (Metrica sample loader + event-recipe labels),
`analysis/16_metrica_convention.py`. Other games only: IDSSE / DFL (7 matches) and Metrica's public
sample games 1–3 (tracking + events). Nothing from BAR-ATH is read. Aggregates only. Follows
[2026-11-09_learned-team-logic.md](2026-11-09_learned-team-logic.md).

## TL;DR for the team

1. **The worry:** we tune on DFL's possession flag. It counts many very short possessions, and
   Metrica's "controlled possession" might count fewer. If so, we'd need a higher switch cost.
2. **How we checked:** Metrica's open data has no owner flag, so both providers go through the
   same lens. That lens is an event recipe:
   - **Dead ball:** from the stoppage to the set piece.
   - **Owner:** the team of the last on-ball event (pass, shot, set piece, recovery, ball lost),
     carried forward. Duels and loose balls keep the last team.

   Under the same recipe, any difference between DFL and Metrica is a difference between the
   providers, not an artefact of the recipe.
3. **Metrica is not stricter than DFL; if anything the opposite.** Under the same recipe:
   - the best switch cost on DFL is 12;
   - the best on Metrica is 6 when degraded like our tracking (4 with the ball as given);
   - Metrica has slightly fewer possessions per live minute (6.1 against 6.8), but the same
     share of very short ones (23 % against 21 % under 2 s).
4. **Decision rule, fixed before running:** change the cost only if 6 scores more than 0.02 below
   Metrica's best and Metrica's best differs from DFL's best. On Metrica, cost 6 is 0.000 below
   the best when degraded like our tracking and 0.003 below with the ball as given. **The cost
   stays at 6.**
5. **The model fits Metrica's convention well.** Trained on DFL only, it scores spellF1 **0.77**
   against Metrica's event possessions when degraded like our tracking; VC scores 0.52. That's
   higher than against DFL's own flag (0.67). Possession boundaries in Metrica's event data
   (recoveries, ball lost) line up with what positions show. DFL's sparse event feed has no
   tackle team and few recoveries, so the DFL recipe tracks DFL's flag poorly (spellF1 0.475).
6. **No submission change.**

## Numbers

Label statistics (live = not D under each label set's own dead mask):

| | DFL flag (7) | DFL event recipe (7) | Metrica event recipe (3) |
|---|---|---|---|
| possessions | 3061 | 2766 | 1040 |
| per live minute | 8.30 | 6.80 | 6.14 |
| share < 2 s | 30.0 % | 20.8 % | 23.0 % |
| share < 5 s | 60.5 % | 51.8 % | 49.5 % |
| length q10 / q25 / q50 / q75 / q90 (s) | 0.8 / 1.7 / 3.7 / 8.5 / 18.6 | 1.2 / 2.2 / 4.7 / 10.0 / 20.7 | 1.0 / 2.1 / 5.1 / 13.6 / 24.8 |

Learned model, mean spellF1 by switch cost. Each label set is decoded inside its own dead mask,
so this measures team logic only. IDSSE uses held-out fold models on the c = 0.5 copy. Metrica uses
the final model trained on all 7 IDSSE matches.

| labels | 0 | 2 | 4 | **6** | 8 | 12 | 16 | 24 | 32 | VC |
|---|---|---|---|---|---|---|---|---|---|---|
| IDSSE vs DFL flag | 0.508 | 0.657 | 0.664 | **0.666** | 0.665 | 0.660 | 0.651 | 0.632 | 0.610 | 0.480 |
| IDSSE vs DFL recipe | 0.382 | 0.523 | 0.537 | 0.547 | 0.552 | **0.558** | 0.557 | 0.552 | 0.542 | 0.398 |
| Metrica (ball as given) vs Metrica recipe | 0.612 | 0.796 | **0.808** | 0.805 | 0.803 | 0.788 | 0.779 | 0.763 | 0.754 | 0.787 |
| Metrica (ATD-like) vs Metrica recipe | 0.583 | 0.748 | 0.763 | **0.769** | 0.768 | 0.760 | 0.761 | 0.748 | 0.729 | 0.522 |

The best cost per match is 4–16 on Metrica (ATD-like: 16 / 8 / 6) and 8–24 on the DFL recipe.
On Metrica at cost 6, the model predicts 0.84 possessions per true one.

Learned model, per-frame H/A accuracy on Metrica recipe-live frames:

| | ball visible | ball hidden |
|---|---|---|
| ATD-like | 94.9 % | 79.9 % |
| ball as given | 95.7 % | 41.0 % |

ATD-like means our calibrated blackouts plus 0.25 s smoothing.

## Caveats

- **This compares Metrica's event data, not Metrica's tracking owner.** The ground truth comes
  from corrected tracking. We have no Metrica owner flag for any other game, so this is the
  closest available proxy.
- **3 anonymised games from an unknown league.** League style and convention are mixed together.
  The per-match best costs spread from 4 to 16.
- **Metrica's sample ball has its own gaps.** Where it's missing "as given", the model is right
  only 41 % of the time, which is below chance. These gaps aren't like ours: they're probably
  placed at restarts or where tracking failed, and the model hasn't seen gaps like them.
  Degraded like our tracking, the result is in line with IDSSE (80 % against 81 %).
- **Event times are logged by people** (±0.5–1 s). That costs boundary IoU equally across costs,
  so it doesn't move the best cost much.
