# Observations

Session findings for the team to discuss. One file per session/topic:
`YYYY-MM-DD_<topic>.md`, author at the top, TL;DR first.

Rules: aggregate numbers only (no rows, frames, screenshots). Public-GT numbers are labelled
"sanity only". Decisions that come out of a discussion go to `docs/decisions.md`.

| date | file | topic |
|---|---|---|
| 2026-11-02 | [2026-11-02_data-anatomy.md](2026-11-02_data-anatomy.md) | what each input is, how it helps H/A/D, first ATD aggregates |
| 2026-11-02 | [2026-11-02_ball-gaps.md](2026-11-02_ball-gaps.md) | ball blackouts: mostly stoppages? cue rates around gaps |
| 2026-11-09 | [2026-11-09_smart-tagging.md](2026-11-09_smart-tagging.md) | Smart Tagging restart-team hints: worth nothing even if perfect (model already right on 95 % of restarts), hurt with tag errors; not adopted. BAR-ATH tags agree with geometry 86 % and our calls 84–93 % |
| 2026-11-09 | [2026-11-09_kickoff-detector.md](2026-11-09_kickoff-detector.md) | Law-8 kickoff detector (positions only): 32/32 testable kickoffs on IDSSE + Metrica within 1 s, no false formations; gated by Smart Tagging, 2nd-half pre-kickoff now all D |
| 2026-11-09 | [2026-11-09_metrica-convention.md](2026-11-09_metrica-convention.md) | Metrica vs DFL possession convention through one event recipe: Metrica not stricter (best cost 6 vs DFL-recipe 12); learned 0.77 vs VC 0.52 on Metrica events; switch cost 6 kept |
| 2026-11-09 | [2026-11-09_learned-team-logic.md](2026-11-09_learned-team-logic.md) | learned P(home) + switch-cost Viterbi: held-out spellF1 0.601 vs VC 0.441, 7/7; BAR-ATH aggregates match IDSSE; now the submission's team logic |
| 2026-11-09 | [2026-11-09_error-anatomy.md](2026-11-09_error-anatomy.md) | where VC loses spellF1: short possessions (60 % of true, 20 % recall), missed opponent gains + false flips; boundary timing worth only ~+0.01 |
| 2026-11-05 | [2026-11-05_blackout-calibration-explained.md](2026-11-05_blackout-calibration-explained.md) | plain-language version: why and how we made the IDSSE practice data lose the ball like our match |
| 2026-11-05 | [2026-11-05_blackout-calibration.md](2026-11-05_blackout-calibration.md) | IDSSE blackouts fitted to BAR-ATH (live 10.6 % missing, not 31 %); D retrained 0.900; VC* 0.438 (was 0.374), zone 2.0 m; one-command submission |
| 2026-11-05 | [2026-11-05_vc-smoothed-ball.md](2026-11-05_vc-smoothed-ball.md) | our ATD ball is smoothed (σ 0.25 s); re-tuned VC on smoothed IDSSE: zone 2.5 m, direction-only; spellF1 0.374 (honest), still beats geometry/v1 7/7 |
| 2026-11-05 | [2026-11-05_vidal-codina-rules.md](2026-11-05_vidal-codina-rules.md) | the team-possession rules (player + team level, parameters); on BAR-ATH the ball-speed trigger never fires (smoothed ATD ball) |
| 2026-11-05 | [2026-11-05_team-possession-vidal-codina.md](2026-11-05_team-possession-vidal-codina.md) | team logic: Vidal-Codina rule beats v1 and Sam's geometry (spellF1 0.523 vs 0.279 / 0.331, our D); team logic is now the bottleneck |
| 2026-11-04 | [2026-11-04_dead-ball-conventions.md](2026-11-04_dead-ball-conventions.md) | DFL flag vs DFL / Metrica events: D ends on the restart; start fuzzier (~0.5 s); draft organiser question |
| 2026-11-04 | [2026-11-04_dead-ball-transfer.md](2026-11-04_dead-ball-transfer.md) | does the D model transfer to our ATD? players yes, ball no; chosen: no ball speed + realistic dead-ball motion |
| 2026-11-04 | [2026-11-04_dead-ball-classifier.md](2026-11-04_dead-ball-classifier.md) | tracking-only D classifier: model beats rules (spellF1 0.881 vs 0.836, DFL teams); DFL frozen-ball leak fixed |
| 2026-11-04 | [2026-11-04_idsse-all-matches.md](2026-11-04_idsse-all-matches.md) | all 7 IDSSE matches: stats stable; flag goes dead 0.16 s after ball leaves pitch; fouls ≈ whistle |
| 2026-11-04 | [2026-11-04_idsse-dead-ball-labels.md](2026-11-04_idsse-dead-ball-labels.md) | IDSSE has per-frame alive/dead + owner labels: usable for a D detector |
