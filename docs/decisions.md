# Decisions

Non-trivial decisions, newest first. One entry each: date, decision, why.

## 2026-11-09 · Smart Tagging restart-team hints not adopted

- Simulated on held-out IDSSE with true restart teams (`analysis/18_restart_hints.py`): perfect
  hints give +0.000 (the model already starts 95 % of hinted stretches right); with 10 % / 25 %
  flipped teams, −0.003 / −0.005, 0/7 matches better.
- Rule fixed before the run: adopt if corners + goal kicks + kickoffs at 25 % flips gave ≥ +0.003
  and won ≥ 5/7. Not met. `learned.decode(starts=...)` stays available, off.
- Phase tags are not used as a team prior: their weight could only be set on BAR-ATH labels.
  Label-free agreement on BAR-ATH is in `analysis/19_smart_tagging_on_atd.py`.

## 2026-11-09 · Opening kickoff: positions-only detector, gated by Smart Tagging, D before the kick

- `dead_ball/kickoff.py`: the two teams in opposite halves for ≥ 3 s, then the formation breaks
  for good. On other games it finds 32 of 32 testable kickoffs, all within 1 s (IDSSE median
  −0.08 s, Metrica +0.24 s), with no false formations in 369 live minutes
  (`analysis/17_kickoff_detector.py`).
- The submission accepts a detection only within ±5 s of Smart Tagging's automatic
  `Kick Off (Start)` interval. Without the gate, the 1st half (graded from the kick itself)
  would latch onto a kickoff after a goal. Everything in the graded window before the accepted
  kick is D.
- Fixes the known gap from 2026-11-05: the 2nd-half pre-kickoff went from ~58 % D to 100 %.
- No frame constants. The FAQ kick frame is shown as sanity only: the detection is +0.12 s from
  it.

## 2026-11-09 · Switch cost stays 6 after the Metrica convention check

- Same event recipe on DFL and on Metrica's sample games 1-3 (tracking + events, other games):
  the best switch cost is 12 under the DFL recipe and 6 under the Metrica recipe (ATD-like ball;
  4 as given). Metrica is not stricter than DFL. At cost 6, Metrica scores 0.000 below its best
  ATD-like and 0.003 below as given (`analysis/16_metrica_convention.py`).
- Rule fixed before running: move only if cost 6 is > 0.02 below Metrica's best and that best
  differs from the DFL-recipe best. Not met, so `learned.SWITCH_COST` stays 6 and the submission
  is unchanged.
- Metrica sample data is used for validation only, not training.

## 2026-11-09 · Team logic: learned P(home) + switch-cost Viterbi replaces Vidal-Codina

- `possession/learned.py`: gradient-boosted P(home) on positions-only, past-and-future features
  (VC's zone runs and gains included), trained on live frames of degraded and smoothed IDSSE with
  team-swap symmetry, decoded by a 2-state Viterbi with switch cost 6 inside predicted live stretches.
- Held-out IDSSE spellF1 0.601 vs VC 0.441 (better in 7/7, min +0.104); switch cost 6 chosen in 7/7
  folds (`analysis/14_learned_team_cv.py`). BAR-ATH aggregates (possession rate, length
  quantiles, confidence, feature ranges) match held-out IDSSE, label-free (`analysis/15`).
- `scripts/make_submission.py` uses it by default; `--team vc` keeps the old rule. Rebuild is
  bit-for-bit.
- Not adopted: constant boundary shifts (+0.006 at best, fitted to DFL's convention;
  `observations/2026-11-09_error-anatomy.md`).

## 2026-11-05 · Ball blackouts calibrated to BAR-ATH; D retrained on them; VC zone 2.0 m

- The old degradation (41 % of frames, coincidence c) hid 23-31 % of live play in 14-17 s gaps.
  BAR-ATH, measured label-free with our D model's calls, hides 10.6 % of live play: gaps sit on
  stoppages and end just before the restart, plus short live gaps (median 5.4 s) after duels.
- New default `blackout="calibrated"` (`dead_ball.cv.DEFAULT_BLACKOUT`) reproduces this: live
  missing 12.3 % vs 10.6 %, dead 70.7 % vs 76.7 %, duel ratio 1.7 vs 1.9
  (`analysis/12_blackout_calibration.py`). Constants in `dead_ball/degrade.py` (`CAL_*`).
- Re-scored on calibrated copies (held-out IDSSE): D model retrained on them 0.900 (setting A)
  vs 0.886 for the old D model; VC* 0.438 vs 0.431. VC zone re-picked 2.0 m in 7/7 folds
  (`TUNED`). Ranking unchanged: VC > geometry (0.347) ≈ v1 (0.347).
- The submission uses both. The coincidence mode stays available (`--blackout coincidence`).

## 2026-11-05 · One-command submission: final D model on all 7 IDSSE matches + VC TUNED

- `uv run python scripts/make_submission.py` builds the file end to end: tracking-only dead-ball
  model (dead_ball.cv defaults) fitted on all 7 IDSSE matches, with decoder thresholds from inner
  3-fold CV by match (picked enter 0.7 / exit 0.3 / min dead 3 s), then Vidal-Codina `TUNED`
  per live stretch. X outside the graded windows.
- One model fitted on all 7 matches rather than an average of the 7 fold models: same procedure
  as each CV fold, one decoder setting, simpler to reproduce.
- Bit-for-bit check: retraining from scratch gives the same sha256. Validator passes.
- Known gap: the 2nd-half pre-kickoff stretch (graded, all D per FAQ) comes out only ~58-65 % D.
  Fix it with a kickoff detector, not a frame constant.

## 2026-11-05 · Smooth the IDSSE ball like ATD's; VC re-tuned to zone 2.5 m

- Our ATD ball's frame-to-frame speed changes match DFL smoothed with a σ = 0.25 s Gaussian
  (`analysis/11_ball_smoothing.py`, label-free). Team logic is now tuned and scored on IDSSE
  with that smoothing (`dead_ball.degrade.smooth_ball`, `analysis/09 --ball-smooth 0.25`).
- Re-tuned VC leave-one-match-out: zone 2.5 m, angle 10°, single-frame steps
  (`possession.vidal_codina.TUNED`). The window-based speed trigger was tested and rejected.
- Expected spellF1 is 0.374, down from the raw-ball 0.523, and still above geometry (0.298)
  and v1 (0.279) in 7/7 matches (`observations/2026-11-05_vc-smoothed-ball.md`).

## 2026-11-05 · Team logic baseline: Vidal-Codina (2022) rule, ported, possession zone 1.0 m

- `possession/vidal_codina.py` ports databallpy 0.8.1's player-possession rule (MIT) instead of
  depending on databallpy. Installing it would downgrade numpy 2.5 → 2.2 and pandas 3.1 → 2.3,
  which would change our pinned environment and could shift existing results.
- On held-out IDSSE with our D model, it scores spellF1 0.523 against 0.331 for Sam's geometry
  decoder and 0.279 for v1, and wins in 7/7 matches
  (`observations/2026-11-05_team-possession-vidal-codina.md`).
- Parameters were chosen leave-one-match-out on other games only: pz 1.0 m, speed jump 5 m/s,
  angle 10°. Nothing was tuned on BAR-ATH.

## 2026-11-04 · Watch-only D-flag playground over the public GT window

- Built a local (127.0.0.1) viewer (not part of this repository) to step through the public 5 minutes with
  the GT label, tracked ball and sound, to learn the GT's dead-ball convention (whistle vs ball
  over the line; restart touch).
- Allowed by RULES.md §1 ("watching the video ... to understand them") and CHALLENGE.md ("the
  public ground truth is where you can see it").
- Guard rails: notes are free text with no frame links; nothing from the tool feeds the pipeline;
  no offset, threshold or model choice is measured on this window. The resulting convention is
  checked on DFL/IDSSE games and asked of the organisers.
