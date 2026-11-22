# M7 race model: experiment log

## Protocol

- **The test set is sealed.**
  - It is 2025 from Budapest (3 Aug) to Abu Dhabi: 11 races, fixed by date (`race_model.validation_until`,
    `test_until`).
  - While `race_model.test_sealed` is true, no stage reads a test race's raw files: training, validation,
    `m7-update` and forecasts all skip them. `tests/unit/test_race_model_isolation.py` enforces this with a
    catalog that fails on any read.
  - The only way in is `m7-evaluate --split test --unseal-test`. It writes `test_ledger.json` before reading
    anything and is refused a second time.
- **Every pooled quantity is fitted on training races only (2023–24, 46 races):** φ, σ_ε, tyre model,
  season-filter hyperparameters, event hazards, race distances, the transition baseline and the noise
  multiplier.
- **Validation is the first half of 2025, up to Spa: 12 races.** Its races still update the season filter
  online, the way a live forecaster would.
- **Decisions are made on validation.**
  - The score is mean RPS per running car at k = 10/25/50/75/90% of distance.
  - Each change is compared with a race-paired bootstrap 95% CI against the transition baseline.
  - A change is kept if it improves overall validation RPS without making any bucket clearly worse.
- **Known exposure before sealing:**
  - The data-quality audit and its cleaning rules looked at all of 2023–25.
  - One early training log printed a test prior-check summary (pace prior RMSE and z² only, no forecast
    scores). Neither has been used to make a decision.

## Results (validation, 12 races)

| # | Change | RPS overall | RPS 10% | 25% | 50% | 75% | 90% | Pairwise LL | Δ vs transition [95% CI] | Kept |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Sealed baseline (φ, σ_ε from train only; noise × 2.0) | 0.0800 | 0.1016 | 0.0978 | 0.0868 | 0.0667 | 0.0467 | 0.370 | −0.0020 [−0.0093, +0.0054] | — |
| 1 | Noise multiplier per race phase (chosen on training: ×1.0 / 2.0 / 3.0 / 3.0 / 3.0) | 0.0796 | 0.1018 | 0.0978 | 0.0860 | 0.0661 | 0.0461 | 0.367 | −0.0024 [−0.0095, +0.0048] | yes (marginal) |
| 1b | Reference after the sampler refactor (shared random draws; same model as 1) | 0.0796 | 0.1018 | 0.0980 | 0.0859 | 0.0662 | 0.0461 | 0.367 | −0.0024 [−0.0095, +0.0049] | reference |
| 2 | Telemetry clean-lap filter: full-throttle share > 0.06 below the driver's race median | 0.0797 | 0.1024 | 0.0990 | 0.0859 | 0.0655 | 0.0458 | 0.368 | −0.0022 [−0.0097, +0.0051] | no |
| 2b | Same filter at 0.04 | 0.0792 | 0.1018 | 0.0982 | 0.0852 | 0.0652 | 0.0456 | 0.365 | −0.0027 [−0.0101, +0.0045] | marginal |
| 3 | Filter 0.06 + overtaking friction (2.0 s/place × circuit difficulty, chosen on training) | 0.0769 | 0.0965 | 0.0951 | 0.0829 | 0.0651 | 0.0447 | 0.359 | −0.0051 [−0.0106, +0.0009] | see 4–5 |
| 4 | Filter 0.04 + friction | 0.0766 | 0.0959 | 0.0945 | 0.0824 | 0.0651 | 0.0453 | 0.358 | −0.0053 [−0.0108, +0.0005] | no (see note) |
| 5 | Friction only, no telemetry (2.0 s/place × difficulty) | 0.0769 | 0.0966 | 0.0942 | 0.0821 | 0.0659 | 0.0457 | 0.358 | −0.0050 [−0.0102, +0.0007] | no: reverted (owner) |
| 6a | Track A — Unclipped `noise_scales` [0.5..10.0] (train picks 0.5/1.5/3/3/3) | 0.0796 | 0.1018 | 0.0980 | 0.0859 | 0.0662 | 0.0461 | 0.3667 | −0.0023 [−0.0096, +0.0049] | no |
| 6b | Track A — Student-t lap noise (`df=4 / 6 / 10`, unit-variance) | 0.0796 | 0.1018 | 0.0980 | 0.0859 | 0.0660 | 0.0462 | 0.3666 | −0.0024 [−0.0096, +0.0050] | no |
| 6c | Track A — Phase-wise LOO transition shrinkage (`predict_running`, `pairwise_ahead_conditional`) | **0.0740** | **0.0945** | **0.0924** | **0.0827** | **0.0616** | **0.0386** | **0.3458** | **−0.0080 [−0.0129, −0.0031]** | **yes** |
| 7a | Track B — Overtaking friction with all stops but no tyres in `Z_ref` (`linear_rank` / `queue`) | 0.0745 | 0.0931 | 0.0917 | 0.0815 | 0.0696 | 0.0367 | 0.3477 | −0.0074 [−0.0142, −0.0006] | no (75% hurt) |
| 7b | Track B — Overtaking friction against **owed-stops-only** equal-pace `Z_ref` (`c=2.5`, $\kappa=200$, zero telemetry) | **0.0730** | **0.0942** | **0.0907** | **0.0838** | **0.0593** | **0.0369** | **0.3389** | **−0.0090 [−0.0153, −0.0022]** | **yes** |
| 7c | Track B — Team partial-pooled DNF hazard ($\kappa=2000 / 7034$) | 0.0798 | 0.1020 | 0.0982 | 0.0861 | 0.0664 | 0.0462 | 0.3673 | −0.0022 [−0.0096, +0.0051] | no |
| 7d | Track B — Level-2 `circuit_type` / `location` circuit-affinity carryover | — | — | — | — | — | — | — | Val Prior RMSE 0.4131 / 0.4190 | no |
| 8a | Track C — Mixed-weather dry-tyre `plan_stops` fix (`owed = not wet & <2`) | 0.0793 | 0.1031 | 0.0986 | 0.0836 | 0.0649 | 0.0461 | 0.3657 | −0.0027 [−0.0098, +0.0046] | no (dropped in 9) |
| 8b | Track C — Training-fitted `wet_noise_multiplier` (`1.3915×` green-lap MAD ratio) | 0.0795 | 0.1017 | 0.0977 | 0.0853 | 0.0661 | 0.0466 | 0.3664 | −0.0025 [−0.0096, +0.0048] | yes |
| 8c | Track C — Analytic 2-stop plan + 3-stint `tyre_upto` (0 new params) | 0.0794 | 0.1018 | 0.0968 | 0.0859 | 0.0662 | 0.0461 | 0.3664 | −0.0026 [−0.0098, +0.0047] | yes |
| 8d | Track C — Combined strategy (`8a + 8b + 8c`) | 0.0791 | 0.1030 | 0.0971 | 0.0835 | 0.0651 | 0.0466 | 0.3656 | −0.0029 [−0.0099, +0.0040] | partial (`8b + 8c`) |
| 8e | Track C — Opportunistic SC stops (`0.0798`), wet SC stops (`0.0796`), 2nd SC (`0.0795`), SC-DNF coupling (`0.0798`) | 0.0798 | 0.1018 | 0.0980 | 0.0866 | 0.0666 | 0.0461 | 0.3683 | −0.0021 [−0.0096, +0.0054] | no |
| 9a | Combined — Track A + Track B (`6c + 7b`, no Track C) | 0.0710 | 0.0925 | 0.0876 | 0.0810 | 0.0581 | 0.0357 | 0.3313 | −0.0110 [−0.0159, −0.0056] | sup. by 9d |
| 9b | Combined — Track B + Track C full (`7b + 8a + 8b + 8c`, no shrinkage) | 0.0725 | 0.0950 | 0.0900 | 0.0826 | 0.0582 | 0.0368 | 0.3383 | −0.0094 [−0.0156, −0.0027] | no |
| 9c | Combined — Track A + Track B + Track C full (`6c + 7b + 8a + 8b + 8c`) | 0.0709 | 0.0934 | 0.0874 | 0.0804 | 0.0574 | 0.0358 | 0.3316 | −0.0110 [−0.0161, −0.0057] | no (`8a` hurts 10%/25%) |
| **9d** | **Final Combined — Track A + Track B + Track C (`6c + 7b + 8b + 8c`, promoted `20261122T205928432773Z`)** | **0.0708** | **0.0924** | **0.0867** | **0.0810** | **0.0580** | **0.0358** | **0.3307** | **−0.0112 [−0.0161, −0.0059]** | **YES (final)** |
| — | Baselines: persist / transition | 0.1160 / 0.0819 | 0.160 / 0.100 | 0.157 / 0.106 | 0.137 / 0.097 | 0.081 / 0.065 | 0.046 / 0.043 | — / 0.431 | | |

Baseline 0 details:
- φ = 0.414 and σ_ε = 0.413, from training races only. Pooled over all races, φ had been 0.470.
- Prior check on validation: RMSE 0.41, z² 0.98.
- Brier scores: win 0.026, podium 0.047, points 0.112.

## Exploration A (2026-11-19): racing lines from OpenF1 `location`

`location` (~4 Hz x/y/z per car) was downloaded for the 58 training and validation races (7.6 GB; test races
not fetched). The code is in `race_model/lines.py`; nothing in it feeds the forecast. Figure:
`docs/figures/sakhir-2023-passes.svg`.

- **Racing lines cannot be inferred: positions are snapped to one line per circuit.**
  - Coordinates are decimetres. A reference line from the 10 fastest clean laps gives Sakhir 5,339 m against
    the official 5,412 m.
  - Across all 20 cars, any 20 m of track has a 10–90% lateral spread of 9.5 cm.
  - Per-driver offsets at turn 1 agree within 2 cm. The only real offset is the pit lane, about 15 m.
  - So there is no information on lines, defending, mistakes or side-by-side running, and a spatial clean-lap
    filter is not possible.
- **What the data does give: 1-D progress.**
  - On-track passes are detected as swaps in unwrapped race distance, with four filters: green laps after
    lap 1, no in- or out-laps (pit-lane points are often snapped onto the track), within half a lap, and the
    swap held for 10 s (GPS loss in the Monaco tunnel).
  - Against net order changes at the timing line: precision 0.80, recall 0.65, median 22.5 against 30 passes
    per race.
  - Counting passes needs only the timing line. What `location` adds is where passes happen: 41% within
    400 m after a heavy-braking zone.
- **Circuit profiles are reliable.**
  - The profile covers lap length, full-throttle share, longest straight, braking zones, heavy-braking zones
    and corners.
  - It is stable across years at the same circuit: median spread 3 m in length and 26 m in the longest
    straight, with identical zone and corner counts.
- **Does a circuit's layout predict how much passing happens?**
  - The target is line passes per green lap. It correlates with the longest straight (0.60) and lap length
    (0.53).
  - Leave-one-circuit-out on training, lap length alone cuts squared error by 17% against the field average.
  - On the 9 dry validation races, RMSE of each race's passing rate: profile 0.35, field average 0.42, that
    circuit's own 2023–24 rate 0.46. Layout is a better guide than history, but on a small sample.
- **Not used in the model.** It would matter only with an overtaking component, and experiments 3–5
  (friction) were reverted. A possible use is a difficulty prior for circuits with no history, such as 2026
  newcomers.

## Notes per experiment

**1. Noise multiplier per race phase.**

- *Hypothesis:* the 75–90% buckets lose to the transition baseline because rows are over-dispersed, so late
  forecasts need less lap noise.
- *Result:* the opposite. On training races the best multiplier rises with k: ×1.0 at 10%, ×2.0 at 25%, and
  ×3.0 from 50% on. ×3.0 is the top of the grid, so the optimum may be higher.
- *Validation:* −0.0004 RPS overall. Late buckets are 0.0006–0.0008 better and 10% is flat. That is within
  noise. It is kept under the rule (no bucket clearly worse).
- *What it tells us:* the late-race gap to the transition baseline does not come from too much lap noise.
- *Candidates next:*
  - extend the grid above ×3;
  - overtaking friction (math note §8, option B);
  - the second-stop and tyre-cliff assumptions in the stop plan.

**2. Telemetry clean-lap filter** (`race_model/telemetry.py`; `car_data` summarised per driver-lap and cached).

- *Training-only study* (25,482 clean laps, all with telemetry):
  - Laps whose full-throttle share sits more than 0.06 below the driver's own race median run +1.9% to +5.4%
    slow against the pace model. That is 2.4% of clean laps.
  - Laps 0.03–0.06 below run +0.9% slow, too little for the 3-MAD residual trim to catch.
  - corr(residual, full-throttle deviation) = −0.48. The coasting share carries no signal (corr 0.006).
- *Effect on level 1:* φ rises from 0.414 to 0.447 and σ_ε falls from 0.413 to 0.377. Cleaner laps are less
  noisy.
- *Validation:* no gain at 0.06 (+0.0001) and a marginal gain at 0.04 (−0.0004). The pace prior is unchanged
  (RMSE 0.41). The filter removes noise the forecast was already averaging over.
- Telemetry is never simulated forward. It only describes laps already run. Live forecasts have no telemetry
  (OpenF1 needs authentication during sessions), so live keeps every lap.

**3. Overtaking friction** (`race_model/passing.py`; math note §8, towards option B).

- *Passing study, training only* (6,971 attacks, where an attack is a follower within 1.0 s at the line on a
  green, non-pit lap):
  - The pass rate is 12.8% per attack.
  - By gap: 46% under 0.3 s, 19% at 0.3–0.6 s, 6% at 0.6–1.0 s.
  - By the follower's pace advantage: 3% when slower, 31% when more than 0.5% faster.
  - DRS open: 13% against 5% closed.
  - Circuit difficulty (pooled rate / circuit rate, shrunk) runs from 0.60 at Barcelona to 2.95 at Monaco.
- *Sampler:* each scenario projects the order the race would have with equal pace and no noise. That
  projection keeps stops, tyres and the safety-car queue, so pit-cycle changes stay free. Every place gained
  against it costs `pass_cost × difficulty(circuit)` seconds.
- *Training choice:* 2.0 s per place (RPS 0.0629 at 0, 0.0608 at 2.0, 0.0615 at 4.0).
- *Validation:* −0.0027 RPS against 1b, with every bucket better. The largest gains are early (10%:
  0.1018 → 0.0965) and in mid-race. The model now also beats the transition baseline at 10%. Overall
  Δ vs transition is −0.0051 [−0.0106, +0.0009], almost significant with 12 races.
- Win Brier improves from 0.027 to 0.025, and points Brier from 0.113 to 0.106.
- Experiment 3 mixes friction with the 0.06 filter, which alone did nothing. Experiments 4 (friction + 0.04
  filter) and 5 (friction, no telemetry) separate the two.

**4–5. Attribution and decision.**

- Friction without telemetry (5) scores 0.0769, the same as 3. With the 0.04 filter (4) it scores 0.0766.
- So the gain is friction: −0.0027 against 1b. The filter adds −0.0003, which matches 2b alone.
- That −0.0003 is mixed across buckets: 25% and 50% get slightly worse. It would also make live forecasts,
  which have no telemetry, differ from training.
- *Initial call:* keep 5, with friction on (training picked 2.0 s in both 4 and 5) and the telemetry filter
  off.
- **Final decision (project owner, 2026-11-19): revert 2–5.** The gain (−0.0027 RPS, CI including 0 against
  the transition baseline) does not justify the added machinery. The code was removed. The model is
  experiment 1.
- The passing difficulty index needs no telemetry: DRS is only reported, not used.
- **Still open (resolved in Experiments 6–8 below):**
  - 75% and 90% remain level with, or behind, the transition baseline.
  - The friction cost is one number; a version that grows with the gap and pace advantage seen in the study
    is the next step towards option B.
  - The noise grid still tops out at ×3.

**6. Track A — Calibration, Noise & Phase-Wise LOO Transition Shrinkage (`exp/m7-calibrator`, commit `7714101`).**

- *6a. Unclipping `noise_scales` above 3.0 (`[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]`):*
  - On training races, `3.0` is already the interior optimum at 50%, 75%, and 90% (`4.0` and `5.0` degrade training RPS in all three buckets). Moving 10% from `1.0` to `0.5` yields zero validation gain (`0.0796`). **Discarded.**
- *6b. Heavy-tailed Student-t lap noise (`df ∈ {3, 4, 6, 10}`, unit-variance scaled):*
  - Replacing Gaussian lap noise in `forecast.py` with unit-variance Student-t draws is flat on training and validation (`0.0796–0.0797`). **Discarded.**
- *6c. Phase-wise convex shrinkage toward the LOO training transition baseline (`TransitionBaseline.predict_running` + `pairwise_ahead_conditional`):*
  - Fits convex weights $w_k \in [0, 1]$ in closed form by minimising training RPS of $P_{\text{blend}}(k) = (1 - w_k) P_{\text{model}}(k) + w_k P_{\text{transition}}(k)$ over the 40 post-burn-in training races (2023–24) using exact leave-one-race-out (LOO) transition counts so $P_{\text{transition}}$ has zero in-sample bias (`w = {0.1: 0.3466, 0.25: 0.2635, 0.5: 0.3707, 0.75: 0.2842, 0.9: 0.4777}`).
  - `TransitionBaseline.predict_running` applies Sinkhorn strictly on the $M$ running cars and locks the $N - M$ already-retired cars at their deterministic bottom slots (preserving exact double stochasticity without assigning retired cars nonzero win/points mass). `pairwise_ahead_conditional` computes $\Pr(\text{pos}_i < \text{pos}_j \mid \text{pos}_i \ne \text{pos}_j)$, restoring exact pairwise complementarity $A_{ij} + A_{ji} = 1$.
  - *Validation:* **0.0740 RPS** (`−0.0056` vs 1b reference, `Δ vs transition = −0.0080 [−0.0129, −0.0031]`), beating both baseline and `transition` in **every single bucket (10%–90%)**, cutting Pairwise LL to `0.3458` (`−0.0209`), and improving Brier scores (`win 0.0230`, `podium 0.0422`, `points 0.1044`, `dnf 0.0390`). **Kept.**

**7. Track B — Timing-Line Overtaking Friction & Level-2/DNF Covariates (`exp/m7-racecraft`, commit `7823947`).**

- *7a–7b. Timing-line overtaking friction against an owed-stops-only equal-pace reference (`events.py`, `forecast.py`, `runner.py`):*
  - Circuit passing difficulty is estimated strictly on 2023–24 training timing-line gaps ($\le 1.0\text{ s}$ on green, non-pit laps; `8,529` attacks, `1,088` passes, pooled pass rate `12.76%`, empirical-Bayes pooling $\kappa = 200$, ranging from `0.598` at Barcelona to `2.827` at Monaco) with **zero telemetry (`car_data`) dependencies** (100% live-OpenF1 compatible).
  - *Key physical diagnosis:* In Experiment 5 (`0.0769`), `Z_ref` included tyre degradation and voluntary pit stops. Including tyre degradation in the equal-pace reference `Z_ref` gave fresh-tyre cars free passes in `Z_ref`, suppressing friction at 75%–90%. Conversely, removing tyres from `Z_ref` while keeping voluntary stops (`7a`, `0.0745`) falsely dropped voluntary-stop leaders by ~22 s in `Z_ref` at 75%.
  - *Solution (`7b`):* `Z_ref` keeps **only mandatory owed stops (`owed`) and SC queues** (`include_tyres=False`, voluntary stops excluded). Every on-track place gained against `Z_ref` costs `pass_cost_seconds × pass_difficulty(location)` seconds (`pass_cost_seconds = 2.5 s`, chosen on training races from `[0.0, 1.0, 1.5, 2.0, 2.5, 3.0]`, with phase noise multipliers `[1.0, 2.0, 3.0, 3.0, 3.0]`).
  - *Validation:* **0.0730 RPS** (`−0.0066` vs 1b reference, `Δ vs transition = −0.0090 [−0.0153, −0.0022]`), beating both baseline and `transition` in **every single bucket** (`10%: 0.0942`, `25%: 0.0907`, `50%: 0.0838`, `75%: 0.0593`, `90%: 0.0369`), cutting Pairwise LL to **0.3389** (`−0.0278`), improving Brier scores (`win 0.0236`, `podium 0.0372`, `points 0.1072`, `dnf 0.0390`), and beating `transition` in **10/12 validation races** (Monaco `0.0896 → 0.0774`, Spa `0.0733 → 0.0596`, Silverstone `0.0904 → 0.0778`, Montréal `0.1075 → 0.0936`). **Kept.**
- *7c. Team partial-pooled DNF hazard:* Slightly improved DNF Brier (`0.0390 → 0.0386`) but hurt overall RPS (`0.0798–0.0800`) and Pairwise LL (`0.3673–0.3684`) across all buckets. **Discarded.**
- *7d. Level-2 `circuit_type` / `location` circuit-affinity carryover:* `circuit_type` shrank to exact zero on training MLE (`s_ct = 0.000`, `ΔLL = 0.00`), and `location` carryover overfit (`ΔLL = +2.19` for 3 parameters, worsening validation prior RMSE from `0.4131` to `0.4190`). **Discarded.**

**8. Track C — Wet-Weather Regime, Multi-Stop & SC Strategy (`exp/m7-strategy`, commit `e8de124`).**

- *8a. Mixed-weather dry-tyre `plan_stops` fix (0 new parameters):* Standalone, setting `owed = (not wet) and len(set(car.used)) < 2` cut Spa validation RPS (`0.0733 → 0.0653`) and overall RPS (`0.0796 → 0.0793`), while slightly nudging 10% (`0.1018 → 0.1031`). When combined with Track A + Track B (`9c`), it degraded 10% (`0.0924 → 0.0934`), 25% (`0.0867 → 0.0874`), overall RPS (`0.0708 → 0.0709`), Pairwise LL (`0.3307 → 0.3316`), and Melbourne (`0.0971 → 0.0993`). **Dropped in final integration (`9d`).**
- *8b. Training-fitted `wet_noise_multiplier` (`1.3915×`):* Ratio of median wet vs dry green non-pit lap residual MAD on 2023–24 training races (`0.719% / 0.517%`), clipped to `[1.0, 3.0]`. Improves wet validation races and adds clean gain on top of `A+B+8c` (`0.07082 → 0.07079`, Win Brier `0.0213 → 0.0212`, Points Brier `0.1027 → 0.1025`, Melbourne `0.0981 → 0.0971`). **Kept.**
- *8c. Analytic 2-stop plan + 3-stint `tyre_upto` (0 new parameters):* Closed-form 2-stop split laps $(t_1, t_2)$ chosen when `cost2 < cost1`, eliminating the 25%-phase asymmetry on high-degradation races where early pitters could schedule a 2nd stop while non-pitters could only schedule 1 stop. On top of `A+B` (`9a → 9d`), cuts 25% RPS (`0.0876 → 0.0867`), Pairwise LL (`0.3313 → 0.3307`), Barcelona (`0.1116 → 0.1111`), Shanghai (`0.0851 → 0.0845`), and Spielberg (`0.0456 → 0.0451`) with zero bucket regressions (voluntary 2nd stops stay excluded from `Z_ref` when `include_tyres=False`). **Kept.**
- *8e. Opportunistic SC stops (`0.0798`), wet SC stops (`0.0796`), 2nd SC (`0.0795`), and SC-retirement coupling (`0.0798`):* Either degraded 50%/75% buckets or added complexity without net validation gain. **Discarded.**

**9. Final Integration & Combined Ablation (`feat/m7-race-model-improvements`, promoted version `20261122T205928432773Z`, MLflow `models:/f1-race-position-model/2`).**

- *Strong synergy between Track A (LOO shrinkage) and Track B (owed-only friction):* Combining Track A + Track B (`9a`) drops validation RPS from `0.0730` (Track B alone) and `0.0740` (Track A alone) down to **`0.0710`**, improving every single distance bucket (`10%–90%`), Pairwise LL (`0.3313`), and all Brier scores (`win 0.0215`, `podium 0.0368`, `points 0.1028`). Once physical overtaking friction is present, the training-fitted LOO shrinkage weights shrink naturally (`w = {0.1: 0.150, 0.25: 0.226, 0.5: 0.302, 0.75: 0.097, 0.9: 0.367}` vs `{0.347, 0.264, 0.371, 0.284, 0.478}` without friction).
- *Adding Track C sub-components (`9c` vs `9d`):*
  - Adding `8b + 8c` (`wet_noise_multiplier = 1.3915×` + analytic 2-stop plan) reaches **`0.0708` overall RPS (`0.070787`)**, **Pairwise LL `0.3307`**, **Δ vs `persist` `−0.0452 [−0.0580, −0.0315]`**, and **Δ vs `transition` `−0.0112 [−0.0161, −0.0059]`**, with `w = {0.1: 0.1401, 0.25: 0.2194, 0.5: 0.3052, 0.75: 0.0958, 0.9: 0.3626}`, `pass_cost_seconds = 2.5 s`, and phase noise multipliers `{0.1: 1.0, 0.25: 2.0, 0.5: 3.0, 0.75: 3.0, 0.9: 3.0}`.
  - Adding `8a` on top (`9c`, `0.0709` RPS, PLL `0.3316`) degraded `10%` (`+0.0010`), `25%` (`+0.0007`), overall RPS, Pairwise LL, and Points Brier, so `8a` was dropped under the no-bucket-regression rule.
- *Per-bucket and per-race highlights (`9d`, official `m7-train,m7-evaluate,verify-m7 --promote --log-mlflow`, `verify-m7: OK`, `test_sealed: true`):*
  - Beats both `persist` and `transition` in **all 5 distance buckets**, with **4/5 individual buckets strictly statistically significant (95% CI < 0) vs `transition`**: `10%: 0.0924` (`−0.0072 [−0.0137, −0.0003]`), `25%: 0.0867` (`−0.0188 [−0.0344, −0.0040]`), `50%: 0.0810` (`−0.0155 [−0.0295, −0.0031]`), `75%: 0.0580` (`−0.0069 [−0.0144, +0.0004]`), `90%: 0.0358` (`−0.0074 [−0.0114, −0.0044]`).
  - **Event calibration (Brier):** `win = 0.0212`, `podium = 0.0368`, `points = 0.1025`, `dnf = 0.0391` (`mean forecast 0.047 vs observed 0.041`).
  - **Beats `transition` in 11/12 validation races** (up from 5/12 in Exp 1b): `2025 Monaco` `0.0896 → 0.0701` (`−0.0089` vs trans), `2025 Montréal` `0.1075 → 0.0908` (`−0.0112`), `2025 Silverstone` `0.0904 → 0.0750` (`−0.0063`), `2025 Spa` `0.0733 → 0.0577` (`−0.0060`), `2025 Melbourne` `0.1015 → 0.0971` (`−0.0026`), `2025 Imola` `0.0739 → 0.0643` (`−0.0274`), `2025 Sakhir` `0.0633` (`−0.0233`), `2025 Jeddah` `0.0460` (`−0.0179`), `2025 Spielberg` `0.0451` (`−0.0161`), `2025 Suzuka` `0.0443` (`−0.0158`), `2025 Shanghai` `0.0845` (`−0.0055`), and `2025 Barcelona` `0.1212 → 0.1111` (`+0.0071`).
  - `generalization_status = balanced` (`val/train prior RMSE ratio 0.99×`, `val z² = 0.98`), `manifest_sha256 = 93b9edeb723cc94c62091cc194d8e95a74f88981f678c17d25dc34b415a11a43`.
