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
- **Still open:**
  - 75% and 90% remain level with, or behind, the transition baseline.
  - The friction cost is one number; a version that grows with the gap and pace advantage seen in the study
    is the next step towards option B.
  - The noise grid still tops out at ×3.
