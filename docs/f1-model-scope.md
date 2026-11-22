# M7 Model Component — Scoping Draft

*2026-11-10. Sources: the "F1 prediction" chat (shared link, Sep 20) and the repo at `master` @ `21b50bb`. Status summary: `f1-race-intelligence-STATUS.md`.*

## 1. The idea, condensed

**Target output:** an N×N matrix (22×22 from 2026 on) where row *i* is the probability distribution of driver *i*'s finishing position.

The chat's main conclusions:

1. **Don't estimate the matrix cell by cell.** That is about 441 free parameters per race, and per-driver models break the column constraint.
2. **Simulate it instead.** Draw full finishing orders from a generative model. Each draw is a permutation matrix, and their average is **doubly stochastic by construction**, with no Sinkhorn repair needed.
3. **Generative model = linear pace plus discrete events.**
   - Lap-time state-space: driver/car pace + fuel (lap number) + tyre degradation (compound × stint age) + track evolution + AR(1) error.
   - Simulate forward, sort the cumulative times, and average the permutations.
   - The non-linear parts are modelled separately: pit stops (jumps of about 20–25 s), safety car/VSC (a regime switch that compresses the gaps), overtaking (a threshold plus a pass probability), and DNFs (their own hazard).
4. **Lap resolution, not 0.5 Hz.** High-frequency data is only worth using for the overtaking sub-model.
5. **DNFs make the matrix N×(N+1).** Rows still sum to 1. Column *j* sums to P(at least *j* finishers), and the DNF column sums to the expected number of retirements. Simulation handles this for free. *(Superseded: v1 uses convention (a). Retired cars take the last positions, so the matrix is N×N and doubly stochastic, and Pr(DNF) is kept as a side output.)*
6. **The Markov chain on positions isn't the main model.** Position alone isn't Markov. It is useful for **lap 1** (grid → end of lap 1) and as a **baseline**: pooled position-transition counts by circuit and race phase, normalized with Sinkhorn.
7. **Evaluation:**
   - Ranked probability score (RPS) per row, and the log-likelihood of the realized order.
   - Calibration of win, podium and points.
   - Report everything **by laps remaining**, never pooled.
   - Baselines: "the current order persists, with a pit correction" and grid order.

## Decision (2026-11-10): v1 forecasts in-race, from lap *k*

**Problem statement.** Given everything observed up to the end of lap *k* of a race, predict the N×N
matrix of final position for every driver. Under convention (a), retired cars fill the last positions, and Pr(DNF) per driver is a side output.

What this implies:

- **Conditioning state at lap *k*:** the order and cumulative times/gaps, each car's compound, tyre age and
  stint number, stops made so far, the current regime (green/SC/VSC), cars already retired, laps remaining,
  and the circuit.
- **Anti-leakage rule (extends the M4/M5 rule):** the forecast at lap *k* may use only data from laps ≤ *k*
  of that race, plus anything fitted on **training races**, plus the **results of races that finished before
  this one** (the form state of the pace prior, updated chronologically and including earlier validation/test
  races). Nothing from lap *k*+1 on, and nothing from this race's final result. Each forecast row needs a guard that recomputes the state from the raw laps truncated at *k*.
- **Evaluation unit:** (race, *k*). Use a fixed grid of *k* values such as 1, 5, 10, 20, … or fractions of the
  race (10%, 25%, 50%, 75%), so that races of different lengths stay comparable. Report scores per bucket.
- **Pre-race forecasting is out of scope for v1.** Lap 1 becomes just the *k* = 1 case, and a grid → lap-1
  transition model is optional. The grid and qualifying lap time are still used, but only as the **pace prior**
  (math note §3.3). Both are known before lap *k*, so they're admissible.
- **Retired cars:** a car that has retired by lap *k* has a deterministic row: it sits in the last positions,
  ordered by laps completed. Leave these rows out of the RPS, so they don't deflate it.
- **Disqualified cars** have a null `position` in `session_result`. A DSQ is a stewards' decision, so the model
  can't forecast it. Drop those rows from scoring.

## OpenF1 data audit (2026-11-10, live API)

Coverage was checked against every Grand Prix from 2023 to 2025: 70 races, after dropping the cancelled 2023 Imola race. The scripts and raw outputs are in the session scratchpad and are not part of the repo.

### Endpoints mapped to model components

| Model component | Endpoint → fields | In M2 today? | Coverage / caveats |
|---|---|---|---|
| **Labels (final order, DNF)** | `session_result` → `position`, `number_of_laps`, `duration`, `gap_to_leader`, `dnf`, `dns`, `dsq` | **No** | All 70 races. Usually 20 rows, and one race per season has 19. 159 DNFs, about 2.3 per race and about 11% of starts. **`position` is null for 154 rows** (the unclassified cars), so convention (a) ranks those by `number_of_laps`. `dnf=true` with a non-null position is a classified retirement (≥ 90% distance); this was checked on Melbourne 2023, where five cars have it. DSQ rows have a null position. |
| **Race state at lap *k*** | `laps` (lap_duration, is_pit_out_lap, date_start), `position`, `intervals` (gap_to_leader, interval, `"+1 LAP"` strings) | Yes | Already consolidated in M4. `intervals` updates about every 4 s, and lapped cars come as strings, not numbers. |
| **Pace prior (grid)** | `starting_grid` → `position`, `lap_duration` (qualifying lap) | **No** | Returns **404 for a race `session_key`**. It works with the **qualifying** `session_key` or with `meeting_key` (checked on 2023, 2024 and 2025). The qualifying lap time gives a *continuous* pace prior, not just a rank. On sprint weekends, check that `meeting_key` doesn't also return the sprint grid. |
| **Tyres / degradation** | `stints` → `compound`, `lap_start`, `lap_end`, `tyre_age_at_start` | Yes | Full coverage. `tyre_age_at_start` > 0 means used tyres, which matters for a_k in s\*. INTERMEDIATE/WET compounds mark wet races, where the two-compound rule doesn't apply. |
| **Owed stop, pit loss Λ** | `pit` → `lane_duration`, `lap_number`, `stop_duration` | Yes | **404 for 6 early 2023 races** (Sakhir, Jeddah, Melbourne, Baku, Miami, Monaco). Use `stints` for stop laps there, and fit Λ on the other races. `stop_duration` (stationary time) exists only from the 2024 US GP on. **`pit_duration` is deprecated** (removed after 2026) and is the same as `lane_duration`. M4 reads both, so switch to `lane_duration`. |
| **Safety car / VSC / red flag** | `race_control` → `category`, `message`, `lap_number`, `flag` | Yes (only as a count) | **Encoding confirmed:** `category="SafetyCar"` with messages `SAFETY CAR DEPLOYED`, `SAFETY CAR IN THIS LAP`, `VIRTUAL SAFETY CAR DEPLOYED`, `VIRTUAL SAFETY CAR ENDING`, `SAFETY CAR THROUGH THE PIT LANE`, and `flag="RED"` / `RED FLAG`. `lap_number` was never null on these events. Frequencies over the 70 races: **SC in 30 (43%)**, of which 8 had 2–3 SCs; VSC in 25; red flag in 8. Deployments outnumber "in this lap" endings because SCs can end in a red flag or at the flag. |
| **DNF hazard covariates** | `race_control` (CarEvent, incidents), `weather.rainfall`, `meetings.circuit_type` | Partly | There's no DNF *reason* field. Crash vs mechanical has to be inferred from race-control messages. `circuit_type` (Permanent / Street / Road) is a natural pooling level for h_c. |
| **Overtaking (option B)** | `overtakes` → overtaking/overtaken driver, `position`, `date` | **No** | All 70 races, median about 240 rows per race. It **includes position swaps from pit cycles and penalties** and is flagged "may be incomplete", so it is not a clean on-track pass count. Not needed for v1 (option A). |
| **Weather** | `weather` → `rainfall`, track/air temp, wind | Yes | About 1 sample per minute. |
| **Circuit** | `meetings` → `circuit_key`, `circuit_type`, `circuit_info_url` | Partial (via sessions) | — |
| Not useful for v1 | `car_data`, `location` (≈ 3.7 Hz), `team_radio`, `championship_*` (beta) | — | The high-frequency data is only for a future pass model. |

### Findings that change the plan

1. **The repo's `session_types: ["Race"]` also pulls in sprints.** M2 matches `session_name` *or* `session_type`, and sprints have `session_type="Race"`. That's 18 sprints over 2023–25 (six per season), which are currently mixed into M4–M6. M7 must filter on `session_name == "Race"`, or model sprints separately: they're shorter, have no mandatory stop, and run on a different strategy.
2. **Red flags break the owed-stop logic.** Teams can change tyres under a red flag, which satisfies the stop for free, and standing restarts reset the gaps (Monaco 2024: red flag on lap 1). Treat the red flag as a regime that clears owed stops and compresses gaps fully.
3. **Early red flags:** three races had one on lap 1 (Suzuka 2024, Monaco 2024, Spa 2025), and São Paulo 2023 had one on lap 2. The k = 1 forecasts in those races need the regime flag, or they should be excluded.
4. **Wet races** are identifiable from `stints.compound` and `weather.rainfall`. The owed-stop rule has to be switched off there.
5. **The DNF count is higher than my earlier guess:** about 160 over 3 seasons, not 110–130. That's roughly 7 per circuit. Partial pooling is still needed.

### Phase 0 additions (revised)

- Add **`session_result`** and **`starting_grid`** (keyed on `meeting_key`) to the M2 endpoints, with M3 specs for both.
- Add a **session filter on `session_name`** for M7 so sprints are excluded.
- Build a **typed per-lap regime** (GREEN / SC / VSC / RED) from the `race_control` messages above.
- Add a **stop-lap fallback from `stints`** for the 6 races without `pit` data.
- Switch M4 from `pit_duration` to `lane_duration`.
- Add the **race-state-at-lap-*k*** table, unchanged.

## 2. How it maps onto the repo

| Idea component | What exists now | Gap |
|---|---|---|
| Pace model (lap-time AR) | M5/M6 target `next_lap_time`, with features `lap_number`, `compound`, `tyre_age`, `stint_lap`, `lap_time_prev_1`, rolling stats, weather, `team_name`, `driver_acronym` | No model yet. **See the simulability problem below.** |
| Cumulative time / gaps | `gap_to_leader_s`, `interval_s`, `position_at_lap_start/end` | Fine for setting the simulator's starting state |
| Pit stops | `is_pit_out_lap`, `stint_number`, `pit` endpoint | Need per-circuit pit-loss and stint-length distributions |
| Safety car / VSC | Only `rc_events_in_lap` (a count) | **Need a typed per-lap SC/VSC/red-flag flag** built from `race_control` category/message |
| DNF hazard | `is_last_lap_for_driver` (leaky, M4 metadata) | **No official classification or DNF reason** |
| Final order (labels) | Inferred from `position` | **Not extracted.** `session_result` covers all 70 races (see the audit) |
| Grid / lap 1 | `position_at_lap_start` on lap 1 approximates it | `starting_grid` via `meeting_key` works (see the audit) |
| Qualifying pace | — | `session_types: ["Race"]` only, so needed only for a pre-race forecast |
| Train/val/test | M6: train 2023+2024, val = first half of 2025, test = second half of 2025, split by whole race | Test is about 12 races, so expect wide confidence intervals |

### The simulability problem (important)

The M6 feature matrix is built for a **one-step-ahead** prediction. It uses lap *t*'s telemetry aggregates (speed, rpm, throttle, DRS share) and its gaps. Simulating 30 laps forward means generating those features for laps that haven't happened, and they can't be generated.

So M7 needs **two feature sets**:

- **One-step model (M7a-1):** the full M6 features. This is a valid standalone deliverable and directly answers the current README goal.
- **Simulation pace model (M7a-2):** only covariates that can be **rolled forward**: driver/team, compound, tyre age/stint lap, lap number (fuel), the previous simulated lap time, the regime (green/SC/VSC) and a circuit effect.

### Grid size

2023–2025 races have **20** cars; 22 is only from 2026 on. The code should build an N×(N+1) matrix with N taken per race, never a hard-coded 22.

## 3. Proposed phases

### Rescope (2026-11-10): analytic core, simulated events

The math walkthrough showed that **between events, each car's finishing time is a Gaussian random walk**. The
mean is the banked time, plus pace × laps left, plus degradation, plus Λ for an owed stop. The variance is
n²·v (pace uncertainty) plus n·σ²·AR factor (noise). So v1 needs no lap-by-lap simulator. Each scenario
draws its discrete events (safety car, red flag, DNFs), then **one Gaussian Z_i per car**. Sorting the Z's gives
a permutation, and averaging over S scenarios gives P. Draw total times per car, never gaps to the leader:
gaps to the leader share the leader's noise, and drawing them independently doubles every non-leader
head-to-head variance. Details are in math note §4.3.

**Component choices for v1:**

| # | Component | v1 | Deferred |
|---|---|---|---|
| 1 | Baselines | "Order persists + owed-stop correction"; transition matrix Q per *k* bucket | — |
| 2 | Pace | θ_i with a **qualifying deficit + recent-history form** prior (candidate d: car and driver random walks across races; fallbacks for no qualifying time or no history). Kalman update on clean laps; linear degradation per compound. Grid-only (c) is the ablation. | Teammate covariance within the race, Student-t |
| 3 | Remaining time | Analytic Gaussian Z_i, sampled and sorted; pairwise Φ used as a unit test | Lap-by-lap simulation |
| 4 | Pit stops | Owed stop at s\* + empirical noise; Λ per circuit; owed stop switched off in wet races | Undercut/overcut reactions |
| 5 | SC / VSC / red | Mixture over timing: SC compresses gaps and makes stops cheap; VSC makes stops cheap without compression; red flag compresses fully and clears owed stops | SC–DNF coupling |
| 6 | DNF | Constant per-lap hazard, partially pooled by circuit (or `circuit_type`) toward the global rate; one wet multiplier | Crash vs. mechanical split |
| 7 | Overtaking | Frictionless (A) | Pass model (B) |
| 8 | Scoring | RPS (J = N), pairwise log loss, calibration per *k* bucket, race-paired bootstrap | Exact-order likelihood |

**Rule for keeping a component:** components 2–4 together must beat baseline 1. Every later component stays
only if it improves the validation RPS in at least one *k* bucket.

### Phases

| Phase | Deliverable | Success criterion |
|---|---|---|
| **0. Data** | See "Phase 0 additions (revised)" in the audit: `session_result` + `starting_grid` in M2/M3; a `session_name == "Race"` filter (no sprints); a typed per-lap regime (GREEN/SC/VSC/RED); a stop-lap fallback from `stints`; `lane_duration` instead of `pit_duration`; the **race-state-at-lap-*k*** table with its truncation guard. | Every 2023–25 GP has a final order with DNF/DSQ marked; the state table passes its leakage guard; no sprint rows reach M7 |
| **1. Eval harness + baselines** | RPS / pairwise log loss / calibration per (race, *k*) bucket. Baselines 1 (order persists + owed stop) and Q (transition matrix; Sinkhorn only for the unequal-row edge). | Baselines scored on validation; the harness accepts any `P` |
| **2. Descriptive pass** | Math note §12: φ, σ_η, d_c / o_c, λ_circ, SC/VSC hazards and compression level, DNF rates. Full-race θ̂ per driver-race, then fit the two-level pace filter (math §3.3 x: ρ_in, ρ_season, q_γ, q_δ, s, λ, σ_q) by Kalman likelihood, and compare it with grid-only τ̂_c and with ρ≡1. | Each parameter estimated with an uncertainty; form beats grid-only on τ̂², or the fallback to (c) is recorded |
| **3. Core model** | Components 2–4: the pace posterior, the Gaussian Z_i sampler, owed stops at s\*. Green flag only. | Simulated head-to-heads match the closed-form Φ (unit test); beats baseline 1 on RPS in some *k* buckets |
| **4. Events** | Components 5–6: the SC/VSC/red mixture and the pooled DNF hazard | Improves validation RPS over phase 3 in at least one bucket |
| **Optional** | M7a-1 one-step `next_lap_time` on the M6 features (if the course needs it); overtaking B, if simulated position changes are far off reality | — |
| **Later** | Form random walk, teammate covariance, SC–DNF coupling, Bayesian fit, pre-race forecasting, MLflow, SageMaker serving | — |

**Code placement (proposal):**
- `src/f1_race_intelligence/simulation/` for pace, events, simulator and matrix.
- `src/f1_race_intelligence/evaluation/` for scores and baselines.
- `--stage m7` in the CLI.
- Artifacts under the lake's reserved `models/` prefix.
- The same patterns as the rest of the pipeline: guards, a report per run, idempotent writes.

## 4. Open decisions

1. ~~Forecast moment~~ **Decided: in-race, from lap *k*.** See the decision section above.
2. **What the matrix is for:** pre-race forecast, comparison with betting markets, or season simulation. This decides how much joint structure (head-to-heads, teammates) has to be kept and reported.
3. **The academic deliverable's scope:** the one-step `next_lap_time` model alone may already satisfy the course. The simulator could be the ambitious extension.
4. **2026 data:** the config currently excludes 2026 (the season is ongoing and the regulations reset). Should it serve as an out-of-time check?
5. **Frequentist vs. Bayesian** for the pace model. Starting with linear plus Kalman is simplest; dynamic skills point toward Bayesian.
6. ~~Model structure~~ **Decided (2026-11-10):** analytic Gaussian core with simulated events. See the rescope
   above. This settles overtaking (A), pit timing (s\*), SC–DNF coupling (independent in v1) and Gaussian noise.
   **Pace prior decided:** qualifying + recent-history form (d), with grid-only (c) as the ablation it must beat.
7. **Sprints:** exclude them (v1 default), or model them later as a separate race type? Either way, M4–M6 currently include them.

## 5. References from the chat (unverified)

Bell et al. 2016 (JQAS); van Kesteren & Bergkamp 2023 (JQAS); Henderson & Kirrane 2018 (Bayesian Analysis); Bekker & Lotz 2009 (JORS); Heilmeier et al. 2018, 2020; Sinkhorn & Knopp 1967; Mena et al. 2018; Linderman et al. 2018.
