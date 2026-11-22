# M7 race model: specification as implemented

What `src/f1_race_intelligence/race_model/` does: its formulation, assumptions, censoring and DNF
conventions, priors, scope and anti-leakage protocol. Derivations and the reasoning behind each decision are
in `f1-model-math.md` (sections cited as §). Scope history is in `f1-model-scope.md`. Validation results are in
`m7-experiments.md`.

## 1. Target

Given everything observed until the leader completes lap *k*, forecast for race *r*:

$$P_{ij} = \Pr(\text{driver } i \text{ finishes in place } j \mid \mathcal F_k),\qquad i, j = 1..N_r,$$

plus $\Pr(\text{retirement}_i)$ as a side output. $N_r$ is read from the race, never hard-coded: 20 in
2023–25, 22 from 2026.

**Convention (a), §1, §7.2.**

- **Retirement.** A car *retires* if it is **not classified**: a null `position` in `session_result`, and
  neither DSQ nor DNS. This is `racedata.is_retirement`, the **single definition** used by the hazard fit
  (`events.fit_events`), the simulator and the evaluation (`FinalOrder.retired`).
- **Classified DNF.** A car with `dnf=True` *and* a position covered ≥ 90% of the distance and was classified.
  For the matrix it finished in that place, so it is not a retirement. It is kept in
  `FinalOrder.classified_dnf` for information.
- **Order.** Finishers are ordered by total race time. Retired cars come after every finisher, ordered by laps
  completed (most first). So $P$ stays $N \times N$ and doubly stochastic.
- **DSQ and DNS** are stewards' or pre-race outcomes the model cannot forecast. They are not scored.
- **Already retired.** Cars retired by lap *k* have deterministic rows and are left out of every score.

## 2. Information set and anti-leakage protocol

- **What the forecaster sees.** It only accepts a race cut by `racedata.truncate_at_lap`:
  - laps, pit stops, race-control messages and intervals after the leader's lap-*k* line crossing are removed;
  - stints are cut at each car's last completed lap;
  - **the official result is dropped**.

  `forecast.car_states` refuses an uncut race, or one still carrying a result.
- **Splits are fixed dates** (`race_model.validation_until`, `test_until`), so adding seasons never moves a
  race between splits:
  - train: 2023–24 (46 races);
  - validation: 2025 up to Spa (12);
  - test: Budapest to Abu Dhabi (11);
  - races after that are `future`.
- **Fitted on training races only:**
  - AR(1) coefficient φ and lap noise σ_ε;
  - tyre model;
  - season-filter hyperparameters (after a 6-race burn-in that adds no likelihood);
  - SC/VSC/red and retirement hazards;
  - pit loss and stop timing;
  - race distances;
  - transition baseline;
  - lap-noise multiplier.

  `tests/unit/test_race_model_isolation.py` checks that replacing the validation races changes **none** of
  these.
- **Allowed online information.** The season filter runs forward through validation races, so each race's
  prior uses the results of races that finished before it, as a live forecaster's would. Hyperparameters stay
  frozen.
- **The test set is sealed** (`race_model.test_sealed`):
  - no stage reads a test race's raw files;
  - the only entry is `m7-evaluate --split test --unseal-test`, recorded in `test_ledger.json` before anything
    is read, and refused a second time;
  - it never promotes a model;
  - test priors are rebuilt online with every hyperparameter frozen.

## 3. Pace

Lap times are expressed in **% of the race's median clean lap**, so pace transfers across circuits.

**Level 1: one race (§2–3).**

$$y_{il} = \alpha_l + \theta_i + o_{c} + d_{c}\,a_{il} + \varepsilon_{il},\qquad \varepsilon_{il} = \phi\,\varepsilon_{i,l-1} + \eta_{il}.$$

- $\alpha_l$ is a lap fixed effect (fuel and track evolution, common to all cars, so it cancels from the
  order). $\theta_i$ is driver-race pace. $c$ is the compound and $a$ the tyre age.
- **Clean laps:**
  - lap > 1; not an in- or out-lap; dry compound;
  - not under SC/VSC/red, nor the lap after (a red flag closes an open SC);
  - the field median for that lap ≤ 1.04 × the race median;
  - free air (≥ 1.5 s to the car ahead, or leading);
  - then a 3-MAD residual trim.
- **Ridge least squares**, with weak Gaussian priors on the compound terms (intercept SD 1%, slope SD 0.1% per
  lap). A compound run by one or two cars then cannot trade off against those cars' pace.
- **SE** comes from the coefficient covariance of the field-mean contrasts, inflated by $\sqrt{(1+\phi)/(1-\phi)}$.
- **Wet races** give no pace observation.

**Level 2: across races (§3.3 x–xi).** For driver $i$ in team $t$:

$$\theta_{ir} = \gamma_{tr} + \delta_{ir} + c_{tr} + w_{ir}$$

$$\gamma_{tr} = \gamma_{t,r-1} + \xi,\ \xi\sim\mathcal N(0, q_\gamma^2)\ (q_{\gamma,\text{season}} \text{ across a winter}),\qquad \delta_{ir} = \delta_{i,r-1} + \xi'.$$

- $c_{tr} \sim \mathcal N(0, s_c^2)$ is team × circuit fit, redrawn every race. $w_{ir} \sim \mathcal N(0, s_w^2)$
  is the race-day transient.
- ρ ≡ 1 (a random walk): fitting ρ gave 0.99 and no validation gain.
- **Qualifying** is a measurement, $q_{ir} = \lambda(\gamma+\delta+c) + u$ with $u\sim\mathcal N(0,\sigma_q^2)$.
  It is skipped when there is no usable time.
- **After the race**, level 2 reads the full-race $\hat\theta$ with noise $s_w^2 + SE^2$.
- **Hyperparameters** are fitted by maximising the Kalman prediction-error likelihood on training races.

**Priors.**

| Quantity | Prior | Why |
|---|---|---|
| New team's γ | $\mathcal N(0, 0.25)$ | ≈ field spread² (% of a lap); not identified from data (§3.3 xi) |
| Rookie's δ | $\mathcal N(0, 0.05)$ | Drivers differ less than cars |
| Compound offset / slope (level 1) | $\mathcal N(\text{0 or pooled}, 1^2)$ / $\mathcal N(\cdot, 0.1^2)$ | Identifies pace when few cars run a compound |
| Pre-race pace for level 1 | $\mathcal N(H\hat x, HPH^\top + s_w^2 I)$ from the filter | Joint across drivers, so teammates co-vary |
| Hazards per circuit | Partial pooling $(e_c + \kappa\bar h)/(n_c + \kappa)$ | About 2 training races per circuit |

**In race (§3.2).** Laps $1..k$ are fitted with the same ridge model. The tyre priors are centred on the
training pooled and circuit values. The field-mean contrasts update the prior by a Kalman step, with σ blended
between the in-race estimate and the pooled one (40-lap weight).

## 4. From pace to positions (§4–7)

Monte Carlo over $S = 4{,}000$ scenarios, vectorised. Per scenario:

1. **Pace.** Draw all cars' pace jointly from the posterior. This carries the $n^2 v$ pace-uncertainty term and
   the teammate covariance.
2. **Stops.**
   - A car that has not used two dry compounds must stop.
   - Any car stops once more if the degradation it saves at $s^*$ exceeds the pit loss.
   - $s^* = (d_2 L - d_1(a_k - k))/(d_1+d_2)$ plus the training bias, and the cheapest legal compound is chosen.
   - Tyre cost uses the planned lap. The empirical spread of stop laps only times stops against
     neutralisations.
3. **Events.**
   - The first SC or red flag after *k* is drawn from per-lap hazards, with empirical SC lengths.
   - **SC/red:** the queue is compressed to 0.8 s per place at the restart. Under SC, owed stops are taken with
     probability 0.7 at the reduced SC pit loss. Under red, owed stops are free.
   - **VSC:** owed stops are taken with probability 0.4 at the VSC pit loss.
   - **Retirement:** each running car retires at a per-lap hazard (circuit-pooled, with a wet multiplier).
4. **Time to the flag.** Banked time, plus remaining laps × pace, plus tyre cost, plus stop losses, plus AR(1)
   lap noise with long-run variance $\sigma_\varepsilon^2(1+\phi)/(1-\phi)$ per lap. The lap noise is scaled
   by a multiplier chosen on training races per race phase.
5. **Order.** Finishers by total time, then retired cars (§1).

Averaging the permutation matrices makes $P$ doubly stochastic for any $S$. With events switched off the totals
are jointly Gaussian, and the simulated head-to-heads match the closed form $\Phi(-E[G]/\sqrt{\mathrm{Var}\,G})$
(unit test).

## 5. Censoring conventions

- **Retired cars in level 1.** They contribute only the clean laps they ran. Their $\hat\theta$ therefore has a
  larger SE, and a car with no clean laps gives no observation. A retirement is never treated as slow pace.
- **Hazard exposure.**
  - Each car is at risk for the laps it actually ran after lap 1.
  - Forecasts start at $k \ge 1$, so lap-1 retirements are excluded from both events and exposure.
  - A classified DNF contributes exposure, but no event (§1).
- **Classification.** Retired cars are ranked by laps completed. The time a retired car would have taken is
  never imputed.
- **Missing inputs.** A wet race gives no pace observation (missing, not zero). With no qualifying time, the
  prior is form only. Cars without an interval reading at a lap are not counted as being in free air.
- **Race distance** is taken from training races at the same circuit. For a circuit with no training history,
  the winner's laps in the result are used, which is a documented exception.

## 6. Assumptions and known limitations

- **Pace is constant within a race** (no drift), and degradation is **linear in tyre age** (no cliff) (§2.2,
  §3.3 vii).
- **Passing is frictionless** (§8, option A). Rows are somewhat over-dispersed late in the race. An overtaking
  cost was tried and reverted (`m7-experiments.md`, experiments 3–5).
- **Gaussian lap noise** after clean-lap filtering. The tails are mildly heavier (§3.3 xi).
- **Hazards are independent.** At most one SC/red and one VSC are modelled after *k*. Retirements do not
  trigger safety cars.
- **Strategy is not reactive.** There are no undercut or overcut responses, and at most one further stop per
  car.
- **Not forecast:** DSQ, DNS, penalties applied after the race, and sprints.

## 7. Scope

- **In:** Grands Prix (`session_name == "Race"`) from 2023 on, in-race from lap *k* ≥ 1, dry and wet, raw
  data or live OpenF1.
- **Out (v1):** pre-race forecasting, sprints, a lap-by-lap overtaking model, telemetry, Bayesian
  hyperparameter uncertainty, and SC–retirement coupling.

## 8. Evaluation and promotion (§9–10)

- **Unit:** (race, *k*), with *k* at 10/25/50/75/90% of distance. Cars running at *k* are scored.
- **Scores:**
  - ranked probability score per row (J = N);
  - pairwise log loss from simulated head-to-heads;
  - Brier scores and reliability tables for win, podium, points and retirement.
- **Baselines:**
  - the order at *k* persists, with an owed-stop correction;
  - a transition matrix counted on training races per *k* bucket.
- **Uncertainty:** a race-paired bootstrap of the RPS difference. The race is the unit.
- **Promotion:** a new version becomes `latest` only if it beats the configured baseline on mean validation
  RPS. Test is never used for that.
