# M7: Working Through the Math (in-race forecast from lap *k*)

*Working note, kept in the order decisions were made (from 2026-11-10). It companions `f1-model-scope.md`.*

- **Early sections** were written before anything was fitted, so their numbers are illustrative unless a section
  says otherwise.
- **Fitted values** are in §3.3 (xi) and §13. Later experiments are in `m7-experiments.md`.
- **The model as implemented** is summarised in `m7-model-spec.md`, which points back to the sections here: its
  formulation, assumptions, censoring and DNF conventions, priors, scope and anti-leakage protocol.

---

## 0. Notation

- Race $r$ with $L$ laps. Drivers $i = 1,\dots,N_r$; $N_r = 20$ for 2023–25 and $22$ from 2026.
- $y_{i,l}$ is driver $i$'s time on lap $l$. $T_{i,l} = \sum_{m \le l} y_{i,m}$ is the cumulative time.
- $\mathcal F_k$ is everything observed up to the end of lap $k$. The laps still to run are $n = L - k$.
- $R_i = \sum_{l=k+1}^{L} y_{i,l}$ is the remaining time. $Z_i = T_{i,k} + R_i$ is the total race time.
- $a_{i,l}$ is tyre age, $c_{i,l}$ is compound, and $m_i$ is the number of stops still owed.

---

## 1. The target object and why simulation preserves its structure

**Decision (2026-11-10): DNF convention (a).** Retired cars fill the last positions, so the matrix stays
**N×N and doubly stochastic**.

- Finishers are ranked by (laps completed ↓, time ↑). Retired cars come after all finishers, ordered by laps
  completed ↓, which puts the earliest retirement last.
- Ties (same retirement lap) break on the time at which the car last crossed the line.
- Cost: the bottom positions mix "slow finisher" with "retired". Keep $\Pr(\text{DNF}_i)$ as a **side output**
  of the simulation, outside the matrix, so it can still be calibrated.

Let $\sigma$ be the random final order under this convention. Define

$$P_{ij} = \Pr(\sigma(i) = j \mid \mathcal F_k),\qquad \textstyle\sum_j P_{ij} = 1,\quad \sum_i P_{ij} = 1.$$

**Monte Carlo estimator.** Draw $S$ races. Each draw $s$ gives a permutation matrix $M^{(s)}$, and the
estimate is $\hat P = \frac1S\sum_s M^{(s)}$.

- Row sums: $\sum_j \hat P_{ij} = \frac1S\sum_s\sum_j M^{(s)}_{ij} = \frac1S\sum_s 1 = 1$, and the same for
  columns. The constraints hold **exactly, for any $S$**, because every draw satisfies them and averaging is
  linear.
- Each cell is unbiased, with $\mathrm{SE}(\hat P_{ij}) = \sqrt{P_{ij}(1-P_{ij})/S} \le 0.5/\sqrt S$.
  $S = 10{,}000$ gives an SE of at most 0.005, and drawing is cheap (§8).

**Consequence:** the modelling effort goes entirely into the **joint distribution of $(Z_1,\dots,Z_N, D)$**.
The matrix is just a readout of it.

---

## 2. Pace: the lap-time equation

$$y_{i,l} = \underbrace{\mu_r + b_r\, l + w_{r,l}}_{\text{common to all cars}} + \theta_{i,r} + \underbrace{o_{c} + d_{c}\, a_{i,l}}_{\text{tyre}} + \underbrace{\Lambda\,\mathbb 1[\text{pit}]}_{\text{stop}} + \varepsilon_{i,l}$$

### 2.1 Everything shared cancels out of the ranking

The order depends only on the **differences** $Z_i - Z_j$. The common block (base lap time $\mu_r$,
fuel burn + track evolution $b_r l$, and the weather shock $w_{r,l}$) is identical for every car and drops out.

- **Model relative pace:** $\tilde y_{i,l} = y_{i,l} - \mathrm{median}_i(y_{i,l})$.
- There is no need to separate fuel from track evolution. Both are linear in $l$, so they aren't separately
  identifiable within a race anyway.
- **Fuel is only approximately common.** Start loads, burn rates, engine modes and lift-and-coast differ
  between cars. Write $b_i = b + (b_i - b)$: the shared part cancels, the deviation does not.
  - v1 folds the deviation into $\theta_{i}$.
  - Check the residuals for driver-specific trends. A trend is indistinguishable from the within-race drift of
    §3.3(vii).
- The caveat is that this holds only on green-flag laps without traffic. The safety car and lapping depend on
  absolute times (§5, §7).

### 2.2 What does *not* cancel

- **Driver-race pace $\theta_{i,r}$.** This is the main object. Driver and car can't be separated within a race
  and don't need to be: forecasting in-race only needs their sum.
- **Tyres, $o_c + d_c a$.** Cars are on different compounds and ages.
  - Degradation $d_c$ is identified from the **reset of $a$ at each stop**. Without stops, $d_c a$ and $b_r l$
    would be collinear.
  - Each circuit appears in only about 2 training races, so pool across circuits:
    $d_{c,\text{circ}} \sim \mathcal N(d_c, \omega^2)$.
  - The linear form ignores the tyre "cliff". This is a known limitation; check the residuals at high age.
- **Stops**, i.e. the pit loss $\Lambda$ (§4).

### 2.3 The error term: AR(1)

$$\varepsilon_{i,l} = \phi\,\varepsilon_{i,l-1} + \eta_{i,l},\quad \eta \sim \mathcal N(0,\sigma_\eta^2),\quad \sigma_\varepsilon^2 = \frac{\sigma_\eta^2}{1-\phi^2}.$$

- Real laps have heavy tails (traffic, mistakes, lock-ups). Either use Student-*t* $\eta$ or exclude laps by a
  rule: in-laps and out-laps, lap 1, SC/VSC laps, and $|\text{resid}| > 3$ robust SD.
- Excluding laps is simpler; the cost is a slightly optimistic noise estimate.

---

## 3. Learning $\theta_{i,r}$ from laps $1..k$ (the core of the in-race forecast)

### 3.1 The prior

$\theta_{i,r} \sim \mathcal N(\bar\theta_{i,r}, \tau^2)$, where $\bar\theta$ is a pre-race expectation.
The candidates (flat, form, grid/qualifying, combined) and the v1 choice, qualifying + form, are in §3.3.

$\tau^2$ is estimated from training: it is the between-race spread of $\theta$ around $\bar\theta$.

### 3.2 The update

Quasi-differencing removes the AR(1) term:

$$\tilde y_{i,l} - \phi\,\tilde y_{i,l-1} - (\text{known tyre terms}) = (1-\phi)\,\theta_{i,r} + \eta_{i,l}.$$

This is a conjugate normal problem (equivalently, a Kalman filter with a static state). With $k'$ clean laps,

$$v_{i,k} \equiv \mathrm{Var}(\theta_{i,r}\mid\mathcal F_k) = \left(\frac{1}{\tau^2} + \frac{k'(1-\phi)^2}{\sigma_\eta^2}\right)^{-1}.$$

**Useful reading of this:** AR(1) errors shrink the information in $k'$ laps to an **effective sample size**
of $k'_{\text{eff}} \approx k'\,\frac{1-\phi}{1+\phi}$. With $\phi = 0.5$, 20 clean laps are worth about 6.7
independent laps. If laps are strongly autocorrelated, pace stays uncertain well into the race, and the prior
$\bar\theta$ matters much longer than intuition suggests.

### 3.3 The pace prior, worked through

**(i) The prior's weight, measured in laps.** From the update in §3.2, the posterior mean is a weighted
average:

$$\hat\theta^{\text{post}}_{i} = (1-B_k)\,\bar\theta_{i} + B_k\,\hat\theta^{\text{data}}_{i},\qquad B_k = \frac{k'}{k' + \kappa},\qquad \kappa = \frac{\sigma_\eta^2}{(1-\phi)^2\,\tau^2}.$$

- $\kappa$ is the **prior's worth in clean laps**: the data overtakes the prior at $k' = \kappa$.
- A better prior has a smaller $\tau$, so a larger $\kappa$. It keeps mattering for longer *and* lowers the
  pace term of the forecast variance:
  $$n^2 v_{i,k} = \frac{n^2}{\tau^{-2} + k'(1-\phi)^2/\sigma_\eta^2}.$$
- So **"which prior?" reduces to "which $\bar\theta$ gives the smallest $\tau^2$ out of sample?"** That can be
  measured before building anything else.

**(ii) Illustrative magnitudes.** *These numbers are made up for intuition, not estimated.* Take
$\sigma_\varepsilon = 0.4$ s and $\phi = 0.5$. Then $\sigma_\eta^2 = 0.12$, and each clean lap carries
$(1-\phi)^2/\sigma_\eta^2 \approx 2.1\ \text{s}^{-2}$ of information.

| Prior | $\tau$ | $\tau^{-2}$ | $\kappa$ (laps) | SD of the pace term at $k=5$ ($k'\approx3$), $n=52$ |
|---|---|---|---|---|
| Flat (field spread) | 0.6 s | 2.8 | 1.3 | ≈ 17 s |
| Grid-based | 0.25 s | 16 | 7.7 | ≈ 11 s |

For comparison, the lap-noise term contributes about 5 s SD over the same 52 laps. If the real numbers look
anything like these, **the prior is the single biggest lever on early-race forecasts**.

**(iii) Candidate priors.** All are expressed in **percent of the field-median lap time**, so that they
transfer across circuits (a 0.3 s gap means different things at Monaco and Spa). Convert back per race.

| | $\bar\theta_{i,r}$ | Comment |
|---|---|---|
| (a) Flat | 0 | $\tau^2$ = the field's spread. The reference point. |
| (b) Form | Kalman/EWMA over previous races (below) | Captures team development and driver form. Weak at season start, after team changes, and after the 2026 reset. |
| (c) Grid / qualifying | $\beta_0 + \beta_1 q_{i,r}$, with $q$ = the qualifying deficit to pole in % (`starting_grid.lap_duration`). Fallback: $h(\text{grid rank})$, a normal score | **Known at lap $k$.** It's in $\mathcal F_k$, so the scope decision to drop the grid is reversed for the prior. Using the qualifying *time* rather than the rank removes grid-penalty distortion. Still distorted by pit-lane starts without a time and by wet qualifying before a dry race. |
| (d) Combined | Form forecast, then a qualifying measurement update, in one Kalman chain (x). The regression $\beta_1 q_{i,r} + \beta_2 m_{i,r\mid r-1}$ is the fallback | Qualifying already contains form and circuit fit. Form corrects cars that run differently in the race than in qualifying, and it covers cars without a usable qualifying time. **Chosen for v1**, as the two-level filter in (x). |

**(iv) Form as a random walk across races** (the principled version of exponential weighting):

$$\theta_{i,r} = \theta_{i,r-1} + \xi_r,\quad \xi_r\sim\mathcal N(0,q^2);\qquad \hat\theta^{\text{full}}_{i,r} = \theta_{i,r} + e_{i,r},\quad e\sim\mathcal N(0,s^2).$$

- Kalman: $m_{r\mid r-1} = m_{r-1}$ and $P_{r\mid r-1} = P_{r-1}+q^2$, with gain
  $K = P_{r\mid r-1}/(P_{r\mid r-1}+s^2)$. In steady state this is an EWMA with weight $K$.
- (x) generalises this to $\gamma_r = \rho\,\gamma_{r-1} + \xi_r$, with $\rho$ and $q$ fitted separately in-season
  and across season boundaries. The driver component travels with the driver when they change teams.
- $s^2$ is not just measurement noise. It also absorbs circuit-specific car fit.

**(v) Teammate structure.** Write $\theta_i = \gamma_{\text{team}(i)} + \delta_i$. The prior covariance is then
block-structured: $\Sigma = \sigma_\gamma^2\,(\text{same-team indicator}) + \sigma_\delta^2 I$.

- The joint MVN update lets each car's laps inform its teammate's pace.
- It produces the positive posterior $\mathrm{Cov}$ that shrinks $\mathrm{Var}\,G_{ij}$ for teammates (§4.2).
- The payoff is mostly at small $k$.

**(vi) Estimating $\tau^2$ correctly.** The obvious estimate,
$\widehat{\mathrm{Var}}(\hat\theta^{\text{full}} - \bar\theta)$, is **inflated by measurement error** in
$\hat\theta^{\text{full}}$. Correct it:

$$\hat\tau^2 = \widehat{\mathrm{Var}}(\hat\theta^{\text{full}}_{i,r} - \bar\theta_{i,r}) - \overline{\mathrm{SE}^2(\hat\theta^{\text{full}}_{i,r})},\qquad \mathrm{SE}^2 \approx \frac{\sigma_\eta^2}{L'(1-\phi)^2}.$$

Without the correction, $\kappa$ is underestimated and the prior gets too little weight.

**(vii) Two threats to the constant-$\theta$ assumption.**

1. **Dirty air biases $\hat\theta^{\text{data}}$.** A car stuck behind a slower one posts laps closer to *that*
   car's pace.
   - Fix: count only **free-air laps** (interval to the car ahead > about 1.5 s; the threshold is to be tuned)
     as clean laps.
   - A car stuck in a train then has small $k'$ and falls back on the prior, which is the right behaviour.
     It also connects to the overtaking decision (§8).
2. **Pace drifts within a race** (damage, engine or fuel modes, managing a gap).
   - Model it as $\theta_{i,l} = \theta_{i,l-1} + \zeta$, $\zeta \sim \mathcal N(0, q_w^2)$. Old laps are then
     discounted, and the forecast variance gains a term $\approx n^3 q_w^2/3$.
   - Test on data: compare first-half vs. second-half $\hat\theta$ within each race, after the tyre terms.
     If the difference is no larger than measurement noise, keep $\theta$ constant.

**(viii) Leakage.**

- The hyperparameters ($\rho$, $q$, $s$, $\lambda$, $\sigma_q$; $\beta$, $\tau$ for the regression fallback) are fitted on the **training
  races only**.
- At forecast time, the form state $m_{r\mid r-1}$ may use the *results* of any earlier race, including earlier
  validation/test races, the way an online forecaster would. It never uses race $r$ beyond lap $k$.
- Grid is pre-race, so it is allowed.

**(ix) Procedure to pick the prior.** This is the first analysis to run once data is available.

1. Fit $\phi$ and $\sigma_\eta$ on free-air, green-flag clean laps of the 2023–24 races.
2. Compute $\hat\theta^{\text{full}}_{i,r}$ and its SE for every driver-race.
3. For candidates (a)–(d), compute the measurement-corrected $\hat\tau^2$ via leave-one-race-out on training,
   and report $\kappa$.
4. Confirm on the validation races: early-$k$ RPS with each prior plugged into the frictionless model (§8 A).
5. Pick the simplest prior whose $\hat\tau^2$ is not worse than the best by more than its bootstrap SE.

**Decision (2026-11-10): v1 uses (d), qualifying + recent-history form, as the two-level state-space model in (x).** Grid alone (c) is kept as the ablation
it has to beat. The teammate block (v) stays deferred, except that it shows up in how form is structured.

**(x) The v1 pace model: two stacked state-space levels** (revised 2026-11-10; replaces the β-regression
version of (d)). Everything is in % of the field-median lap time.

The model has two time scales, each a state-space model, and they are chained:

| Level | Index | State | Dynamics | Observations |
|---|---|---|---|---|
| 1. Within race $r$ | lap $l$ | $\theta_{i,r}$ (static in the race) | — | lap times, with an AR(1) error |
| 2. Across races | race $r$ | $\gamma_{t,r}$ (car), $\delta_{i,r}$ (driver) | AR(1) in $r$ | $\hat\theta^{\text{full}}_{i,r}$ from level 1; qualifying $q_{i,r}$ |

**Level 1: laps (§2–3.2, unchanged).**

$$y_{i,l} = \underbrace{f(\text{lap},\ \text{compound},\ \text{tyre age},\ \text{regime})}_{\text{deterministic: race conditions}} + \theta_{i,r} + \varepsilon_{i,l},\qquad \varepsilon_{i,l} = \phi\,\varepsilon_{i,l-1} + \eta_{i,l}.$$

- $f$ is the deterministic mean from race conditions. Its shared parts cancel out of the ranking (§2.1).
- $\theta_{i,r}$ is this race's pace. Its distribution at lap 0 is the prior that level 2 supplies.
- $\varepsilon$ is the lap-to-lap AR(1) error. Laps $1..k$ update $\theta$ via §3.2.

**Level 2: races.** Split pace into car and driver, $\theta_{i,r} = \gamma_{t(i),r} + \delta_{i,r}$, with

$$\gamma_{t,r} = \rho_r\,\gamma_{t,r-1} + \xi^\gamma_{t,r},\quad \xi^\gamma \sim \mathcal N(0, q_{\gamma,r}^2);\qquad \delta_{i,r} = \delta_{i,r-1} + \xi^\delta_{i,r},\quad \xi^\delta \sim \mathcal N(0, q_\delta^2).$$

- $\gamma$ is centred on the field median (0), so $\rho<1$ means **reversion toward the field**.
- $\rho_r$ and $q_{\gamma,r}$ take two values: in-season ($\rho_{\text{in}}, q_\gamma$) and across a season
  boundary ($\rho_{\text{season}}, q_{\gamma,\text{season}}$). That replaces the ad-hoc "bigger $q$ at
  season boundaries" with something that can be estimated.
- Expected values: $\rho_{\text{in}} \approx 1$ (development is permanent), and $\rho_{\text{season}} < 1$
  (the field converges over a winter under stable rules). At the 2026 regulation reset, $\rho \approx 0$ with
  a wide $q$: form says almost nothing, and the prior falls back on qualifying.
- $\delta$ (driver) **travels with the driver** when they change teams. $\gamma$ stays with the team.

**What this is, as a time series.** Observed with noise, the race-pace series is:

- with $\rho = 1$: a random walk plus noise, so the observations are **IMA(1,1) = ARIMA(0,1,1)**. The
  steady-state Kalman filter is an EWMA with weight $K$, set by $q^2/s^2$. This was the earlier spec.
- with $\rho < 1$: an AR(1) plus noise, so the observations are **ARMA(1,1)**, mean-reverting to the field.

$\rho$ is fitted, so the data chooses between them. A stationary pure MA($q$) of the pace *level* is ruled out:
it would make development shocks transient. A plain average of the last $q$ races is kept only as a crude
baseline.

**Measurements on race weekend $r$.** The filter takes three steps per race, in order, all in one Kalman chain:

1. **Predict (form).** $m_{r\mid r-1} = \rho_r\,m_{r-1}$ (for $\gamma$; identity for $\delta$), and
   $P_{r\mid r-1} = \rho_r^2 P_{r-1} + Q_r$. This prior covers the whole grid jointly, teammates included.
2. **Qualifying update (pre-race, in $\mathcal F_k$).** Treat the qualifying deficit as a noisy reading of race
   pace:
   $$q_{i,r} = \lambda\,\theta_{i,r} + u_{i,r},\qquad u \sim \mathcal N(0,\sigma_q^2).$$
   - $\lambda$ maps race pace to qualifying pace. $\sigma_q^2$ absorbs race-vs-quali differences (tyre
     management, fuel load, set-up).
   - **No usable qualifying time** (pit-lane start, no lap set, DSQ'd from quali): skip the step. The fallback
     is automatic: the prior is form only, with its larger $P$.
   - Wet qualifying before a dry race: inflate $\sigma_q^2$ by a fitted factor (or skip). Settle this in §12.
   - The precision weighting between form and qualifying is no longer a fitted $\beta_2$. It falls out of $P$
     and $\sigma_q^2$.
3. **In-race update (laps $1..k$, level 1).** Prior: $\theta_{i,r}\sim\mathcal N(\bar\theta_{i,r}, \tau^2_{i,r})$,
   where $\bar\theta$ and $\tau^2$ are the post-qualifying mean and variance. §3.2 is then applied unchanged.
   $\tau^2$ is now **per car**: thin history or a missing qualifying time gives a weaker prior.

**After race $r$ finishes (feeds race $r+1$).** Level 1's full-race estimate becomes level 2's observation:

$$\hat\theta^{\text{full}}_{i,r} = \gamma_{t,r} + \delta_{i,r} + e_{i,r},\qquad e \sim \mathcal N(0,\ s^2 + \mathrm{SE}^2_{i,r}).$$

- $\mathrm{SE}^2$ comes from level 1 including the AR inflation (vi). So a race with few clean laps moves the
  form state less, which is the right behaviour.
- $s^2$ is circuit-specific car fit.
- Both teammates observe the same $\gamma$, so this is a small joint update per team.
- Non-finishers contribute only their clean laps, with a correspondingly larger SE. No clean laps means no
  update.
- No double counting: compute $\hat\theta^{\text{full}}$ from the **laps only** (flat prior). Race $r$ then
  gives level 2 two independent readings of $\theta_{i,r}$: qualifying (step 2) and the laps.

**Initialisation and burn-in.** OpenF1 starts in 2023, so the first races of 2023 have no history.

- Start with $m_0 = 0$ and a wide $P_0$.
- A rookie's $\delta$ starts at 0 with a wide variance. A new team's $\gamma$ starts at 0 with a wide variance.
- Score early-2023 races in training only. That's harmless, since validation and test are in 2025.

**Parameters:**

| Level | Parameters |
|---|---|
| 1 | $\phi$, $\sigma_\eta$, tyre terms |
| 2, dynamics | $\rho_{\text{in}}$, $\rho_{\text{season}}$, $q_\gamma$, $q_{\gamma,\text{season}}$, $q_\delta$ |
| 2, measurement | $s$, $\lambda$, $\sigma_q$ (and a wet-quali inflation factor) |

- Level 1 is fitted first, on free-air green laps (ix step 1).
- Level 2 is fitted by **maximising the Kalman prediction-error likelihood** over training races, of both the
  qualifying readings and $\hat\theta^{\text{full}}$.
- $\rho_{\text{season}}$ rests on only two training boundaries (2023→24, 2024→25) times ~10 teams. Treat it as
  weakly identified; fixing it at 1 is the fallback.
- The $\tau^2$ of §3.1 is no longer a separate fitted quantity. It is the filter's post-qualifying variance.
  Check it against the measurement-corrected $\hat\tau^2$ of (vi): that check is now the calibration test of
  the pre-race prior.

**Leakage, restated:** the filter runs **chronologically through every race**, including validation and test
races, using only races that finished before race $r$ plus race $r$'s qualifying. That's what an online
forecaster would know. Hyperparameters are frozen after training. Race $r$'s own laps enter only through step 3,
and only up to lap $k$.

**The check that justifies keeping it** (ix steps 3–4). Compare out-of-sample $\hat\tau^2$ and early-$k$ RPS for:

1. grid-only (c), the $\beta_1 q$ regression;
2. this filter with $\rho \equiv 1$;
3. this filter with $\rho$ fitted.

Keep the simplest one that isn't worse than the best by more than its bootstrap SE. If neither filter beats
(c), the complexity goes, and v1 falls back to (c).

**(xi) First fit of the season model (2026-11-12; revised after the calibration fix).**

*Data.* 69 races (2025 Miami dropped). Train = 2023–24 (46 races, of which the first 6 are burn-in: the filter
runs through them but they add no likelihood). Validation = the first half of 2025 (11 races). The test half
was not touched. Scripts and outputs are in `Integrador 2/m7-analysis/` (`level1.py`, `level2_v2.py`, `m7out/`).

**Level 1 as fitted.**

- **Clean laps:**
  - lap > 1; not an in-lap or out-lap; a dry compound;
  - no SC/VSC/red, and a red flag closes any open SC (2023 São Paulo had a standing restart with no
    "SC in" message);
  - the field median for that lap number ≤ 1.04 × the race median;
  - free air (interval ≥ 1.5 s, or leading);
  - then a 3-MAD residual trim.
- **Ridge regression per race:** lap-number fixed effects, driver pace, and a compound intercept plus a
  tyre-age slope.
  - The compound terms carry weak priors (intercept SD 1%, slope SD 0.1% per lap).
  - Reason: at 2023 Melbourne only cars 2 and 21 ran mediums in clean air. Unregularised, their pace and the
    medium offset traded off, giving θ̂ ≈ +8%.
  - **SEs come from the coefficient covariance** (the contrast to the field mean), × $\sqrt{(1+\phi)/(1-\phi)}$,
    not from $\sigma/\sqrt{n}$. Collinear cases now carry large SEs (Melbourne's two cars: +3.0% ± 0.55).
- **Wet races** (8) give no pace observation.
- **Results:**
  - 1,125 driver-race estimates; median 32 clean laps; median SE 0.12%.
  - Pooled $\hat\phi = 0.47$; median $\hat\sigma_\varepsilon = 0.40\%$ of a lap.
  - Monaco is the outlier: $\hat\sigma_\varepsilon$ = 0.87% in 2024 and 1.58% in 2025.
- **Sanity checks:** team means are plausible (McLaren −0.18 → −0.97 → −1.11%; Red Bull −0.88 → −0.76 →
  −0.39%). $\mathrm{corr}(\hat\theta, q) = 0.51$.

**Model amendment found while fitting.** Qualifying only adds information beyond form if both measure a
*race-specific* pace. So the race-$r$ pace is

$$\theta_{i,r} = \gamma_{t,r} + \delta_{i,r} + c_{t,r} + w_{i,r},\qquad c_{t,r}\sim\mathcal N(0,s_c^2),\quad w_{i,r}\sim\mathcal N(0,s_w^2),$$

with these readings and roles:

- **Qualifying reads** $\lambda(\gamma + \delta + c) + u$.
- **The race reads** $\theta + e$.
- $c_{t,r}$ is the **team × circuit fit**. It is shared by teammates, transient, and redrawn every race.
- $w_{i,r}$ is a race-day effect that only laps reveal.
- The prior handed to level 1 has variance $\tau^2_{i,r} = \mathrm{Var}(\gamma+\delta+c \mid \text{form, quali}) + s_w^2$.

**Fitted parameters** (the chosen model: ρ ≡ 1, no variance scaling; % of a lap):

| Parameter | Value | Meaning |
|---|---|---|
| $q_\gamma$ | 0.071 | In-season car-development step per race |
| $q_{\gamma,\text{season}}$ | 0.21 | Winter step (≈ 3× the in-season step) |
| $q_\delta$ | 0.017 | Driver form is nearly constant |
| $s_c$ | 0.23 | Team × circuit fit |
| $s_w$ | 0.21 | Race-day transient |
| $\lambda$ | 0.97 | Qualifying deficit per unit of race pace |
| $\sigma_q$ | 0.88 | Qualifying noise. This includes Q1/Q2/Q3 track evolution, since $q$ is each car's best time. |
| $P_{0,\gamma}$, $P_{0,\delta}$ | 0.02, 0.03 | **Not identified** (burn-in races are excluded from the likelihood). For the 2026 reset, set them by hand from the field spread (≈ 0.5). |

**Calibration: what was wrong and what fixed it.**

- The first fit looked calibrated on train ($z^2 = 1.05$) but not on validation (0.51).
- By half-season, train was 2.49 in 2023a and 0.4–0.7 afterwards. The variances were being inflated to cover:
  1. burn-in, while the filter learns from nothing;
  2. the Melbourne collinearity artefact (two cars at $z^2 \approx 170$).
- The level-1 ridge plus covariance SEs, and excluding the burn-in from the likelihood, fixed it:
  $s_w$ fell from 0.43 to 0.21, and validation $z^2$ became 0.96.
- **The field-compression hypothesis was rejected.** An online variance scale, multiplying $s_c^2, s_w^2,
  \sigma_q^2$ and updated after each race by its mean standardised innovation with a fitted discount
  $\alpha$, fits $\alpha \to 0$.
- **What remains is mild heavy tails.** After burn-in: excess kurtosis 2.1; 5.2% of errors beyond 2σ
  (Gaussian 4.6%) and 0.6% beyond 3σ (0.27%). The tails concentrate in a few races: 2023 Spa, 2024 Suzuka
  and 2024 Monaco, which is why 2024a sits at $z^2$ = 1.39. A Student-t observation (or a mixture) is a later
  refinement.

**What ρ does.**

- Fitting ρ gives $\rho_{\text{in}} = 0.991$ and $\rho_{\text{season}} \to 1$.
- The log-likelihood gain is 0.4 for 2 extra parameters, and validation is no better.
- **Decision: ρ ≡ 1 (a random walk; IMA(1,1) observations).**

**Prior comparison** (pre-race prior vs. the full-race $\hat\theta$; LS = mean predictive log density;
$\hat\tau^2_c$ = measurement-corrected; $z^2$ = calibration, 1 is ideal):

| Prior | Train LS | Train RMSE | Train $\hat\tau^2_c$ | Train $z^2$ | Val LS | Val RMSE | Val $\hat\tau^2_c$ | Val $z^2$ |
|---|---|---|---|---|---|---|---|---|
| (a) flat | −1.07 | 0.72 | 0.49 | 0.97 | −1.13 | 0.75 | 0.50 | 1.04 |
| (c) grid-only regression | −0.86 | 0.59 | 0.32 | 0.97 | −0.80 | 0.54 | 0.24 | 0.80 |
| (b) form only, ρ ≡ 1 | −0.56 | 0.45 | 0.18 | 1.16 | −0.56 | 0.43 | 0.13 | 0.99 |
| **(d) filter, ρ ≡ 1** | **−0.50** | **0.43** | **0.16** | 1.14 | **−0.49** | **0.41** | **0.11** | **0.96** |
| (d) filter, ρ fitted | −0.50 | 0.43 | 0.16 | 1.13 | −0.50 | 0.41 | 0.11 | 0.98 |
| (e) filter, ρ ≡ 1 + online scale | −0.50 | 0.43 | 0.16 | 1.14 | −0.49 | 0.41 | 0.11 | 0.97 |

**Reading the comparison.**

1. **The check in (x) passes.** On validation the filter's $\hat\tau^2$ is less than half grid-only's (0.11 vs
   0.24), and RMSE drops from 0.54 to 0.41. Form alone already beats qualifying alone by a wide margin.
   Qualifying adds a further 0.02.
2. **The prior's worth in laps.**
   - $\kappa = \sigma_\eta^2/((1-\phi)^2\tau^2)$. With $\sigma_\eta^2 = \sigma_\varepsilon^2(1-\phi^2) \approx 0.125$
     and the model's $\tau^2 \approx 0.12$, that gives **κ ≈ 3.7 clean laps**.
   - About 60% of laps are clean, so the prior dominates for roughly the first 6 race laps and still carries
     weight beyond that. The early-$k$ RPS buckets are where it shows.

---

## 4. Remaining time and the pairwise closed form

### 4.1 The remaining time, ignoring events

$$\mathbb E[R_i \mid \mathcal F_k] = n\,\hat\theta_{i} + \sum_{l=k+1}^{L} (o_{c_{i,l}} + d_{c_{i,l}} a_{i,l}) + m_i\,\lambda + (\text{AR carry-over } \tfrac{\phi(1-\phi^n)}{1-\phi}\hat\varepsilon_{i,k})$$

$$\mathrm{Var}(R_i\mid\mathcal F_k) \approx \underbrace{n^2\, v_{i,k}}_{\text{pace uncertainty}} + \underbrace{n\,\frac{\sigma_\eta^2}{(1-\phi)^2}}_{\text{lap noise (long-run var.)}} + \underbrace{m_i\, s_\Lambda^2}_{\text{stop variability}}$$

- **Pace uncertainty grows like $n^2$, noise like $n$.** Early in the race the forecast is almost entirely
  about how well $\theta$ is known. Late in the race it is about noise and discrete events.
- This tells us where to spend modelling effort for each $k$-bucket.

### 4.2 Pairwise probabilities in closed form

Let $G_{ij} = Z_i - Z_j$ (the order runs from lowest time to highest). Under the Gaussian, no-events model,

$$\Pr(i \text{ ahead of } j\mid\mathcal F_k) = \Phi\!\left(-\frac{\mathbb E G_{ij}}{\sqrt{\mathrm{Var}\, G_{ij}}}\right),\qquad \mathrm{Var}\,G_{ij} = \mathrm{Var}R_i + \mathrm{Var}R_j - 2\,\mathrm{Cov}(R_i,R_j).$$

- Teammates may have positively correlated posteriors, which shrinks $\mathrm{Var}\,G$.
- **The no-events model is Thurstonian** with latent variable $Z_i$, the same family as the chat's pre-race
  suggestion, except the latent here has a physical meaning (seconds).
- The full position matrix has no closed form: it needs MVN orthant probabilities. But drawing $Z \sim$ MVN and
  sorting costs $O(SN\log N)$, which is trivial.
- These pairwise formulas give an **exact unit test for the simulator**. With events switched off, the simulated
  head-to-head frequencies must match $\Phi(\cdot)$.
- **Use the formula over a horizon, never per lap.** Sum the means and variances over the laps that remain and
  take a single $\Phi$. Multiplying per-lap probabilities would answer a different question ("ahead on every
  lap"), and consecutive laps share the same cumulative gap, so they aren't independent.

### 4.3 The v1 sampler (decided 2026-11-10)

The gap is a **random walk**, not an MA process. Every lap's shock stays in the cumulative time with weight 1,
which is why the variance grows like $n$. With AR(1) lap errors, the gap is ARIMA(1,1,0). A safety car is an
intervention on the walk's *level*; it doesn't change the noise.

So v1 draws the events, then one Gaussian per car, with no lap loop:

```
for s in 1..S:
    draw regime events: SC/VSC/red ~ per-circuit hazards over laps k+1..L (restart lap r)
    draw DNFs: per-lap hazard h_i over the remaining laps
    if SC/red: replace the banked T_{i,k} with compressed, order-preserving gaps at r;
               owed stops taken at reduced Λ (red: owed stops cleared); n ← L − r
    Z_i ~ N(E[Z_i | events], Var(Z_i | events))     independently for each running car
    order = sort(Z) among finishers, then retired cars by laps completed ↓
    P += permutation_matrix(order)
P /= S
```

- **Draw per-car totals, not gaps to the leader.** Every gap to the leader contains the leader's noise. Drawing
  gaps independently gives $\mathrm{Var}(\text{gap}_i-\text{gap}_j)=4\sigma^2 n$ instead of the true
  $2\sigma^2 n$, which doubles the variance of every non-leader head-to-head.
- Teammate covariance (deferred) means replacing the independent draws with an MVN draw with a block covariance.
- Cost is $O(SN\log N)$ for the sorts plus the event draws, and it's vectorisable over $S$.

---

## 5. Pit stops: the owed stop and its timing

### 5.1 The owed stop

In a dry race each car must use two different dry compounds. A car that hasn't done so by lap $k$ has at least
one owed stop ($m_i \ge 1$), which is deterministic given $\mathcal F_k$. Further stops ($m_i = 2$) are
probabilistic: use the empirical share by circuit.

### 5.2 Pit loss

$\Lambda \sim \mathcal N(\lambda_{\text{circ}}, s_\Lambda^2)$, measured as total time lost versus staying out
(in-lap plus out-lap). The `pit` endpoint's lane duration is only part of it.

### 5.3 Stop timing from degradation (instead of a purely empirical distribution)

A one-stop driver on compound $c_1$ (age $a_k$) switching to $c_2$ at lap $s$ incurs a remaining tyre cost

$$C(s) = d_1\!\!\sum_{l=k+1}^{s}\!(a_k + l - k) \;+\; d_2\!\!\sum_{l=s+1}^{L}\!(l-s) \;+\; n\text{-independent offsets}.$$

Setting the marginal cost of staying out one more lap equal to the marginal saving on the new tyre,
$d_1(a_k + s - k) = d_2(L - s)$, gives

$$s^* = \frac{d_2\,L - d_1\,(a_k - k)}{d_1 + d_2}.$$

Sanity check: with $d_1 = d_2$, $a_k = 0$, $k = 0$, $s^* = L/2$. ✓

- Simulate $s \sim s^* + \text{noise}$, with the noise SD fitted to how far real stops fall from $s^*$.
- This ties stop timing to the degradation estimates rather than to a separate empirical table.
- Undercut and overcut interactions (reacting to rivals) are **ignored in v1**.

---

## 6. Safety car and VSC: the main source of joint randomness

### 6.1 When it happens

A per-lap hazard $h^{\mathrm{SC}}$ (circuit-dependent, pooled). Over $n$ remaining laps,
$\pi_{\mathrm{SC}} = 1-(1-h^{\mathrm{SC}})^n$. The duration $D_{\mathrm{SC}}$ follows its empirical distribution.

### 6.2 Its effect is compression

At the restart lap $e$, the gaps collapse to a small $g_j$ per position. The race effectively restarts with
the current order:

$$G_{ij} \;\to\; (g_i - g_j) + \sum_{l>e} (\tilde y_{i,l} - \tilde y_{j,l}).$$

- This **erases the accumulated $T_{i,k}$ advantage**, so the matrix becomes a mixture:
  $P = (1-\pi)P^{\text{green}} + \pi\,\mathbb E[P\mid\text{SC}]$.
- A cheap stop under SC (pit loss $\lambda^{\mathrm{SC}} \ll \lambda$) means cars with an owed stop take it
  then. This is a large and correlated reshuffle.
- **Timing matters, so $\mathbb E[P\mid \text{SC}]$ has to average over the restart lap.** After the restart, with
  gap $g$, pace advantage $\delta$ per lap and $n$ laps left,
  $z(n) = (g+\delta n)/(\sigma\sqrt{2n})$. The mean grows like $n$ and the SD like $\sqrt n$.
  - With equal pace ($\delta=0$), $z$ falls as $n$ grows, so an *early* SC hurts the leader most.
    Worked example: $g = 1$ s, $\sigma = 0.4$ s; 2 laps left gives Pr ≈ 0.89, and 20 laps gives ≈ 0.66.
  - With $\delta>0$, $z$ is smallest at $n = g/\delta$. With $\delta = 0.2$ s/lap, the worst case is about 5 laps
    left; 2 laps gives Pr ≈ 0.96 and 20 laps gives ≈ 0.98.

### 6.3 VSC

Speeds are scaled down roughly proportionally, so the gaps are approximately preserved, but the pit loss falls.
Model it as "cheap stop, no compression".

### 6.4 Red flag

Teams may change tyres under a red flag, which satisfies the owed stop for free. A standing or rolling restart
then removes the gaps. Model it as **full compression and owed stops cleared**. In 2023–25, 8 of the 70 GPs had
a red flag, 3 of them on lap 1.

**Frequencies (OpenF1 `race_control`, 70 GPs, 2023–25):** SC in 30 races (8 with 2–3 SCs), VSC in 25, red flag in 8.
The messages are `SAFETY CAR DEPLOYED` / `SAFETY CAR IN THIS LAP` / `VIRTUAL SAFETY CAR DEPLOYED` /
`VIRTUAL SAFETY CAR ENDING` / `RED FLAG`, all with `lap_number`.

### 6.5 Coupling with DNFs

Many safety cars are *caused* by crash retirements. A joint model where each crash DNF triggers an SC with
probability $q$, plus a background hazard, is more coherent than two independent hazards. It is also the
reason teammates' and rivals' outcomes are correlated.

---

## 7. DNFs, classification, lapped cars

### 7.1 Hazard

A per-driver-lap hazard $h_i$, with survival to the flag $(1-h_i)^n$. Converting from a per-race rate $p$:
$h = 1-(1-p)^{1/L}$.

- Lap 1 has a spike, but in-race from $k \ge 1$ that has already happened.
- Shrink team-level rates toward the field: 46 training races is little data per team.
- **Circuit and weather (v1):** partial pooling, with the same logic as the pace prior in §3.3:
  $\hat h_c = B_c\,\bar h_c + (1-B_c)\,\bar h$ with $B_c = n_c/(n_c+\kappa_h)$, where $n_c$ is the number of starts
  at circuit $c$ (or `circuit_type`). Add one multiplier for wet races.
  - About 160 DNFs over the 70 GPs from 2023–25 (2.3 per race, about 11% of starts), so roughly 7 per circuit.
- Split crash vs. mechanical retirements if §6.5 is adopted. OpenF1 has no DNF reason, so the split has to come
  from race-control messages.

### 7.2 Definition matters (a data question)

The FIA classifies a car that covered at least 90% of the winner's distance even if it retired. Possible
definitions of the DNF column:

- (a) **not classified**, or
- (b) OpenF1's `dnf` flag.

These differ for late retirements. **Checked:** a classified retirement has both `dnf = true` and a position
(Melbourne 2023 has five). Unclassified cars and DSQs have a null position. For convention (a), "retired" means a
null position other than a DSQ, and those cars are ordered by `number_of_laps`. DSQ rows are dropped from scoring.

### 7.3 Lapped cars

The classification sorts by **(laps completed ↓, time ↑)**, not by $Z_i$ at lap $L$. In the simulation, when
the leader takes the flag at $T^*$, each car finishes the lap it is on. A car's laps completed are
$\max\{l: T_{i,l-1} < T^*\}$. The error from ignoring this is small at the front but real in the midfield.

---

## 8. Overtaking friction: the structural choice for v1

Sorting $Z$ (§4) assumes **passing is free**: any pace advantage turns into position. In reality, a faster car
within about 1 s loses time in dirty air and passes only with some probability.

- **(A) Frictionless, sort $Z$ (v1 candidate).**
  - Vectorised over all laps at once, with the closed-form checks of §4.2.
  - Biased toward too many position changes, i.e. over-dispersed rows.
  - Check: compare the simulated vs. real distribution of the number of position changes per race.
- **(B) Lap-by-lap with a pass model.**
  - If $\text{gap}_{j\to i} < \delta$ and follower $j$'s draft lap is faster, $j$ passes with probability
    $p(\Delta\text{pace},\text{circuit})$. Otherwise $j$ is held: $y_j \leftarrow y_i + \text{gap}_{\min}$.
  - Sequential in $l$ and conditional on the order, so it is non-Gaussian and has no closed form.
  - $p(\cdot)$ is a logistic model fitted on `position` + `intervals` pairs.

Recommendation: build (A) first. Use the (A)-vs-reality gap in position changes to decide whether (B) is worth
its cost.

---

## 9. Evaluation math

### 9.1 Ranked probability score per row

Order the categories $1,\dots,N,\mathrm{DNF}$ and let $J = N+1$. With cumulative forecast $F_{ij}$ and
cumulative outcome $O_{ij}$,

$$\mathrm{RPS}_i = \frac{1}{J-1}\sum_{j=1}^{J-1}(F_{ij}-O_{ij})^2.$$

It is proper and sensitive to distance (missing by one place costs less than missing by ten). Exclude cars
already retired at $k$ (their score is trivially 0).

### 9.2 "Log-likelihood of the realized order" isn't feasible as stated

The probability of the exact order is around $10^{-10}$ or smaller. Monte Carlo won't hit it, and the
Gaussian model has no closed form. Use instead:

- the **row log score** $\log \hat P_{i,\sigma(i)}$, smoothed: $\hat P \leftarrow (S\hat P + \alpha Q)/(S+\alpha)$,
  with $Q$ the baseline (§10), so an unseen cell is never $-\infty$;
- the **pairwise log loss** $\sum_{i<j}\log \Pr(\text{realized } i\text{-vs-}j)$. It is cheap and uses the
  joint structure the matrix throws away.

### 9.3 Calibration

- Win: $P_{i1}$. Podium: $\sum_{j\le3}P_{ij}$. Points: $\sum_{j\le10}P_{ij}$.
- Expected points: $\sum_j \text{pts}_j P_{ij}$.
- Draw reliability diagrams per $k$-bucket.

### 9.4 Inference and power

The unit is the race. Forecasts at different $k$ within a race are correlated.

- Compare model and baseline **paired by race**: $\Delta_r = \overline{\mathrm{RPS}}^{\text{model}}_r - \overline{\mathrm{RPS}}^{\text{base}}_r$,
  with a bootstrap over races.
- With about 12 test races, $\mathrm{SE}(\bar\Delta) = \mathrm{sd}(\Delta)/\sqrt{12}$. Only fairly large
  improvements will be distinguishable from zero.
- So decide on the validation races, report on the test races, and state the CIs honestly.

---

## 10. Baseline: the direct transition matrix (it needs no Sinkhorn)

Estimate $\Pr(\text{final} = b \mid \text{position } a \text{ at lap } k)$ directly, per $k$-bucket, from the
training races:

$$Q^{(\text{bucket})}_{ab} = \frac{\#\{\text{(race, }k)\text{ in bucket: pos}_k = a,\ \text{final} = b\}}{\#\{\text{pos}_k = a\}},\qquad P^{\text{base}} = \Pi_k\, Q.$$

Here $\Pi_k$ is the permutation for the current order.

- **Column structure:** each (race, $k$) contributes a partial permutation from running positions to final
  categories. If every row $a$ has the same count, then $Q = C/R$ and its column sums are exactly the empirical
  frequency that position $b$ is filled. The structure from §1 holds **without Sinkhorn**.
- The equal-count condition fails only for rows $a > n_k$ (fields shrunk by retirements, or 20 vs. 22 cars).
  That edge, and only that edge, needs IPF/Sinkhorn.
- This partly refines the chat's suggestion. Multiplying per-lap matrices $T_1\cdots T_L$ is unnecessary when
  the $k \to$ finish matrix can be counted directly.
- A per-circuit version is too sparse (about 2 races per circuit). Shrink toward the pooled matrix, or use a
  stickiness index per circuit.

**A second baseline:** "order at $k$ persists with a pit correction", i.e. move owed-stop cars back by
$\lambda$ seconds and re-sort. It is degenerate (0/1 cells), so it is scored on RPS only.

---

## 11. Decisions this math puts on the table

*Rescope (2026-11-10): v1 is the analytic Gaussian core with simulated events (§4.3). Items 1, 3, 4 and 6 are
decided for v1.*

1. ~~Overtaking~~ **Decided: (A) frictionless** for v1. Revisit (B) only if the simulated position changes per
   race are far from the real ones. (§8)
2. ~~The pace prior~~ **Decided: (d), qualifying deficit + recent-history form** (car/driver random walk across
   races). Grid-only (c) is the ablation it must beat on $\hat\tau^2$ and early-$k$ RPS. (§3.3 x)
3. ~~Pit timing~~ **Decided: $s^*$ + empirical noise.** Owed stop switched off in wet races and cleared by red flags. (§5, §6.4)
4. ~~SC and DNF coupling~~ **Decided: independent hazards** in v1; coupling deferred. (§6.5)
5. ~~DNF column definition~~ **Decided: convention (a).** Retired cars take the last positions in an N×N
   matrix (§1). Ranking scores (§9.1) then use $J = N$ categories. §7.2 still matters for *which* cars count as
   retired.
6. ~~Noise~~ **Decided: Gaussian with clean-lap filtering** for v1. Student-*t* deferred. (§2.3)

## 12. What the data has to tell us first (a descriptive pass on 2023–24 before any proposal)

| Quantity | Why it matters |
|---|---|
| $\phi$, $\sigma_\eta$ on clean laps | Sets $k'_{\text{eff}}$ and the noise term of §4 |
| $\tau$ (between-race spread of $\theta$) | Prior weight at small $k$ |
| $d_c$, $o_c$ by compound; residuals vs. age | Whether linear degradation holds |
| $\lambda_{\text{circ}}$, $s_\Lambda$ | Pit correction, baseline 2 |
| SC/VSC hazard per lap, durations | Mixture weight $\pi$ |
| DNF rate, crash vs. mechanical | §7, §6.5 |
| Position changes per race (real) | Target for the A-vs-B decision |

All of this needs M2–M5 run locally (or the data copied in) first: `data/` is empty in this clone. Filter to
`session_name == "Race"` first. The current M2 config also pulls in the 18 sprints.

---

## 13. Implementation v1 (2026-11-19): `race_model/` in the repo

The whole model now lives in `src/f1_race_intelligence/race_model/` and runs as pipeline stages (`m7-train`,
`m7-evaluate --promote`, `m7-update`, `m7-forecast`). Deployment is in `infra/README.md`.

**Reproduction check.** The in-package fit matches §3.3 (xi):

- level 1: φ = 0.470, σ_ε = 0.401;
- level 2: q_γ = 0.073, q_γ,season = 0.21, q_δ = 0.016, s_c = 0.233, s_w = 0.212, σ_q = 0.877, λ = 0.974;
- validation prior: RMSE 0.41, τ̂²_c 0.11, z² 0.97.

The validation split now has 12 races, the first half of the 23 kept 2025 races, rather than 11 calendar slots.

**Decisions made while implementing the sampler (§4.3, §5).**

1. **Every car plans its remaining strategy.**
   - A car that owes its second compound always stops.
   - Any other car stops once more if, at s*, the tyre time saved exceeds the pit loss. It switches to the
     cheapest legal compound.
   - Without this, linear degradation extrapolated over the remaining laps made an owed stop look like a
     free set of new tyres. Shanghai 2025, lap 30: Stroll was given Pr(win) = 0.35.
   - At most one further stop per car.
2. **The spread of stop laps around s* (σ ≈ 8.6 laps) does not enter the tyre cost.**
   - It is mostly error of the s* model.
   - Charging it as degradation added about ±9 s of noise per car.
   - It still decides how a stop meets a safety car or VSC.
3. **Safety-car queues compare cars at the same lap count.**
   - At the cutoff most cars are part-way through lap k (laps_done = k − 1) without being lapped.
   - Treating them as a lap down put them about one lap ahead in the restart queue.
   - Genuinely lapped cars still reach lap e later, so they rank behind. Wave-bys then put them on the lead
     lap.
4. **The lap-noise multiplier is 2.0.** It was chosen on the last 16 training races by RPS. Clean-lap σ_ε
   understates real lap noise.

**Validation (12 races, 2025 first half; k at 10/25/50/75/90% of distance, S = 4,000).** RPS is mean
per running car, lower is better. "Pairwise" is the pairwise log loss.

| Bucket | Model | Persist | Transition | Δ vs transition, 95% CI | Pairwise (model / transition) |
|---|---|---|---|---|---|
| 10% | 0.102 | 0.160 | 0.100 | [−0.007, +0.011] | 0.447 / 0.496 |
| 25% | 0.098 | 0.157 | 0.106 | [−0.027, +0.009] | 0.445 / 0.526 |
| 50% | 0.086 | 0.137 | 0.097 | [−0.029, +0.009] | 0.411 / 0.480 |
| 75% | 0.067 | 0.081 | 0.065 | [−0.009, +0.011] | 0.327 / 0.369 |
| 90% | 0.047 | 0.046 | 0.043 | [−0.001, +0.008] | 0.212 / 0.283 |
| **All** | **0.080** | 0.116 | 0.082 | [−0.010, +0.005] | **0.368** / 0.431 |

- **Phase 3–4 criterion met.** The model beats "persist + owed stop" clearly overall: race-paired Δ = −0.036.
- **Against the transition matrix:** it is ahead on RPS overall and in mid-race, but not significantly with
  12 races. It is better on pairwise log loss in every bucket.
- **Late race (90%):** RPS is slightly worse than both baselines. This is consistent with §8 (A): with
  frictionless passing, rows are over-dispersed when few laps remain. This is the bucket where the
  overtaking model (B) would help.
- **Brier scores:** win 0.027, podium 0.047, points 0.112, DNF 0.043. The DNF forecast mean is 4.6%;
  observed is 4.6%.
- **Test split:** not scored.
