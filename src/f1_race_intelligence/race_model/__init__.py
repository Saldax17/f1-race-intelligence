"""M7: in-race forecast of the finishing order, from the end of lap k.

The output is an N×N matrix ``P[i, j] = Pr(driver i finishes j-th | laps 1..k)``
plus ``Pr(DNF)`` per driver. Retired cars take the last positions (convention
(a)), so the matrix is doubly stochastic by construction.

Modules, in the order data flows through them:

``racedata``   one race's raw records, and the lap-k truncation guard
``regime``     green / SC / VSC / red per lap, from race-control messages
``pace``       level 1: clean laps, per-race pace θ̂ with SEs, tyre terms, AR(1)
``season``     level 2: car/driver form across races + qualifying (Kalman)
``events``     safety-car, VSC, red-flag and DNF hazards, pit loss, stop timing
``forecast``   the in-race posterior and the Monte Carlo position matrix
``scoring``    RPS, pairwise log loss, calibration
``baselines``  "order persists + owed stop" and the transition matrix Q
``artifact``   the versioned model file and the ``latest`` pointer
``runner``     train / evaluate / update / forecast, as pipeline stages
``lambda_handler``  the AWS Lambda entry point for live and on-demand forecasts

The math and every decision behind it are in ``f1-model-math.md``.
"""
