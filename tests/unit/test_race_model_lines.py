"""Racing-line geometry and progress-based passes, on a synthetic circular track."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from f1_race_intelligence.race_model import lines
from f1_race_intelligence.race_model.racedata import RaceRef, build_race

R_UNITS = 1000.0  # circle radius in coordinate units (decimetres -> 100 m)
T0 = datetime(2024, 3, 2, 15, 0, tzinfo=timezone.utc)


def circle(theta, radius=R_UNITS):
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def test_reference_line_length_and_projection():
    theta = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    line = lines.ReferenceLine(points=lines.resample_path(np.vstack([circle(theta), circle(theta[:1])]), 5.0),
                               step=5.0, laps_used=1)
    assert line.length == pytest.approx(2 * np.pi * R_UNITS, rel=0.01)
    probe = np.vstack([circle(np.array([np.pi / 2])), circle(np.array([np.pi / 2]), R_UNITS + 30)])
    proj = line.project(probe)
    assert proj["s"][0] == pytest.approx(np.pi / 2 * R_UNITS, rel=0.02)
    assert abs(proj["d"][0]) < 1 and abs(abs(proj["d"][1]) - 30) < 1  # 30 units off the line, to one side


def synthetic_two_car_race(laps=6, lap_seconds=60.0):
    """Car 1 leads; car 2 is faster and passes it during lap 4, half a lap in."""
    speed = 2 * np.pi * R_UNITS / lap_seconds
    rows, lap_rows = [], []
    for car, (start, pace) in {1: (2.0, 1.0), 2: (0.0, 1.03)}.items():
        t = np.arange(0, laps * lap_seconds / pace, 0.25)
        dist = start * speed * 0 + t * speed * pace - (0 if car == 2 else -speed * 4.0)  # car 1 starts 4 s ahead
        xy = circle(dist / R_UNITS)
        for ti, (x, y) in zip(t, xy):
            rows.append({"date": T0 + timedelta(seconds=float(ti)), "driver_number": car, "x": x, "y": y, "z": 1})
        lap_len = 2 * np.pi * R_UNITS
        for lap in range(1, laps + 1):
            s0 = (lap - 1) * lap_len - (0 if car == 2 else speed * 4.0)
            t_start = max(0.0, s0 / (speed * pace))
            lap_rows.append({"driver_number": car, "lap_number": lap, "lap_duration": lap_seconds / pace,
                             "date_start": (T0 + timedelta(seconds=t_start)).isoformat(), "is_pit_out_lap": False})
    ref = RaceRef(2024, 1, 1, "Circle", T0.isoformat())
    race = build_race(ref, {"drivers": [{"driver_number": 1}, {"driver_number": 2}], "laps": lap_rows, "stints": [],
                            "pit": [], "race_control": [], "intervals": [], "session_result": [], "starting_grid": []})
    return race, pd.DataFrame(rows)


def test_a_pass_on_track_is_found_once_and_attributed():
    race, location = synthetic_two_car_race()
    samples = lines.tag_laps(lines.positions_with_telemetry(location, None), race)
    theta = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    line = lines.ReferenceLine(points=lines.resample_path(np.vstack([circle(theta), circle(theta[:1])]), 5.0),
                               step=5.0, laps_used=1)
    samples = lines.race_distance(samples, line, metres_per_unit=0.1)
    passes = lines.on_track_passes(samples, race, lap_length_m=line.length * 0.1)
    assert len(passes) == 1
    assert (passes.iloc[0]["passer"], passes.iloc[0]["passed"]) == (2, 1)


def test_sentinel_positions_are_dropped():
    frame = pd.DataFrame({"date": pd.to_datetime([T0, T0 + timedelta(seconds=1)], utc=True), "driver_number": [1, 1],
                          "x": [0, 10], "y": [0, 5], "z": [0, 1]})
    assert len(lines.positions_with_telemetry(frame, None)) == 1
