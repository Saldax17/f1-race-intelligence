"""One race's raw records, where to find them, and the lap-k truncation guard.

A :class:`RaceData` holds what the model may read about one Grand Prix. It is
built either from the raw tree (:class:`RaceCatalog`, local disk or S3) or
live from OpenF1 (:func:`load_live_race`). :func:`truncate_at_lap` is the
anti-leakage guard of the scope note: the forecaster only ever receives a
race cut at the moment the leader completed lap ``k``, with the official
result removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from f1_race_intelligence.storage.raw_catalog import RawDataCatalog, RawFileError
from f1_race_intelligence.storage.store import DataStore
from f1_race_intelligence.utils.logging import get_logger

logger = get_logger(__name__)

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
WET_COMPOUNDS = ("INTERMEDIATE", "WET")

RACE_ENDPOINTS = (
    "drivers",
    "laps",
    "stints",
    "pit",
    "race_control",
    "intervals",
    "session_result",
    "starting_grid",
)

# Team names change with sponsors; the car (and its form) does not.
TEAM_ALIASES = {
    "Alfa Romeo": "Sauber",
    "Kick Sauber": "Sauber",
    "Stake F1 Team Kick Sauber": "Sauber",
    "AlphaTauri": "RB",
    "RB": "RB",
    "Racing Bulls": "RB",
    "Visa Cash App RB": "RB",
    "Haas F1 Team": "Haas",
}


def canonical_team(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return TEAM_ALIASES.get(name, name)


@dataclass(frozen=True)
class RaceRef:
    """Identifies one Grand Prix (``session_name == "Race"``; sprints are excluded)."""

    year: int
    meeting_key: int
    session_key: int
    location: str
    date_start: str
    circuit_key: Optional[int] = None
    circuit_type: Optional[str] = None
    qualifying_session_key: Optional[int] = None

    @property
    def tag(self) -> str:
        return f"{self.year} {self.location}"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RaceRef":
        return cls(**{key: data.get(key) for key in cls.__dataclass_fields__})


@dataclass
class RaceData:
    """Everything the model may read about one race.

    ``result`` is the official classification. It is the *label*: present
    for finished historical races, always empty after :func:`truncate_at_lap`.
    """

    ref: RaceRef
    drivers: Dict[int, Dict[str, Any]]
    laps: pd.DataFrame
    stints: List[Dict[str, Any]]
    pit: List[Dict[str, Any]]
    race_control: List[Dict[str, Any]]
    intervals: pd.DataFrame
    grid: List[Dict[str, Any]]
    result: List[Dict[str, Any]] = field(default_factory=list)
    truncated_at: Optional[int] = None

    @property
    def tag(self) -> str:
        return self.ref.tag

    @property
    def wet(self) -> bool:
        """Any intermediate or wet stint: the two-compound rule does not apply."""
        return any(stint.get("compound") in WET_COMPOUNDS for stint in self.stints)

    @property
    def driver_numbers(self) -> List[int]:
        numbers = set(self.drivers)
        numbers |= {int(d) for d in self.laps["driver_number"].unique()} if len(self.laps) else set()
        numbers |= {int(r["driver_number"]) for r in self.result if r.get("driver_number") is not None}
        return sorted(numbers)

    def team_of(self, driver_number: int) -> Optional[str]:
        return canonical_team(self.drivers.get(driver_number, {}).get("team_name"))

    def acronym_of(self, driver_number: int) -> str:
        return self.drivers.get(driver_number, {}).get("name_acronym") or str(driver_number)

    def leader_laps(self) -> int:
        return int(self.laps["lap_number"].max()) if len(self.laps) else 0


# -- building frames from raw records ---------------------------------------


def laps_frame(records: Iterable[Mapping[str, Any]], max_laps: Optional[Mapping[int, int]] = None) -> pd.DataFrame:
    """Lap records as a frame with an end time per lap.

    ``t_end`` is ``date_start + lap_duration``, or the next lap's
    ``date_start`` when the duration is missing (lap 1, red-flag laps).
    ``max_laps`` drops the trailing null lap the feed adds after the flag.
    """
    columns = ["driver_number", "lap_number", "lap_duration", "date_start", "is_pit_out_lap"]
    frame = pd.DataFrame(list(records))
    if frame.empty:
        return pd.DataFrame(columns=columns + ["t_start", "t_end"])
    for column in columns:
        if column not in frame:
            frame[column] = None
    frame = frame[columns].copy()
    frame["driver_number"] = frame["driver_number"].astype(int)
    frame["lap_number"] = frame["lap_number"].astype(int)
    frame["lap_duration"] = pd.to_numeric(frame["lap_duration"], errors="coerce")
    frame["is_pit_out_lap"] = frame["is_pit_out_lap"].fillna(False).astype(bool)
    if max_laps:
        limit = frame["driver_number"].map(max_laps)
        frame = frame[limit.isna() | (frame["lap_number"] <= limit)]
    frame = frame.sort_values(["driver_number", "lap_number"]).drop_duplicates(["driver_number", "lap_number"])
    start = pd.to_datetime(frame["date_start"], format="ISO8601", utc=True, errors="coerce")
    frame["t_start"] = start.astype("datetime64[ns, UTC]")
    end = frame["t_start"] + pd.to_timedelta(frame["lap_duration"], unit="s")
    following = frame.groupby("driver_number")["t_start"].shift(-1)
    frame["t_end"] = end.fillna(following)
    return frame.reset_index(drop=True)


def intervals_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(records))
    if frame.empty:
        return pd.DataFrame(columns=["date", "driver_number", "interval", "gap_to_leader"])
    for column in ("interval", "gap_to_leader"):
        if column not in frame:
            frame[column] = None
    frame["date"] = pd.to_datetime(frame["date"], format="ISO8601", utc=True, errors="coerce").astype(
        "datetime64[ns, UTC]"
    )
    frame["driver_number"] = frame["driver_number"].astype(int)
    # "+1 LAP" strings become NaN: a lapped car is never in a fight for position.
    frame["interval"] = pd.to_numeric(frame["interval"], errors="coerce")
    frame["gap_to_leader"] = pd.to_numeric(frame["gap_to_leader"], errors="coerce")
    return frame[["date", "driver_number", "interval", "gap_to_leader"]].dropna(subset=["date"]).reset_index(drop=True)


def qualifying_grid(grid: Iterable[Mapping[str, Any]], qualifying_session_key: Optional[int]) -> List[Dict[str, Any]]:
    """The grid rows of the Qualifying session (drops the sprint grid)."""
    rows = [dict(row) for row in grid if isinstance(row, Mapping)]
    if qualifying_session_key is None:
        keys = {row.get("session_key") for row in rows}
        return rows if len(keys) <= 1 else []
    return [row for row in rows if row.get("session_key") == qualifying_session_key]


def build_race(ref: RaceRef, records: Mapping[str, List[Any]]) -> RaceData:
    """Assemble a :class:`RaceData` from per-endpoint record lists."""
    result = [dict(r) for r in records.get("session_result") or [] if isinstance(r, Mapping)]
    max_laps = {int(r["driver_number"]): int(r["number_of_laps"] or 0) for r in result if r.get("number_of_laps") is not None}
    drivers = {
        int(d["driver_number"]): dict(d)
        for d in records.get("drivers") or []
        if isinstance(d, Mapping) and d.get("driver_number") is not None
    }
    return RaceData(
        ref=ref,
        drivers=drivers,
        laps=laps_frame(records.get("laps") or [], max_laps or None),
        stints=[dict(s) for s in records.get("stints") or [] if isinstance(s, Mapping)],
        pit=[dict(p) for p in records.get("pit") or [] if isinstance(p, Mapping)],
        race_control=sorted(
            (dict(m) for m in records.get("race_control") or [] if isinstance(m, Mapping)),
            key=lambda m: m.get("date") or "",
        ),
        intervals=intervals_frame(records.get("intervals") or []),
        grid=qualifying_grid(records.get("starting_grid") or [], ref.qualifying_session_key),
        result=result,
    )


# -- the raw tree -------------------------------------------------------------


class RaceCatalog:
    """Finds Grands Prix in the raw tree and loads them."""

    def __init__(self, store: DataStore, raw_path: str = "data/raw") -> None:
        self._catalog = RawDataCatalog(raw_path, store=store)

    def sessions(self, years: Optional[Sequence[int]] = None) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for year in years or [None]:
            for raw_file in self._catalog.discover(["sessions"], year=year):
                try:
                    out.extend(r for r in self._catalog.load(raw_file).records if isinstance(r, Mapping))
                except RawFileError as exc:
                    logger.warning("sessions_file_unreadable", extra={"path": raw_file.uri, "error": str(exc)})
        unique = {s["session_key"]: s for s in out if s.get("session_key") is not None}
        return list(unique.values())

    def races(self, years: Optional[Sequence[int]] = None, exclude: Sequence[str] = ()) -> List[RaceRef]:
        sessions = self.sessions(years)
        qualifying = {
            s["meeting_key"]: s["session_key"] for s in sessions if s.get("session_name") == "Qualifying"
        }
        refs = []
        for s in sessions:
            if s.get("session_name") != "Race" or s.get("is_cancelled"):
                continue
            ref = RaceRef(
                year=int(s["year"]),
                meeting_key=int(s["meeting_key"]),
                session_key=int(s["session_key"]),
                location=str(s.get("location") or s.get("circuit_short_name") or s["session_key"]),
                date_start=str(s.get("date_start") or ""),
                circuit_key=s.get("circuit_key"),
                circuit_type=s.get("circuit_type"),
                qualifying_session_key=qualifying.get(s["meeting_key"]),
            )
            if ref.tag in set(exclude):
                continue
            refs.append(ref)
        return sorted(refs, key=lambda r: r.date_start)

    def load(self, ref: RaceRef) -> RaceData:
        records: Dict[str, List[Any]] = {}
        for raw_file in self._catalog.discover(
            RACE_ENDPOINTS, year=ref.year, meeting_key=ref.meeting_key, session_key=ref.session_key
        ):
            try:
                records.setdefault(raw_file.endpoint, []).extend(self._catalog.load(raw_file).records)
            except RawFileError as exc:
                logger.warning("race_file_unreadable", extra={"path": raw_file.uri, "error": str(exc)})
        missing = [ep for ep in ("laps", "stints", "drivers") if not records.get(ep)]
        if missing:
            raise LookupError(f"{ref.tag}: no raw {', '.join(missing)} for session {ref.session_key}")
        return build_race(ref, records)


def load_live_race(client: Any, session_key: int) -> RaceData:
    """Fetch one race straight from OpenF1 (in progress or finished).

    The official result is not requested: a live forecast has no label.
    """
    session = next(iter(client.get_sessions(session_key=session_key) or []), None)
    if not session:
        raise LookupError(f"OpenF1 has no session {session_key}")
    meeting_sessions = client.get_sessions(meeting_key=session["meeting_key"]) or []
    qualifying = next((s["session_key"] for s in meeting_sessions if s.get("session_name") == "Qualifying"), None)
    ref = RaceRef(
        year=int(session["year"]),
        meeting_key=int(session["meeting_key"]),
        session_key=int(session_key),
        location=str(session.get("location") or session_key),
        date_start=str(session.get("date_start") or ""),
        circuit_key=session.get("circuit_key"),
        circuit_type=session.get("circuit_type"),
        qualifying_session_key=qualifying,
    )

    def fetch(method: str, **kwargs: Any) -> List[Any]:
        try:
            return getattr(client, method)(**kwargs) or []
        except Exception as exc:  # a missing endpoint (pit 404) must not stop a forecast
            logger.warning("live_endpoint_failed", extra={"method": method, "error": str(exc)})
            return []

    records = {
        "drivers": fetch("get_drivers", session_key=session_key),
        "laps": fetch("get_laps", session_key=session_key),
        "stints": fetch("get_stints", session_key=session_key),
        "pit": fetch("get_pit", session_key=session_key),
        "race_control": fetch("get_race_control", session_key=session_key),
        "intervals": fetch("get_intervals", session_key=session_key),
        "starting_grid": fetch("get_starting_grid", meeting_key=ref.meeting_key),
    }
    return build_race(ref, records)


# -- the anti-leakage guard -----------------------------------------------------


def lap_cutoff_time(race: RaceData, k: int) -> Optional[pd.Timestamp]:
    """When the leader finished lap ``k``: the information horizon of a lap-k forecast."""
    ends = race.laps.loc[race.laps["lap_number"] == k, "t_end"].dropna()
    return ends.min() if len(ends) else None


def truncate_at_lap(race: RaceData, k: int) -> RaceData:
    """The race as it was known when the leader completed lap ``k``.

    Laps, pit stops, race-control messages and intervals after that moment
    are removed, stints are cut at each car's last completed lap, and the
    official result is dropped. This is the only form of a race the
    forecaster accepts.
    """
    cutoff = lap_cutoff_time(race, k)
    if cutoff is None:
        raise ValueError(f"{race.tag}: lap {k} was never completed")
    laps = race.laps[race.laps["t_end"].notna() & (race.laps["t_end"] <= cutoff)].copy()
    done = laps.groupby("driver_number")["lap_number"].max().to_dict()

    stints = []
    for stint in race.stints:
        driver, start = stint.get("driver_number"), stint.get("lap_start")
        last = done.get(driver)
        if last is None or start is None or start > last + 1:
            continue
        cut = dict(stint)
        cut["lap_end"] = min(stint.get("lap_end") or last, max(last, start))
        stints.append(cut)

    def before(records: List[Dict[str, Any]], key: str = "date") -> List[Dict[str, Any]]:
        out = []
        for record in records:
            when = pd.to_datetime(record.get(key), format="ISO8601", utc=True, errors="coerce")
            if when is not pd.NaT and not pd.isna(when) and when <= cutoff:
                out.append(record)
        return out

    intervals = race.intervals[race.intervals["date"] <= cutoff].copy()
    return replace(
        race,
        laps=laps.reset_index(drop=True),
        stints=stints,
        pit=before(race.pit),
        race_control=before(race.race_control),
        intervals=intervals.reset_index(drop=True),
        result=[],
        truncated_at=k,
    )


# -- the label ------------------------------------------------------------------


@dataclass(frozen=True)
class FinalOrder:
    """The realised order under convention (a).

    ``position`` maps driver → 1-based place: classified cars by position,
    then unclassified retirements by laps completed (most first). DSQ and
    DNS cars are listed in ``excluded`` and carry no position. ``retired`` is
    exactly the cars for which :func:`is_retirement` holds; ``classified_dnf``
    (classified despite ``dnf=True``) is informational only.
    """

    position: Dict[int, int]
    retired: frozenset
    excluded: frozenset
    classified_dnf: frozenset = frozenset()


def is_excluded(row: Mapping[str, Any]) -> bool:
    """DSQ (a stewards' decision the model cannot forecast) or DNS: never scored."""
    return bool(row.get("dsq") or row.get("dns"))


def is_retirement(row: Mapping[str, Any]) -> bool:
    """The model's retirement event, convention (a): the car was **not classified**.

    The single definition shared by the hazard fit (``events.fit_events``),
    the simulator (a retired car is ranked after every finisher) and the
    evaluation (``FinalOrder.retired``). A car with ``dnf=True`` and a
    position covered at least 90% of the distance and was classified: for the
    position matrix it finished in that place, so it is *not* a retirement
    here (it is kept in ``FinalOrder.classified_dnf`` for information).
    """
    return row.get("position") is None and not is_excluded(row)


def final_order(result: Sequence[Mapping[str, Any]]) -> FinalOrder:
    rows = [r for r in result if r.get("driver_number") is not None]
    excluded = {int(r["driver_number"]) for r in rows if is_excluded(r)}
    classified = sorted(
        (r for r in rows if r.get("position") is not None and not is_excluded(r)),
        key=lambda r: r["position"],
    )
    unclassified = sorted(
        (r for r in rows if is_retirement(r)),
        key=lambda r: -(r.get("number_of_laps") or 0),
    )
    order = [int(r["driver_number"]) for r in classified + unclassified]
    return FinalOrder(
        position={d: i + 1 for i, d in enumerate(order)},
        retired=frozenset(int(r["driver_number"]) for r in unclassified),
        excluded=frozenset(excluded),
        classified_dnf=frozenset(int(r["driver_number"]) for r in classified if r.get("dnf")),
    )


def scheduled_laps(race: RaceData, known: Optional[Mapping[str, int]] = None) -> int:
    """Race distance in laps.

    Prefers the circuit's distance from earlier races (``known``, keyed by
    location), then the winner's laps in the result. A live race at a new
    circuit falls back to the most laps seen so far plus nothing: callers
    should pass ``known`` or an explicit override.
    """
    if known and race.ref.location in known:
        return int(known[race.ref.location])
    laps = [r.get("number_of_laps") or 0 for r in race.result if r.get("position") == 1]
    if laps and laps[0]:
        return int(laps[0])
    return int(race.leader_laps())


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def race_to_forecast_payload(race: RaceData) -> Dict[str, Any]:
    """Serialize a ``RaceData`` into a label-free JSON-safe snapshot for inference.

    The official ``session_result`` is intentionally excluded so inference
    payloads (including MLflow Pyfunc inputs) can never carry post-race labels.
    """
    laps_records: List[Dict[str, Any]] = []
    if race.laps is not None and not race.laps.empty:
        for row in race.laps.to_dict(orient="records"):
            dur = row.get("lap_duration")
            date_start = row.get("date_start")
            if (date_start is None or pd.isna(date_start)) and row.get("t_start") is not None and not pd.isna(row.get("t_start")):
                ts = pd.Timestamp(row["t_start"])
                date_start = (ts.tz_convert("UTC") if ts.tzinfo is not None else ts).isoformat()
            laps_records.append(
                {
                    "driver_number": int(row["driver_number"]),
                    "lap_number": int(row["lap_number"]),
                    "lap_duration": None if dur is None or pd.isna(dur) else float(dur),
                    "date_start": None if date_start is None or pd.isna(date_start) else str(date_start),
                    "is_pit_out_lap": bool(row.get("is_pit_out_lap", False)),
                }
            )

    interval_records: List[Dict[str, Any]] = []
    if race.intervals is not None and not race.intervals.empty:
        for row in race.intervals.to_dict(orient="records"):
            when = row.get("date")
            if when is None or pd.isna(when):
                continue
            ts = pd.Timestamp(when)
            when_iso = (ts.tz_convert("UTC") if ts.tzinfo is not None else ts).isoformat()
            iv = row.get("interval")
            gap = row.get("gap_to_leader")
            interval_records.append(
                {
                    "date": when_iso,
                    "driver_number": int(row["driver_number"]),
                    "interval": None if iv is None or pd.isna(iv) else float(iv),
                    "gap_to_leader": None if gap is None or pd.isna(gap) else float(gap),
                }
            )

    return {
        "ref": race.ref.to_dict(),
        "drivers": [dict(d) for _, d in sorted(race.drivers.items(), key=lambda kv: int(kv[0]))],
        "laps": laps_records,
        "stints": [dict(s) for s in race.stints],
        "pit": [dict(p) for p in race.pit],
        "race_control": [dict(m) for m in race.race_control],
        "intervals": interval_records,
        "starting_grid": [dict(g) for g in race.grid],
    }


def race_from_forecast_payload(payload: Mapping[str, Any]) -> RaceData:
    """Reconstruct a label-free ``RaceData`` from a serialized forecast payload."""
    if not isinstance(payload, Mapping):
        raise ValueError("race forecast payload must be a JSON object")
    ref_raw = payload.get("ref")
    if not isinstance(ref_raw, Mapping):
        raise ValueError("race forecast payload requires a 'ref' object")
    for required_ref in ("year", "meeting_key", "session_key", "location", "date_start"):
        if required_ref not in ref_raw:
            raise ValueError(f"race forecast payload 'ref' missing required field {required_ref!r}")
    ref = RaceRef.from_dict(ref_raw)
    records: Dict[str, List[Any]] = {}
    for endpoint in ("drivers", "laps", "stints", "pit", "race_control", "intervals", "starting_grid"):
        raw_list = payload.get(endpoint, [])
        if raw_list is None:
            raw_list = []
        if not isinstance(raw_list, list):
            raise ValueError(f"race forecast payload field {endpoint!r} must be a list")
        records[endpoint] = raw_list
    # Never accept or propagate session_result labels through inference payloads.
    records["session_result"] = []
    return build_race(ref, records)
