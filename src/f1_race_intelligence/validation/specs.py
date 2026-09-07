"""What each OpenF1 endpoint is expected to look like.

This module is data, not logic: every expectation below was measured
against real OpenF1 responses (2023, 2024 and 2025 races) rather than
assumed from the documentation, because the two disagree in places that
matter. The checks that consume these specs live in
:mod:`f1_race_intelligence.validation.validators`, so tightening a range
or adding an endpoint is an edit here and nothing else.

Three bands of expectation, mapping onto the three severities:

* ``minimum`` / ``maximum`` — physically impossible outside this, so a
  violation is an ERROR.
* ``plausible`` — possible but surprising; a violation is a WARNING.
* ``documented`` / ``allowed`` — what the API documents or what we have
  observed; outside it is an INFO, because the source is allowed to
  surprise us without being wrong.

Measured facts worth keeping in view while reading:

* ``car_data.throttle`` and ``car_data.brake`` reach 104 although both are
  documented as 0–100, and ``brake`` only ever takes {0, 100, 104}.
* ``car_data.drs`` is a state code — {0,1,2,3,8,10,12,14} observed — not a
  boolean.
* ``intervals.gap_to_leader`` is a float, ``None``, *or* a string like
  ``"+1 LAP"`` for lapped cars (10.3% of records in the race sampled).
* ``race_control`` is legitimately sparse: ``qualifying_phase`` was 100%
  null in a race, ``driver_number`` 78.7%.
* Every natural key below was verified to be unique on real sessions, which
  is what makes duplicates an ERROR rather than a WARNING.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple, Type

# Tyre compounds OpenF1 reports. A value outside this set is reported as
# INFO, not an error: F1 can introduce a new compound name at any time.
KNOWN_COMPOUNDS: FrozenSet[str] = frozenset(
    {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN", "TEST_UNKNOWN"}
)

# DRS state codes seen in real telemetry. 0-3 mean "not open", 8/10/12/14
# mean available or open; the exact meaning is not needed to validate them.
KNOWN_DRS_CODES: FrozenSet[int] = frozenset({0, 1, 2, 3, 8, 10, 12, 14})

# Lapped cars report their gap as text instead of a number.
LAPPED_GAP_PATTERN = r"^\+\d+ LAPS?$"

NUMBER: Tuple[Type, ...] = (int, float)


@dataclass(frozen=True)
class FieldSpec:
    """What one field of one endpoint may contain."""

    types: Tuple[Type, ...] = NUMBER
    required: bool = False
    """The key must be present on every record."""

    nullable: bool = True
    """``None`` is an acceptable value."""

    sparse: bool = False
    """Nulls are normal here, however many there are.

    Set for fields that only apply to some records — ``race_control`` only
    names a driver for driver-specific messages — so a high null ratio is
    reported as INFO instead of being measured against the threshold.
    """

    minimum: Optional[float] = None
    maximum: Optional[float] = None
    exclusive_minimum: bool = False
    """``minimum`` is a bound the value must exceed, not merely reach.

    Used for durations, where zero is as impossible as a negative number.
    """

    plausible: Optional[Tuple[float, float]] = None
    documented: Optional[Tuple[float, float]] = None
    allowed: Optional[FrozenSet[Any]] = None
    string_pattern: Optional[str] = None
    """Shape a string value is expected to take, when strings are allowed."""

    note: str = ""
    """Why this expectation is what it is; surfaced in report details."""


@dataclass(frozen=True)
class EndpointSpec:
    """What a whole endpoint's payload is expected to look like."""

    name: str
    fields: Mapping[str, FieldSpec] = field(default_factory=dict)

    natural_key: Optional[Tuple[str, ...]] = None
    """Fields that together identify a record uniquely.

    ``None`` means no defensible key exists, and the duplicate rule is
    skipped rather than guessed at.
    """

    natural_key_note: str = ""

    timestamp_fields: Tuple[str, ...] = ()
    """Fields that must parse as ISO 8601 timestamps."""

    order_field: Optional[str] = None
    """Field the records are expected to be ordered by."""

    order_group: Optional[str] = None
    """Field records are grouped by before checking order (usually the driver)."""

    session_scoped: bool = True
    """Records carry a ``session_key`` that must exist in the sessions catalog."""

    @property
    def required_fields(self) -> Tuple[str, ...]:
        return tuple(name for name, spec in self.fields.items() if spec.required)


def _identifier(maximum: Optional[float] = None, minimum: float = 1) -> FieldSpec:
    """A key field: always present, never null, always a positive integer."""
    return FieldSpec(types=(int,), required=True, nullable=False, minimum=minimum, maximum=maximum)


def _timestamp(required: bool = True, nullable: bool = False) -> FieldSpec:
    return FieldSpec(types=(str,), required=required, nullable=nullable)


# Driver numbers are 1-99 by regulation; 0 has never been issued but the
# range is kept generous rather than clever.
_DRIVER_NUMBER = _identifier(maximum=99)

_SESSION_KEY = _identifier(minimum=0, maximum=None)
_MEETING_KEY = FieldSpec(types=(int,), required=False, nullable=False, minimum=0)


# Descriptive fields carry no bounds worth enforcing, but are listed so that
# an unknown field genuinely means a new one rather than one we skipped.
_TEXT = FieldSpec(types=(str,))

MEETINGS = EndpointSpec(
    name="meetings",
    natural_key=("meeting_key",),
    natural_key_note="One record per race weekend.",
    timestamp_fields=("date_start", "date_end"),
    session_scoped=False,
    fields={
        "meeting_key": _identifier(minimum=0),
        "year": FieldSpec(types=(int,), required=True, nullable=False, minimum=1950, maximum=2100),
        "meeting_name": FieldSpec(types=(str,), required=True, nullable=False),
        "meeting_official_name": _TEXT,
        "country_name": _TEXT,
        "country_code": _TEXT,
        "country_key": FieldSpec(types=(int,), minimum=0),
        "country_flag": _TEXT,
        "circuit_key": FieldSpec(types=(int,), minimum=0),
        "circuit_short_name": _TEXT,
        "circuit_image": _TEXT,
        "circuit_info_url": _TEXT,
        "circuit_type": _TEXT,
        "location": _TEXT,
        "gmt_offset": _TEXT,
        "date_start": _timestamp(),
        "date_end": _timestamp(required=False, nullable=True),
        "is_cancelled": FieldSpec(types=(bool,), nullable=True),
    },
)

SESSIONS = EndpointSpec(
    name="sessions",
    natural_key=("session_key",),
    natural_key_note="One record per session of a weekend.",
    timestamp_fields=("date_start", "date_end"),
    session_scoped=False,
    fields={
        "session_key": _SESSION_KEY,
        "meeting_key": _identifier(minimum=0),
        "year": FieldSpec(types=(int,), required=True, nullable=False, minimum=1950, maximum=2100),
        "session_name": FieldSpec(types=(str,), required=True, nullable=False),
        "session_type": FieldSpec(types=(str,), required=True, nullable=False),
        "date_start": _timestamp(),
        "date_end": _timestamp(),
        "circuit_key": FieldSpec(types=(int,), minimum=0),
        "circuit_short_name": _TEXT,
        "country_name": _TEXT,
        "country_code": _TEXT,
        "country_key": FieldSpec(types=(int,), minimum=0),
        "location": _TEXT,
        "gmt_offset": _TEXT,
        "is_cancelled": FieldSpec(types=(bool,), nullable=True),
    },
)

DRIVERS = EndpointSpec(
    name="drivers",
    natural_key=("session_key", "driver_number"),
    natural_key_note="One record per driver entered in a session.",
    fields={
        "session_key": _SESSION_KEY,
        "meeting_key": _MEETING_KEY,
        "driver_number": _DRIVER_NUMBER,
        "full_name": _TEXT,
        "first_name": _TEXT,
        "last_name": _TEXT,
        "broadcast_name": _TEXT,
        "name_acronym": _TEXT,
        "team_name": _TEXT,
        "team_colour": _TEXT,
        "country_code": _TEXT,
        "headshot_url": _TEXT,
    },
)

LAPS = EndpointSpec(
    name="laps",
    natural_key=("session_key", "driver_number", "lap_number"),
    natural_key_note="Verified unique across 2023/2024/2025 races.",
    timestamp_fields=("date_start",),
    order_field="lap_number",
    order_group="driver_number",
    fields={
        "session_key": _SESSION_KEY,
        "meeting_key": _MEETING_KEY,
        "driver_number": _DRIVER_NUMBER,
        # Monaco, the longest race on the calendar, is 78 laps.
        "lap_number": _identifier(maximum=100),
        "lap_duration": FieldSpec(
            minimum=0,
            exclusive_minimum=True,
            plausible=(40.0, 400.0),
            note="Observed 82-191 s across races; a safety car stretches laps, a red flag can distort them.",
        ),
        "duration_sector_1": FieldSpec(minimum=0, exclusive_minimum=True, plausible=(10.0, 200.0)),
        "duration_sector_2": FieldSpec(minimum=0, exclusive_minimum=True, plausible=(10.0, 200.0)),
        "duration_sector_3": FieldSpec(minimum=0, exclusive_minimum=True, plausible=(10.0, 200.0)),
        "i1_speed": FieldSpec(types=(int,), minimum=0, maximum=400, note="24.5% null in the race sampled."),
        "i2_speed": FieldSpec(types=(int,), minimum=0, maximum=400),
        "st_speed": FieldSpec(types=(int,), minimum=0, maximum=400, note="14.7% null in the race sampled."),
        "is_pit_out_lap": FieldSpec(types=(bool,)),
        "date_start": _timestamp(required=False, nullable=True),
        "segments_sector_1": FieldSpec(types=(list,)),
        "segments_sector_2": FieldSpec(types=(list,)),
        "segments_sector_3": FieldSpec(types=(list,)),
    },
)

CAR_DATA = EndpointSpec(
    name="car_data",
    natural_key=("session_key", "driver_number", "date"),
    natural_key_note="Telemetry samples; verified unique per driver and instant.",
    timestamp_fields=("date",),
    order_field="date",
    fields={
        "session_key": _SESSION_KEY,
        "meeting_key": _MEETING_KEY,
        "driver_number": _DRIVER_NUMBER,
        "date": _timestamp(),
        # The outright F1 speed record is around 372 km/h.
        "speed": FieldSpec(types=(int,), minimum=0, maximum=400),
        # The current formula is limited to 15 000 rpm by regulation.
        "rpm": FieldSpec(types=(int,), minimum=0, maximum=20000),
        "n_gear": FieldSpec(types=(int,), minimum=0, maximum=8),
        "throttle": FieldSpec(
            types=(int,),
            minimum=0,
            maximum=110,
            documented=(0, 100),
            note="Values of 104 occur in real telemetry despite the documented 0-100 range.",
        ),
        "brake": FieldSpec(
            types=(int,),
            minimum=0,
            maximum=110,
            documented=(0, 100),
            note="Only {0, 100, 104} observed.",
        ),
        "drs": FieldSpec(
            types=(int,),
            minimum=0,
            allowed=KNOWN_DRS_CODES,
            note="State code, not a boolean; {0,1,2,3,8,10,12,14} observed.",
        ),
    },
)

POSITION = EndpointSpec(
    name="position",
    natural_key=("session_key", "driver_number", "date"),
    natural_key_note="One record per position change.",
    timestamp_fields=("date",),
    order_field="date",
    order_group="driver_number",
    fields={
        "session_key": _SESSION_KEY,
        "meeting_key": _MEETING_KEY,
        "driver_number": _DRIVER_NUMBER,
        "date": _timestamp(),
        # Grids have historically reached 26 cars; 30 is a safe ceiling.
        "position": FieldSpec(types=(int,), required=True, nullable=False, minimum=1, maximum=30),
    },
)

INTERVALS = EndpointSpec(
    name="intervals",
    natural_key=("session_key", "driver_number", "date"),
    natural_key_note="One record per driver per timing sample.",
    timestamp_fields=("date",),
    order_field="date",
    order_group="driver_number",
    fields={
        "session_key": _SESSION_KEY,
        "meeting_key": _MEETING_KEY,
        "driver_number": _DRIVER_NUMBER,
        "date": _timestamp(),
        "gap_to_leader": FieldSpec(
            types=(int, float, str),
            minimum=0,
            string_pattern=LAPPED_GAP_PATTERN,
            note="Lapped cars report '+1 LAP' / '+2 LAPS' instead of a number.",
        ),
        "interval": FieldSpec(types=(int, float, str), minimum=0, string_pattern=LAPPED_GAP_PATTERN),
    },
)

STINTS = EndpointSpec(
    name="stints",
    natural_key=("session_key", "driver_number", "stint_number"),
    natural_key_note="One record per tyre stint.",
    fields={
        "session_key": _SESSION_KEY,
        "meeting_key": _MEETING_KEY,
        "driver_number": _DRIVER_NUMBER,
        "stint_number": _identifier(maximum=50),
        "lap_start": _identifier(maximum=100),
        "lap_end": _identifier(maximum=100),
        "compound": FieldSpec(types=(str,), allowed=KNOWN_COMPOUNDS),
        "tyre_age_at_start": FieldSpec(types=(int,), minimum=0, maximum=100),
    },
)

PIT = EndpointSpec(
    name="pit",
    natural_key=("session_key", "driver_number", "lap_number"),
    natural_key_note="One record per pit stop; verified unique on 2025 data.",
    timestamp_fields=("date",),
    fields={
        "session_key": _SESSION_KEY,
        "meeting_key": _MEETING_KEY,
        "driver_number": _DRIVER_NUMBER,
        "lap_number": _identifier(minimum=0, maximum=100),
        "date": _timestamp(),
        "pit_duration": FieldSpec(minimum=0, exclusive_minimum=True, plausible=(1.0, 300.0)),
        "lane_duration": FieldSpec(minimum=0, exclusive_minimum=True, plausible=(1.0, 300.0)),
        "stop_duration": FieldSpec(minimum=0, exclusive_minimum=True, plausible=(1.0, 300.0)),
    },
)

WEATHER = EndpointSpec(
    name="weather",
    natural_key=("session_key", "date"),
    natural_key_note="One reading per minute per session.",
    timestamp_fields=("date",),
    order_field="date",
    fields={
        "session_key": _SESSION_KEY,
        "meeting_key": _MEETING_KEY,
        "date": _timestamp(),
        "air_temperature": FieldSpec(minimum=-30, maximum=70, plausible=(-10.0, 50.0)),
        "track_temperature": FieldSpec(minimum=-30, maximum=100, plausible=(-10.0, 70.0)),
        "humidity": FieldSpec(minimum=0, maximum=100),
        "pressure": FieldSpec(minimum=800, maximum=1100),
        "rainfall": FieldSpec(types=(int, float), minimum=0, allowed=frozenset({0, 1})),
        "wind_direction": FieldSpec(types=(int,), minimum=0, maximum=360),
        "wind_speed": FieldSpec(minimum=0, maximum=150, plausible=(0.0, 50.0)),
    },
)

RACE_CONTROL = EndpointSpec(
    name="race_control",
    natural_key=("session_key", "date", "message"),
    natural_key_note="Verified unique; messages repeat but not at the same instant.",
    timestamp_fields=("date",),
    order_field="date",
    fields={
        "session_key": _SESSION_KEY,
        "meeting_key": _MEETING_KEY,
        "date": _timestamp(),
        "message": FieldSpec(types=(str,), required=True, nullable=False),
        "category": FieldSpec(types=(str,)),
        "lap_number": FieldSpec(types=(int,), minimum=0, maximum=100),
        "driver_number": FieldSpec(types=(int,), minimum=1, maximum=99, sparse=True),
        "flag": FieldSpec(types=(str,), sparse=True),
        "scope": FieldSpec(types=(str,), sparse=True),
        "sector": FieldSpec(types=(int,), minimum=0, maximum=40, sparse=True),
        "qualifying_phase": FieldSpec(types=(int,), sparse=True, note="Always null outside qualifying."),
    },
)


ENDPOINT_SPECS: Dict[str, EndpointSpec] = {
    spec.name: spec
    for spec in (
        MEETINGS,
        SESSIONS,
        DRIVERS,
        LAPS,
        CAR_DATA,
        POSITION,
        INTERVALS,
        STINTS,
        PIT,
        WEATHER,
        RACE_CONTROL,
    )
}


def get_spec(endpoint: str) -> Optional[EndpointSpec]:
    """The spec for an endpoint, or ``None`` if we have no expectations for it."""
    return ENDPOINT_SPECS.get(endpoint)
