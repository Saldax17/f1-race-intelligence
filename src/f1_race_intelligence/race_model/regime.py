"""Green / SC / VSC / red per lap, from race-control messages.

Encoding confirmed on 2023–25 (scope note, OpenF1 audit): ``category ==
"SafetyCar"`` with ``SAFETY CAR DEPLOYED`` / ``SAFETY CAR IN THIS LAP`` /
``VIRTUAL SAFETY CAR DEPLOYED`` / ``VIRTUAL SAFETY CAR ENDING``, and
``flag == "RED"``. A red flag closes any open SC or VSC: 2023 São Paulo had a
standing restart and no "SC in" message, which otherwise leaves the safety
car out for the rest of the race.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

GREEN, SC, VSC, RED = "GREEN", "SC", "VSC", "RED"


@dataclass(frozen=True)
class RegimeEvent:
    """One neutralisation: its kind, the lap it began and the lap racing resumed.

    ``end_lap`` is ``None`` while it is still running (a live race) or if
    the race finished under it.
    """

    kind: str
    start_lap: int
    end_lap: Optional[int]

    @property
    def laps(self) -> Optional[int]:
        return None if self.end_lap is None else max(1, self.end_lap - self.start_lap)


def regime_events(messages: Iterable[Mapping[str, Any]]) -> List[RegimeEvent]:
    """Neutralisations in order, from race-control messages sorted by date."""
    events: List[RegimeEvent] = []
    open_sc: Optional[int] = None
    open_vsc: Optional[int] = None
    for message in sorted(messages, key=lambda m: m.get("date") or ""):
        text = str(message.get("message") or "").upper()
        lap = int(message.get("lap_number") or 0)
        if message.get("flag") == "RED" or text.startswith("RED FLAG"):
            if open_sc is not None:
                events.append(RegimeEvent(SC, open_sc, lap))
            if open_vsc is not None:
                events.append(RegimeEvent(VSC, open_vsc, lap))
            open_sc = open_vsc = None
            if not events or events[-1].kind != RED or events[-1].start_lap != lap:
                events.append(RegimeEvent(RED, lap, lap + 1))
        elif "VIRTUAL SAFETY CAR DEPLOYED" in text:
            if open_vsc is None:
                open_vsc = lap
        elif "VIRTUAL SAFETY CAR ENDING" in text:
            if open_vsc is not None:
                events.append(RegimeEvent(VSC, open_vsc, lap))
                open_vsc = None
        elif text.startswith("SAFETY CAR DEPLOYED"):
            if open_sc is None:
                open_sc = lap
        elif text.startswith("SAFETY CAR IN THIS LAP"):
            if open_sc is not None:
                events.append(RegimeEvent(SC, open_sc, lap))
                open_sc = None
    if open_sc is not None:
        events.append(RegimeEvent(SC, open_sc, None))
    if open_vsc is not None:
        events.append(RegimeEvent(VSC, open_vsc, None))
    return sorted(events, key=lambda e: (e.start_lap, e.kind))


def neutralised_laps(messages: Iterable[Mapping[str, Any]], last_lap: int) -> Set[int]:
    """Laps whose times say nothing about pace: under SC/VSC/red, plus the lap after.

    The lap after a safety car ends is the restart lap, still slow; around a
    red flag the laps either side are also dropped.
    """
    laps: Set[int] = set()
    for event in regime_events(messages):
        if event.kind == RED:
            laps |= {event.start_lap - 1, event.start_lap, event.start_lap + 1, event.start_lap + 2}
        else:
            end = event.end_lap if event.end_lap is not None else last_lap
            laps |= set(range(event.start_lap, end + 2))
    return {lap for lap in laps if lap >= 1}


def regime_by_lap(messages: Iterable[Mapping[str, Any]], last_lap: int) -> Dict[int, str]:
    """The regime of every lap 1..last_lap (red beats SC beats VSC)."""
    out = {lap: GREEN for lap in range(1, last_lap + 1)}
    rank = {GREEN: 0, VSC: 1, SC: 2, RED: 3}
    for event in regime_events(messages):
        end = event.end_lap if event.end_lap is not None else last_lap
        for lap in range(event.start_lap, min(end, last_lap) + 1):
            if lap in out and rank[event.kind] > rank[out[lap]]:
                out[lap] = event.kind
    return out


def current_regime(messages: Iterable[Mapping[str, Any]]) -> str:
    """The regime in force now: an SC/VSC that has not ended yet, else green."""
    for event in reversed(regime_events(messages)):
        if event.end_lap is None:
            return event.kind
    return GREEN
