"""
Phase 1-A: Data Preparation for StatsBomb Events.

Responsibilities
────────────────
1.  Load competitions / matches / events from the StatsBomb open-data JSON tree.
2.  Filter to tactically meaningful on-ball event types.
3.  Attach match-level context (competition, season, teams) to every event.

Coordinate convention
─────────────────────
StatsBomb already stores coordinates so that the **acting team always attacks
left → right** (opponent goal at x = 120).  There is therefore no per-period
flip required.  We normalise the raw (0–120, 0–80) values to (0–1, 0–1) and
keep the left-to-right attacking direction.

Left and right roles (e.g. Right Wing vs Left Wing) are kept distinct in the
feature encoder by default so that preferred foot and tactical side matter for
player similarity. This module stores the canonical StatsBomb coordinates;
optional mirroring is applied only when ``feature.mirror_sides`` is True.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import DataConfig, get_config, PITCH_LENGTH, PITCH_WIDTH


# ── Dataclasses ──────────────────────────────────────────────────────────

@dataclass
class MatchInfo:
    """Lightweight match metadata."""

    match_id: int
    competition_id: int
    competition_name: str
    season_id: int
    season_name: str
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str
    home_score: int
    away_score: int


@dataclass
class ParsedEvent:
    """
    A single on-ball event with all fields the downstream encoder needs.

    Coordinates are in *raw* StatsBomb units (x ∈ [0, 120], y ∈ [0, 80])
    with the acting team attacking left → right.
    """

    # Identifiers
    event_id: str
    match_id: int
    competition_id: int
    season_id: int

    # Temporal
    period: int
    minute: int
    second: int
    timestamp: str

    # Actor
    player_id: int
    player_name: str
    team_id: int
    team_name: str
    position_name: str

    # Spatial
    location_x: float
    location_y: float
    end_location_x: Optional[float]
    end_location_y: Optional[float]

    # Event semantics
    event_type: str
    play_pattern: str
    duration: float

    # Context flags
    under_pressure: bool
    counterpress: bool

    # Pass-specific
    pass_outcome: Optional[str]
    pass_type: Optional[str]
    pass_height: Optional[str]
    pass_body_part: Optional[str]
    pass_length: Optional[float]
    pass_angle: Optional[float]
    pass_switch: bool
    pass_cross: bool
    pass_through_ball: bool

    # Shot-specific
    shot_outcome: Optional[str]
    shot_type: Optional[str]
    shot_body_part: Optional[str]
    shot_technique: Optional[str]
    shot_first_time: bool
    shot_xg: Optional[float]

    # Dribble-specific
    dribble_outcome: Optional[str]
    dribble_nutmeg: bool

    # Carry-specific
    carry_end_x: Optional[float]
    carry_end_y: Optional[float]

    # Possession context
    possession_number: int
    possession_team_id: int
    possession_team_name: str

    # StatsBomb 360 spatial context (only set when use_360=True and frame exists)
    # List of {"teammate": bool, "actor": bool, "keeper": bool, "location": [x, y]}
    freeze_frame: Optional[List[Dict]] = None


# ── Loader ───────────────────────────────────────────────────────────────

class StatsBombDataLoader:
    """
    Loads and filters StatsBomb open data.

    Usage::

        loader = StatsBombDataLoader(config.data)
        matches = loader.load_matches()          # list[MatchInfo]
        events  = loader.load_all_events(matches) # pd.DataFrame
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.base_path = Path(config.statsbomb_base_path)

    # ── public API ───────────────────────────────────────────────────

    def load_competitions(self) -> List[Dict]:
        """Return the full competitions.json list, optionally filtered."""
        path = self.base_path / "competitions.json"
        with open(path) as f:
            comps = json.load(f)

        if self.config.competition_ids:
            comps = [c for c in comps if c["competition_id"] in self.config.competition_ids]
        if self.config.season_ids:
            comps = [c for c in comps if c["season_id"] in self.config.season_ids]

        return comps

    def load_matches(self, competitions: Optional[List[Dict]] = None) -> List[MatchInfo]:
        """
        Load match metadata for the selected competitions/seasons.

        Returns a flat list of MatchInfo objects.
        """
        if competitions is None:
            competitions = self.load_competitions()

        matches: List[MatchInfo] = []

        for comp in tqdm(competitions, desc="Loading matches"):
            comp_id = comp["competition_id"]
            season_id = comp["season_id"]
            match_file = self.base_path / "matches" / str(comp_id) / f"{season_id}.json"

            if not match_file.exists():
                continue

            with open(match_file) as f:
                match_list = json.load(f)

            for m in match_list:
                matches.append(MatchInfo(
                    match_id=m["match_id"],
                    competition_id=comp_id,
                    competition_name=comp.get("competition_name", ""),
                    season_id=season_id,
                    season_name=comp.get("season_name", ""),
                    home_team_id=m["home_team"]["home_team_id"],
                    home_team_name=m["home_team"]["home_team_name"],
                    away_team_id=m["away_team"]["away_team_id"],
                    away_team_name=m["away_team"]["away_team_name"],
                    home_score=m.get("home_score", 0),
                    away_score=m.get("away_score", 0),
                ))

        if self.config.use_360:
            match_ids_with_360 = self._get_match_ids_with_360()
            matches = [m for m in matches if m.match_id in match_ids_with_360]

        return matches

    def _get_match_ids_with_360(self) -> set:
        """Return set of match_id for which three-sixty/{match_id}.json exists."""
        three60_dir = self.base_path / self.config.three_sixty_dir
        if not three60_dir.exists():
            return set()
        return {int(f.stem) for f in three60_dir.glob("*.json") if f.stem.isdigit()}

    def load_match_events(self, match: MatchInfo) -> List[ParsedEvent]:
        """Parse every on-ball event for a single match. When use_360=True, only events that have a 360 frame are kept."""
        event_file = self.base_path / "events" / f"{match.match_id}.json"
        if not event_file.exists():
            return []

        with open(event_file) as f:
            raw_events = json.load(f)

        event_uuid_to_freeze: Dict[str, List[Dict]] = {}
        if self.config.use_360:
            three60_file = self.base_path / self.config.three_sixty_dir / f"{match.match_id}.json"
            if three60_file.exists():
                with open(three60_file) as f:
                    frames_360 = json.load(f)
                for fr in frames_360:
                    event_uuid_to_freeze[fr["event_uuid"]] = fr.get("freeze_frame") or []

        allowed = set(self.config.event_types)
        parsed: List[ParsedEvent] = []

        for raw in raw_events:
            if raw.get("period", 1) == 5:
                continue  # penalty shootout — not open play

            if not raw.get("player", {}).get("id"):
                continue  # no identified player — cannot attribute to an actor

            etype = raw["type"]["name"]
            if etype not in allowed:
                continue

            loc = raw.get("location")
            if loc is None or not isinstance(loc, (list, tuple)) or len(loc) < 2:
                continue

            if self.config.use_360:
                event_id = raw["id"]
                if event_id not in event_uuid_to_freeze:
                    continue
                freeze_frame = event_uuid_to_freeze[event_id]
            else:
                freeze_frame = None

            parsed.append(self._parse_event(raw, match, freeze_frame=freeze_frame))

        return parsed

    def load_all_events(
        self,
        matches: Optional[List[MatchInfo]] = None,
    ) -> pd.DataFrame:
        """
        Load events for every match and return a single DataFrame.

        The DataFrame has one row per event; the column names match the
        `ParsedEvent` field names.
        """
        if matches is None:
            matches = self.load_matches()

        all_events: List[Dict] = []
        skipped = 0

        for match in tqdm(matches, desc="Loading events"):
            try:
                events = self.load_match_events(match)
                for ev in events:
                    all_events.append(vars(ev))
            except Exception as exc:
                skipped += 1
                if skipped <= 3:
                    print(f"  Skipped match {match.match_id}: {exc}")

        if skipped > 3:
            print(f"  … and {skipped - 3} more skipped matches")

        df = pd.DataFrame(all_events)
        return df

    # ── internals ────────────────────────────────────────────────────

    def _parse_event(
        self, raw: Dict, match: MatchInfo, freeze_frame: Optional[List[Dict]] = None
    ) -> ParsedEvent:
        """Convert one raw StatsBomb event dict into a ParsedEvent. freeze_frame is from 360 when use_360=True."""

        loc = raw["location"]
        etype = raw["type"]["name"]

        # Clamp coordinates to valid pitch bounds (handle StatsBomb edge cases)
        location_x = max(0.0, min(float(loc[0]), PITCH_LENGTH))
        location_y = max(0.0, min(float(loc[1]), PITCH_WIDTH))

        # End location (varies by event type)
        end_x, end_y = self._extract_end_location(raw, etype)
        if end_x is not None:
            end_x = max(0.0, min(float(end_x), PITCH_LENGTH))
        if end_y is not None:
            end_y = max(0.0, min(float(end_y), PITCH_WIDTH))

        # Pass fields
        p = raw.get("pass", {})
        pass_outcome = p.get("outcome", {}).get("name") if p.get("outcome") else "Complete"
        pass_type = p.get("type", {}).get("name") if p.get("type") else None
        pass_height = p.get("height", {}).get("name") if p.get("height") else None
        pass_body_part = p.get("body_part", {}).get("name") if p.get("body_part") else None

        # Shot fields
        s = raw.get("shot", {})
        shot_outcome = s.get("outcome", {}).get("name") if s.get("outcome") else None
        shot_type = s.get("type", {}).get("name") if s.get("type") else None
        shot_body_part = s.get("body_part", {}).get("name") if s.get("body_part") else None
        shot_technique = s.get("technique", {}).get("name") if s.get("technique") else None

        # Dribble fields
        d = raw.get("dribble", {})
        dribble_outcome = d.get("outcome", {}).get("name") if d.get("outcome") else None

        # Carry fields
        c = raw.get("carry", {})
        carry_end = c.get("end_location")

        return ParsedEvent(
            event_id=raw["id"],
            match_id=match.match_id,
            competition_id=match.competition_id,
            season_id=match.season_id,
            period=raw["period"],
            minute=raw["minute"],
            second=raw["second"],
            timestamp=raw["timestamp"],
            player_id=raw.get("player", {}).get("id", -1),
            player_name=raw.get("player", {}).get("name", "Unknown"),
            team_id=raw["team"]["id"],
            team_name=raw["team"]["name"],
            position_name=raw.get("position", {}).get("name", "Unknown"),
            location_x=location_x,
            location_y=location_y,
            end_location_x=end_x,
            end_location_y=end_y,
            event_type=etype,
            play_pattern=raw["play_pattern"]["name"],
            duration=max(0.0, float(raw.get("duration") or 0.0)),
            under_pressure=bool(raw.get("under_pressure", False)),
            counterpress=bool(raw.get("counterpress", False)),
            pass_outcome=pass_outcome if etype == "Pass" else None,
            pass_type=pass_type if etype == "Pass" else None,
            pass_height=pass_height if etype == "Pass" else None,
            pass_body_part=pass_body_part if etype == "Pass" else None,
            pass_length=(
                max(0.0, min(float(p.get("length") or 0.0), 100.0)) if etype == "Pass" else None
            ),
            pass_angle=p.get("angle") if etype == "Pass" else None,
            pass_switch=bool(p.get("switch", False)) if etype == "Pass" else False,
            pass_cross=bool(p.get("cross", False)) if etype == "Pass" else False,
            pass_through_ball=bool(p.get("through_ball", False)) if etype == "Pass" else False,
            shot_outcome=shot_outcome if etype == "Shot" else None,
            shot_type=shot_type if etype == "Shot" else None,
            shot_body_part=shot_body_part if etype == "Shot" else None,
            shot_technique=shot_technique if etype == "Shot" else None,
            shot_first_time=bool(s.get("first_time", False)) if etype == "Shot" else False,
            shot_xg=(
                max(0.0, min(float(s.get("statsbomb_xg") or 0.0), 1.0)) if etype == "Shot" else None
            ),
            dribble_outcome=dribble_outcome if etype == "Dribble" else None,
            dribble_nutmeg=bool(d.get("nutmeg", False)) if etype == "Dribble" else False,
            carry_end_x=(
                max(0.0, min(float(carry_end[0]), PITCH_LENGTH)) if carry_end and len(carry_end) >= 2 else None
            ),
            carry_end_y=(
                max(0.0, min(float(carry_end[1]), PITCH_WIDTH)) if carry_end and len(carry_end) >= 2 else None
            ),
            possession_number=raw.get("possession", 0),
            possession_team_id=raw.get("possession_team", {}).get("id", -1),
            possession_team_name=raw.get("possession_team", {}).get("name", "Unknown"),
            freeze_frame=freeze_frame,
        )

    @staticmethod
    def _extract_end_location(raw: Dict, etype: str) -> Tuple[Optional[float], Optional[float]]:
        """Pull end_location from the event-type-specific sub-dict."""
        if etype == "Pass":
            end = raw.get("pass", {}).get("end_location")
        elif etype == "Shot":
            end = raw.get("shot", {}).get("end_location")
        elif etype == "Carry":
            end = raw.get("carry", {}).get("end_location")
        else:
            end = None

        if end and len(end) >= 2:
            return end[0], end[1]
        return None, None


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    config = get_config()
    loader = StatsBombDataLoader(config.data)

    comps = loader.load_competitions()
    print(f"Competitions/seasons available: {len(comps)}")

    matches = loader.load_matches(comps[:3])
    print(f"Matches loaded: {len(matches)}")

    if matches:
        events = loader.load_match_events(matches[0])
        print(f"\nEvents in match {matches[0].match_id}: {len(events)}")
        if events:
            ev = events[0]
            print(f"  First event: {ev.event_type} by {ev.player_name} "
                  f"at ({ev.location_x:.1f}, {ev.location_y:.1f})")
