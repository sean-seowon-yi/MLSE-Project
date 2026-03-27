"""
Match StatsBomb 360 players to FIFA/EA FC player attribute datasets.

Reads all players from the StatsBomb open-data repository (cloned if not
present locally), matches them against the FIFA CSV files in ../FIFA_data/,
and writes:
  - FIFA_data/statsbomb_male_players_fifa.csv
  - FIFA_data/statsbomb_female_players_fifa.csv
  - FIFA_data/unmatched_players.txt

Usage:
    python match_fifa_players.py [--statsbomb_path PATH] [--fifa_path PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

STATSBOMB_REPO_URL = "https://github.com/statsbomb/open-data.git"

AVAILABLE_FIFA_VERSIONS = {15, 16, 17, 18, 19, 20, 21, 22, 23, 26}

COUNTRY_ALIASES: Dict[str, str] = {
    # All keys MUST be the output of normalize_name() — lowercase ASCII, no
    # punctuation, single-spaced.  Raw Unicode keys are dead code because
    # normalize_country() normalises before lookup.
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "korea south": "south korea",
    "korea dpr": "north korea",
    "korea north": "north korea",
    "ir iran": "iran",
    "islamic republic of iran": "iran",
    "iran islamic republic of": "iran",
    "china pr": "china",
    "chinese taipei": "taiwan",
    "turkiye": "turkey",
    "cote d ivoire": "ivory coast",
    "cote divoire": "ivory coast",
    "united states of america": "usa",
    "united states": "usa",
    "us": "usa",
    "bosnia and herzegovina": "bosnia herzegovina",
    "bosnia herzegovina": "bosnia herzegovina",
    "trinidad and tobago": "trinidad tobago",
    "trinidad tobago": "trinidad tobago",
    "antigua and barbuda": "antigua barbuda",
    "saint kitts and nevis": "saint kitts nevis",
    "sao tome and principe": "sao tome principe",
    "congo dr": "dr congo",
    "congo kinshasa": "dr congo",
    "democratic republic of congo": "dr congo",
    "republic of ireland": "ireland",
    "eswatini": "swaziland",
    "cabo verde": "cape verde",
    "czech republic": "czechia",
    "faroe islands": "faroe",
    "hong kong china": "hong kong",
    "macau": "macao",
    "north macedonia": "macedonia",
    "fyr macedonia": "macedonia",
    "macedonia republic of": "macedonia",
    "timor leste": "east timor",
    "brunei darussalam": "brunei",
    "lao peoples democratic republic": "laos",
    "viet nam": "vietnam",
    "curacao": "curacao",
    "england": "england",
    "scotland": "scotland",
    "wales": "wales",
    "northern ireland": "northern ireland",
    "new zealand": "new zealand",
}


# ────────────────────────────────────────────────────────────────────────────
# Name / country normalisation helpers
# ────────────────────────────────────────────────────────────────────────────

# Characters that survive NFKD without decomposing into ASCII base + combining mark.
_PRE_TRANSLITERATE = str.maketrans({
    "\u00D8": "O",  # Ø
    "\u00F8": "o",  # ø  (Ødegaard)
    "\u0110": "D",  # Đ
    "\u0111": "d",  # đ  (Đorđe)
    "\u00D0": "D",  # Ð  (Icelandic)
    "\u00F0": "d",  # ð  (Sigurðsson)
    "\u0141": "L",  # Ł
    "\u0142": "l",  # ł  (Łukasz)
    "\u00DF": "ss", # ß
    "\u00C6": "AE", # Æ
    "\u00E6": "ae", # æ
    "\u00DE": "Th", # Þ
    "\u00FE": "th", # þ
    "\u0126": "H",  # Ħ
    "\u0127": "h",  # ħ
    "\u014A": "N",  # Ŋ
    "\u014B": "n",  # ŋ
})


def normalize_name(name: str) -> str:
    """Lowercase, strip diacritics and non-Latin scripts, collapse whitespace."""
    if not name:
        return ""
    # Pre-transliterate characters that NFKD cannot decompose to ASCII
    name = name.translate(_PRE_TRANSLITERATE)
    nfkd = unicodedata.normalize("NFKD", name)
    # Keep only ASCII letters, digits, spaces (strip combining marks + non-Latin)
    ascii_chars = []
    for ch in nfkd:
        if unicodedata.combining(ch):
            continue
        if ch.isascii():
            ascii_chars.append(ch)
        # non-ASCII non-combining chars (e.g. Arabic, CJK) are dropped
    text = "".join(ascii_chars).lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_country(name: str) -> str:
    """Canonical lowercase country name."""
    if not name:
        return ""
    norm = normalize_name(name)
    return COUNTRY_ALIASES.get(norm, norm)


def name_tokens(norm_name: str) -> List[str]:
    """Split a normalised name into tokens, dropping single-char initials."""
    return [t for t in norm_name.split() if len(t) > 1]


def tokens_subset_match(tokens_a: List[str], tokens_b: List[str]) -> bool:
    """True if every token of the shorter list appears in the longer list."""
    if not tokens_a or not tokens_b:
        return False
    short, long = (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    long_set = set(long)
    return all(t in long_set for t in short)


def fuzzy_ratio(a: str, b: str) -> float:
    """SequenceMatcher ratio between two normalised names."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# ────────────────────────────────────────────────────────────────────────────
# Season → FIFA version mapping
# ────────────────────────────────────────────────────────────────────────────

def _season_to_year(season_name: str) -> int:
    """
    Extract the base calendar year from a season_name.

    '2023/2024' → 2023  (the *start* year — FIFA 24 covers the 23/24 season)
    '2022'      → 2022
    '2020'      → 2020

    Football convention: FIFA N covers the (N-1)/N season, so the start year
    of a slash-separated season determines the version.
    """
    nums = re.findall(r"\d{4}", season_name)
    if not nums:
        raise ValueError(f"Cannot parse year from season_name={season_name!r}")
    return min(int(n) for n in nums)


def best_fifa_version(season_name: str) -> int:
    """
    Pick the best *available* FIFA version for a competition season.

    FIFA editions are released each autumn (September/October):
      FIFA 21 → released Oct 2020, covers 2020/21 season
      FIFA 22 → released Oct 2021, covers 2021/22 season
      ...
      FC 26   → released 2025

    Heuristic: fifa_version = base_year - 2000 + 1, clamped to available.
    For a 2020/2021 season the base year is 2020 → version 21.
    For a 2022 tournament → version 23.
    For a 2023/2024 season → version 24 → nearest available = 23.
    """
    year = _season_to_year(season_name)
    ideal = year - 2000 + 1  # e.g. 2022 → 23, 2024 → 25
    if ideal in AVAILABLE_FIFA_VERSIONS:
        return ideal
    # Pick closest available version
    return min(AVAILABLE_FIFA_VERSIONS, key=lambda v: abs(v - ideal))


# ────────────────────────────────────────────────────────────────────────────
# Step 1 – Load StatsBomb player roster
# ────────────────────────────────────────────────────────────────────────────

def ensure_statsbomb_data(statsbomb_path: Optional[Path]) -> Path:
    """Return the path to open-data/data, cloning the repo if needed."""
    if statsbomb_path and (statsbomb_path / "competitions.json").exists():
        return statsbomb_path

    candidates = [
        Path(__file__).resolve().parent.parent / "StatsBomb" / "data",
        Path(__file__).resolve().parent.parent / "open-data" / "data",
        Path(__file__).resolve().parent.parent / "statsbomb-open-data" / "data",
    ]
    for p in candidates:
        if (p / "competitions.json").exists():
            print(f"[StatsBomb] Found local data at {p}")
            return p

    print("[StatsBomb] Data not found locally — cloning from GitHub (shallow)...")
    tmp = Path(tempfile.mkdtemp(prefix="statsbomb_"))
    subprocess.run(
        ["git", "clone", "--depth", "1", STATSBOMB_REPO_URL, str(tmp / "open-data")],
        check=True,
    )
    data_path = tmp / "open-data" / "data"
    if not (data_path / "competitions.json").exists():
        raise FileNotFoundError(f"competitions.json not found after clone at {data_path}")
    print(f"[StatsBomb] Cloned to {data_path}")
    return data_path


def load_statsbomb_players(sb_path: Path) -> pd.DataFrame:
    """
    Extract every player from every 360-available competition via lineups.

    Returns DataFrame with columns:
        sb_player_id, sb_player_name, sb_player_nickname, sb_jersey_number,
        sb_country, sb_position, sb_team_name,
        competition_id, season_id, competition_name, season_name,
        competition_gender, competition_international
    """
    with open(sb_path / "competitions.json", encoding="utf-8") as f:
        all_comps = json.load(f)

    comps_360 = [c for c in all_comps if c.get("match_available_360")]
    if not comps_360:
        raise RuntimeError("No competitions with 360 data found in competitions.json")

    print(f"\n[StatsBomb] {len(comps_360)} competitions with 360 data:")
    for c in comps_360:
        print(f"  {c['competition_name']} {c['season_name']} ({c['competition_gender']})")

    three60_dir = sb_path / "three-sixty"
    match_ids_with_360 = set()
    if three60_dir.exists():
        match_ids_with_360 = {int(f.stem) for f in three60_dir.glob("*.json") if f.stem.isdigit()}

    rows: List[Dict] = []
    for comp in tqdm(comps_360, desc="Loading competitions"):
        comp_id = comp["competition_id"]
        season_id = comp["season_id"]
        match_file = sb_path / "matches" / str(comp_id) / f"{season_id}.json"
        if not match_file.exists():
            continue
        with open(match_file, encoding="utf-8") as f:
            matches = json.load(f)

        match_ids = [m["match_id"] for m in matches]
        if match_ids_with_360:
            match_ids = [mid for mid in match_ids if mid in match_ids_with_360]

        for mid in tqdm(match_ids, desc=f"  {comp['competition_name']} {comp['season_name']}", leave=False):
            lineup_file = sb_path / "lineups" / f"{mid}.json"
            if not lineup_file.exists():
                continue
            with open(lineup_file, encoding="utf-8") as f:
                teams = json.load(f)

            for team in teams:
                team_name = team.get("team_name", "")
                for player in team.get("lineup", []):
                    positions = player.get("positions") or []
                    primary_pos = positions[0]["position"] if positions else "Unknown"
                    country_info = player.get("country") or {}
                    rows.append({
                        "sb_player_id": player["player_id"],
                        "sb_player_name": player.get("player_name", ""),
                        "sb_player_nickname": player.get("player_nickname") or "",
                        "sb_jersey_number": player.get("jersey_number"),
                        "sb_country": country_info.get("name", ""),
                        "sb_position": primary_pos,
                        "sb_team_name": team_name,
                        "competition_id": comp_id,
                        "season_id": season_id,
                        "competition_name": comp["competition_name"],
                        "season_name": comp["season_name"],
                        "competition_gender": comp["competition_gender"],
                        "competition_international": comp.get("competition_international", False),
                    })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["sb_player_id", "competition_id", "season_id"])
    print(f"\n[StatsBomb] {len(df)} unique (player, competition, season) entries")
    print(f"  Male:   {(df['competition_gender'] == 'male').sum()}")
    print(f"  Female: {(df['competition_gender'] == 'female').sum()}")
    return df


# ────────────────────────────────────────────────────────────────────────────
# Step 2 – Load FIFA data
# ────────────────────────────────────────────────────────────────────────────

def load_fifa_data(
    fifa_path: Path,
    gender: str,
    needed_versions: set[int],
) -> pd.DataFrame:
    """
    Load FIFA player data for the requested gender and versions.

    For male:  FIFA23/male_players (legacy).csv  +  FIFA26/FC26_20250921.csv
    For female: FIFA23/female_players (legacy).csv + FIFA23/female_players.csv
                (deduplicated to latest update per player per version)

    Returns DataFrame with all original columns plus:
        _norm_long, _norm_short, _norm_nationality
    """
    frames: List[pd.DataFrame] = []

    if gender == "male":
        legacy_male = fifa_path / "FIFA23" / "male_players (legacy).csv"
        if legacy_male.exists():
            print(f"[FIFA] Loading {legacy_male.name} ...")
            df = pd.read_csv(legacy_male, low_memory=False)
            v_mask = df["fifa_version"].isin(needed_versions & {15,16,17,18,19,20,21,22,23})
            frames.append(df[v_mask])

        fc26 = fifa_path / "FIFA26" / "FC26_20250921.csv"
        if fc26.exists() and 26 in needed_versions:
            print(f"[FIFA] Loading {fc26.name} ...")
            frames.append(pd.read_csv(fc26, low_memory=False))

    elif gender == "female":
        legacy_female = fifa_path / "FIFA23" / "female_players (legacy).csv"
        if legacy_female.exists():
            print(f"[FIFA] Loading {legacy_female.name} ...")
            df = pd.read_csv(legacy_female, low_memory=False)
            v_mask = df["fifa_version"].isin(needed_versions & {15,16,17,18,19,20,21,22,23})
            frames.append(df[v_mask])

        full_female = fifa_path / "FIFA23" / "female_players.csv"
        if full_female.exists():
            print(f"[FIFA] Loading {full_female.name} (full, for broader coverage) ...")
            df = pd.read_csv(full_female, low_memory=False)
            v_mask = df["fifa_version"].isin(needed_versions & {15,16,17,18,19,20,21,22,23})
            frames.append(df[v_mask])

        fc26 = fifa_path / "FIFA26" / "FC26_20250921.csv"
        if fc26.exists() and 26 in needed_versions:
            print(f"[FIFA] Loading {fc26.name} (checking for female players) ...")
            df = pd.read_csv(fc26, low_memory=False)
            if "league_name" in df.columns:
                women_mask = df["league_name"].str.contains(r"women|female|feminine|\bW$", case=False, na=False)
                if women_mask.any():
                    frames.append(df[women_mask])
                    print(f"  Found {women_mask.sum()} female rows in FC26")
                else:
                    print("  No female players found in FC26")

    if not frames:
        print(f"[FIFA] WARNING: No data loaded for gender={gender}")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate: keep latest update per player per fifa_version
    combined = combined.sort_values(
        ["fifa_version", "player_id", "fifa_update"], ascending=[True, True, False]
    )
    combined = combined.drop_duplicates(subset=["player_id", "fifa_version"], keep="first")

    # Pre-compute normalised fields for matching
    combined["_norm_long"] = combined["long_name"].fillna("").apply(normalize_name)
    combined["_norm_short"] = combined["short_name"].fillna("").apply(normalize_name)
    combined["_norm_nationality"] = combined["nationality_name"].fillna("").apply(normalize_country)

    print(f"[FIFA] {len(combined)} rows for gender={gender}, versions={sorted(needed_versions)}")
    return combined


# ────────────────────────────────────────────────────────────────────────────
# Step 3 – Multi-stage matching
# ────────────────────────────────────────────────────────────────────────────

def _pick_best_candidate(
    candidates: pd.DataFrame,
    sb_country_norm: str,
    sb_position: str,
) -> pd.Series:
    """Disambiguate among multiple FIFA candidates for a single SB player."""
    if len(candidates) == 1:
        return candidates.iloc[0]

    scored = candidates.copy()
    scored["_score"] = 0

    # Prefer matching nationality
    country_match = scored["_norm_nationality"] == sb_country_norm
    scored.loc[country_match, "_score"] += 100

    # Prefer matching position (rough — just check if any SB position token in FIFA positions)
    sb_pos_tokens = set(normalize_name(sb_position).split())
    if sb_pos_tokens:
        for idx, row in scored.iterrows():
            fifa_positions = normalize_name(str(row.get("player_positions", "")))
            fifa_pos_tokens = set(fifa_positions.replace(",", " ").split())
            if sb_pos_tokens & fifa_pos_tokens:
                scored.at[idx, "_score"] += 10

    # Prefer higher overall rating (more likely the well-known player)
    scored["_score"] += scored["overall"].fillna(0).astype(float) / 100.0

    best_idx = scored["_score"].idxmax()
    return scored.loc[best_idx]


def match_players(
    sb_players: pd.DataFrame,
    fifa_data: pd.DataFrame,
    gender: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Multi-stage matching of StatsBomb players to FIFA rows.

    Returns (matched_df, unmatched_df).
    """
    if sb_players.empty:
        return pd.DataFrame(), pd.DataFrame()
    if fifa_data.empty:
        unmatched = sb_players.copy()
        unmatched["unmatched_reason"] = "No FIFA data available for this gender"
        return pd.DataFrame(), unmatched

    # Pre-compute normalised SB fields
    sb = sb_players.copy()
    sb["_norm_name"] = sb["sb_player_name"].fillna("").apply(normalize_name)
    sb["_norm_nickname"] = sb["sb_player_nickname"].fillna("").apply(normalize_name)
    sb["_norm_country"] = sb["sb_country"].fillna("").apply(normalize_country)
    sb["_fifa_version"] = sb["season_name"].apply(best_fifa_version)

    # Build per-version FIFA lookup indices
    fifa_by_version: Dict[int, pd.DataFrame] = {}
    for v in sb["_fifa_version"].unique():
        subset = fifa_data[fifa_data["fifa_version"] == v]
        if subset.empty:
            closest = min(fifa_data["fifa_version"].unique(), key=lambda x: abs(x - v))
            print(f"  [Match] No FIFA data for version {v}, falling back to {closest}")
            subset = fifa_data[fifa_data["fifa_version"] == closest]
        fifa_by_version[v] = subset

    matched_rows: List[Dict] = []
    unmatched_rows: List[Dict] = []

    total = len(sb)
    stage_counts = Counter()

    for _, sb_row in tqdm(sb.iterrows(), total=total, desc=f"Matching {gender} players"):
        sb_norm = sb_row["_norm_name"]
        sb_nick = sb_row["_norm_nickname"]
        sb_country_norm = sb_row["_norm_country"]
        sb_position = sb_row["sb_position"]
        fv = sb_row["_fifa_version"]

        fifa_pool = fifa_by_version.get(fv)
        if fifa_pool is None or fifa_pool.empty:
            unmatched_rows.append({**sb_row.to_dict(), "unmatched_reason": f"No FIFA pool for version {fv}"})
            continue

        sb_tokens = name_tokens(sb_norm)
        fifa_match = None
        method = ""
        score = 0.0

        if not sb_norm:
            unmatched_rows.append({**sb_row.to_dict(),
                                   "target_fifa_version": int(fv),
                                   "unmatched_reason": "Empty player name"})
            continue

        # ── Stage 1: exact normalised long_name ──
        exact_long = fifa_pool[fifa_pool["_norm_long"] == sb_norm]
        if len(exact_long) == 1:
            fifa_match = exact_long.iloc[0]
            method, score = "exact_long_name", 1.0
        elif len(exact_long) > 1:
            fifa_match = _pick_best_candidate(exact_long, sb_country_norm, sb_position)
            method, score = "exact_long_name", 1.0

        # ── Stage 2: exact normalised nickname ↔ short_name ──
        if fifa_match is None and sb_nick:
            exact_nick = fifa_pool[fifa_pool["_norm_short"] == sb_nick]
            if len(exact_nick) == 1:
                fifa_match = exact_nick.iloc[0]
                method, score = "nickname_to_short", 1.0
            elif len(exact_nick) > 1:
                fifa_match = _pick_best_candidate(exact_nick, sb_country_norm, sb_position)
                method, score = "nickname_to_short", 1.0

        # ── Stage 2b: exact normalised SB name ↔ FIFA short_name ──
        if fifa_match is None:
            exact_short = fifa_pool[fifa_pool["_norm_short"] == sb_norm]
            if len(exact_short) == 1:
                fifa_match = exact_short.iloc[0]
                method, score = "exact_short_name", 1.0
            elif len(exact_short) > 1:
                fifa_match = _pick_best_candidate(exact_short, sb_country_norm, sb_position)
                method, score = "exact_short_name", 1.0

        # ── Stage 3: token-subset + nationality ──
        if fifa_match is None and sb_tokens:
            candidates = []
            for idx, frow in fifa_pool.iterrows():
                if frow["_norm_nationality"] != sb_country_norm:
                    continue
                f_tokens = name_tokens(frow["_norm_long"])
                if tokens_subset_match(sb_tokens, f_tokens):
                    candidates.append(frow)
            if len(candidates) == 1:
                fifa_match = candidates[0]
                method, score = "token_subset_nationality", 0.95
            elif len(candidates) > 1:
                cdf = pd.DataFrame(candidates)
                fifa_match = _pick_best_candidate(cdf, sb_country_norm, sb_position)
                method, score = "token_subset_nationality", 0.95

        # ── Stage 4: fuzzy match with nationality (threshold 0.85) ──
        if fifa_match is None:
            nat_pool = fifa_pool[fifa_pool["_norm_nationality"] == sb_country_norm]
            if not nat_pool.empty:
                best_ratio = 0.0
                best_idx_set: set = set()
                best_rows = []
                names_to_try = [sb_norm] + ([sb_nick] if sb_nick else [])
                for probe in names_to_try:
                    for idx, frow in nat_pool.iterrows():
                        r = max(
                            fuzzy_ratio(probe, frow["_norm_long"]),
                            fuzzy_ratio(probe, frow["_norm_short"]),
                        )
                        if r > best_ratio:
                            best_ratio = r
                            best_rows = [frow]
                            best_idx_set = {idx}
                        elif r == best_ratio and r > 0 and idx not in best_idx_set:
                            best_rows.append(frow)
                            best_idx_set.add(idx)

                if best_ratio >= 0.85 and best_rows:
                    if len(best_rows) == 1:
                        fifa_match = best_rows[0]
                    else:
                        cdf = pd.DataFrame(best_rows)
                        fifa_match = _pick_best_candidate(cdf, sb_country_norm, sb_position)
                    method, score = "fuzzy_with_nationality", round(best_ratio, 4)

        # ── Stage 5: fuzzy match without nationality (threshold 0.92) ──
        if fifa_match is None:
            best_ratio = 0.0
            best_idx_set = set()
            best_rows = []
            names_to_try = [sb_norm] + ([sb_nick] if sb_nick else [])
            for probe in names_to_try:
                for idx, frow in fifa_pool.iterrows():
                    r = max(
                        fuzzy_ratio(probe, frow["_norm_long"]),
                        fuzzy_ratio(probe, frow["_norm_short"]),
                    )
                    if r > best_ratio:
                        best_ratio = r
                        best_rows = [frow]
                        best_idx_set = {idx}
                    elif r == best_ratio and r > 0 and idx not in best_idx_set:
                        best_rows.append(frow)
                        best_idx_set.add(idx)

            if best_ratio >= 0.92 and best_rows:
                if len(best_rows) == 1:
                    fifa_match = best_rows[0]
                else:
                    cdf = pd.DataFrame(best_rows)
                    fifa_match = _pick_best_candidate(cdf, sb_country_norm, sb_position)
                method, score = "fuzzy_no_nationality", round(best_ratio, 4)

        # ── Stage 6: last-name + first-name similarity + nationality ──
        # Requires: same last name, same nationality, and the first names
        # must be plausibly the same person (prefix or fuzzy ≥ 0.5).
        if fifa_match is None and len(sb_tokens) >= 2:
            sb_first = sb_tokens[0]
            sb_last = sb_tokens[-1]
            candidates = []
            for idx, frow in fifa_pool.iterrows():
                if frow["_norm_nationality"] != sb_country_norm:
                    continue
                f_tokens = name_tokens(frow["_norm_long"])
                if not f_tokens or len(f_tokens) < 2:
                    continue
                if f_tokens[-1] != sb_last:
                    continue
                f_first = f_tokens[0]
                first_ok = (
                    sb_first == f_first
                    or (len(sb_first) >= 3 and f_first.startswith(sb_first))
                    or (len(f_first) >= 3 and sb_first.startswith(f_first))
                    or fuzzy_ratio(sb_first, f_first) >= 0.65
                )
                if first_ok:
                    candidates.append(frow)
            if len(candidates) == 1:
                fifa_match = candidates[0]
                method, score = "lastname_firstname_nationality", 0.80

        # ── Record result ──
        if fifa_match is not None:
            stage_counts[method] += 1
            out = sb_row.to_dict()
            out["target_fifa_version"] = int(fv)
            out["match_method"] = method
            out["match_score"] = score
            for col in fifa_pool.columns:
                if col.startswith("_norm"):
                    continue
                out[f"fifa_{col}"] = fifa_match[col] if col in fifa_match.index else None
            matched_rows.append(out)
        else:
            row_dict = sb_row.to_dict()
            row_dict["target_fifa_version"] = int(fv)
            row_dict["unmatched_reason"] = f"No match in FIFA {fv} {gender} data"
            unmatched_rows.append(row_dict)

    # Summary
    print(f"\n[Match] {gender.upper()} results: {len(matched_rows)} matched, {len(unmatched_rows)} unmatched")
    for m, c in stage_counts.most_common():
        print(f"  {m:30s}: {c}")

    matched_df = pd.DataFrame(matched_rows) if matched_rows else pd.DataFrame()
    unmatched_df = pd.DataFrame(unmatched_rows) if unmatched_rows else pd.DataFrame()

    return matched_df, unmatched_df


# ────────────────────────────────────────────────────────────────────────────
# Step 4 – Write outputs
# ────────────────────────────────────────────────────────────────────────────

_METHOD_RANK = {
    "exact_long_name": 0,
    "exact_short_name": 1,
    "nickname_to_short": 2,
    "token_subset_nationality": 3,
    "lastname_firstname_nationality": 4,
    "fuzzy_with_nationality": 5,
    "fuzzy_no_nationality": 6,
}


def _dedup_matches(
    matched: pd.DataFrame,
    unmatched: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """If two+ SB players in the same competition map to the same FIFA player,
    keep the best match (lower method rank, then higher score) and move the
    rest to *unmatched*."""
    if matched.empty:
        return matched, unmatched

    matched = matched.copy()
    matched["_method_rank"] = matched["match_method"].map(_METHOD_RANK).fillna(99)

    dup_key = ["fifa_player_id", "competition_id", "season_id"]
    dup_mask = matched.duplicated(subset=dup_key, keep=False)

    if not dup_mask.any():
        matched.drop(columns=["_method_rank"], inplace=True)
        return matched, unmatched

    clean_rows = matched[~dup_mask]
    dup_rows = matched[dup_mask]

    keep_idx = []
    demote_idx = []
    for _, grp in dup_rows.groupby(dup_key):
        best = grp.sort_values(
            ["_method_rank", "match_score"], ascending=[True, False]
        ).iloc[0]
        keep_idx.append(best.name)
        demote_idx.extend([i for i in grp.index if i != best.name])

    kept = matched.loc[keep_idx]
    demoted = matched.loc[demote_idx]

    demoted_records = []
    for _, r in demoted.iterrows():
        rec = {c: r[c] for c in unmatched.columns if c in r.index}
        rec["unmatched_reason"] = (
            f"Demoted: FIFA player {int(r['fifa_player_id'])} already matched "
            f"to a better SB candidate"
        )
        demoted_records.append(rec)

    new_matched = pd.concat([clean_rows, kept], ignore_index=True)
    new_matched.drop(columns=["_method_rank"], inplace=True)

    if demoted_records:
        extra_unmatched = pd.DataFrame(demoted_records)
        new_unmatched = pd.concat([unmatched, extra_unmatched], ignore_index=True)
        print(f"  [Dedup] Removed {len(demoted_records)} duplicate FIFA mappings")
    else:
        new_unmatched = unmatched

    return new_matched, new_unmatched


def _validate_fuzzy_matches(
    matched: pd.DataFrame,
    unmatched: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Post-match check: demote fuzzy matches where the first names match
    well but the last names clearly differ — a strong signal of a false
    positive (different person with common first name + same nationality)."""
    if matched.empty:
        return matched, unmatched

    fuzzy_methods = {"fuzzy_with_nationality", "fuzzy_no_nationality"}
    fuzzy_mask = matched["match_method"].isin(fuzzy_methods)
    if not fuzzy_mask.any():
        return matched, unmatched

    demote_idx = []
    for idx, row in matched[fuzzy_mask].iterrows():
        sb_toks = name_tokens(normalize_name(str(row["sb_player_name"])))
        fifa_toks = name_tokens(normalize_name(str(row.get("fifa_long_name", ""))))
        if len(sb_toks) < 2 or len(fifa_toks) < 2:
            continue
        sb_first, sb_last = sb_toks[0], sb_toks[-1]
        f_first, f_last = fifa_toks[0], fifa_toks[-1]
        first_sim = fuzzy_ratio(sb_first, f_first)
        last_sim = fuzzy_ratio(sb_last, f_last)
        if first_sim >= 0.8 and last_sim < 0.55:
            demote_idx.append(idx)

    if not demote_idx:
        return matched, unmatched

    demoted = matched.loc[demote_idx]
    demoted_records = []
    for _, r in demoted.iterrows():
        rec = {c: r[c] for c in unmatched.columns if c in r.index}
        rec["unmatched_reason"] = (
            f"Demoted: fuzzy match likely wrong person "
            f"(same first name, different last name)"
        )
        demoted_records.append(rec)

    new_matched = matched.drop(index=demote_idx).reset_index(drop=True)
    extra_unmatched = pd.DataFrame(demoted_records)
    new_unmatched = pd.concat([unmatched, extra_unmatched], ignore_index=True)
    print(f"  [Validate] Demoted {len(demote_idx)} fuzzy matches (same first, diff last name)")

    return new_matched, new_unmatched


def _clean_output(df: pd.DataFrame) -> pd.DataFrame:
    """Drop internal normalisation columns before writing."""
    drop_cols = [c for c in df.columns if c.startswith("_")]
    return df.drop(columns=drop_cols, errors="ignore")


def write_outputs(
    matched_male: pd.DataFrame,
    matched_female: pd.DataFrame,
    unmatched_male: pd.DataFrame,
    unmatched_female: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not matched_male.empty:
        out = output_dir / "statsbomb_male_players_fifa.csv"
        _clean_output(matched_male).to_csv(out, index=False)
        print(f"[Output] {out}  ({len(matched_male)} rows)")

    if not matched_female.empty:
        out = output_dir / "statsbomb_female_players_fifa.csv"
        _clean_output(matched_female).to_csv(out, index=False)
        print(f"[Output] {out}  ({len(matched_female)} rows)")

    # Unmatched report
    all_unmatched = pd.concat(
        [unmatched_male, unmatched_female], ignore_index=True
    )
    out_txt = output_dir / "unmatched_players.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Unmatched StatsBomb players — {len(all_unmatched)} total\n")
        f.write("=" * 80 + "\n\n")

        if all_unmatched.empty:
            f.write("All players matched!\n")
        else:
            for gender_label in ["male", "female"]:
                subset = all_unmatched[all_unmatched.get("competition_gender") == gender_label]
                if subset.empty:
                    continue
                f.write(f"--- {gender_label.upper()} ({len(subset)}) ---\n\n")
                for _, row in subset.iterrows():
                    name = row.get("sb_player_name", "?")
                    team = row.get("sb_team_name", "?")
                    comp = row.get("competition_name", "?")
                    season = row.get("season_name", "?")
                    country = row.get("sb_country", "?")
                    reason = row.get("unmatched_reason", "unknown")
                    f.write(f"  {name:35s} | {team:25s} | {country:20s} | "
                            f"{comp} {season} | {reason}\n")
                f.write("\n")

    print(f"[Output] {out_txt}  ({len(all_unmatched)} unmatched players)")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Match StatsBomb 360 players to FIFA attributes")
    parser.add_argument(
        "--statsbomb_path", type=str, default=None,
        help="Path to StatsBomb open-data/data/ directory (cloned automatically if absent)",
    )
    parser.add_argument(
        "--fifa_path", type=str, default=None,
        help="Path to FIFA_data/ directory (default: ../FIFA_data relative to this script)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    fifa_path = Path(args.fifa_path) if args.fifa_path else script_dir.parent / "FIFA_data"
    sb_override = Path(args.statsbomb_path) if args.statsbomb_path else None

    if not fifa_path.exists():
        print(f"ERROR: FIFA data directory not found at {fifa_path}")
        sys.exit(1)

    # Step 1: Load StatsBomb data
    sb_data_path = ensure_statsbomb_data(sb_override)
    sb_players = load_statsbomb_players(sb_data_path)

    if sb_players.empty:
        print("No StatsBomb players found. Exiting.")
        sys.exit(1)

    # Assign FIFA version to each row
    sb_players["_fifa_version"] = sb_players["season_name"].apply(best_fifa_version)

    male_sb = sb_players[sb_players["competition_gender"] == "male"].copy()
    female_sb = sb_players[sb_players["competition_gender"] == "female"].copy()

    # Step 2: Load FIFA data
    male_versions = set(male_sb["_fifa_version"].unique()) if not male_sb.empty else set()
    female_versions = set(female_sb["_fifa_version"].unique()) if not female_sb.empty else set()

    print(f"\n[FIFA] Male versions needed: {sorted(male_versions)}")
    print(f"[FIFA] Female versions needed: {sorted(female_versions)}")

    fifa_male = load_fifa_data(fifa_path, "male", male_versions) if male_versions else pd.DataFrame()
    fifa_female = load_fifa_data(fifa_path, "female", female_versions) if female_versions else pd.DataFrame()

    # Step 3+4: Match
    print("\n" + "=" * 60)
    print("MATCHING MALE PLAYERS")
    print("=" * 60)
    matched_male, unmatched_male = match_players(male_sb, fifa_male, "male")

    print("\n" + "=" * 60)
    print("MATCHING FEMALE PLAYERS")
    print("=" * 60)
    matched_female, unmatched_female = match_players(female_sb, fifa_female, "female")

    # Step 4b: Post-match validation and deduplication
    matched_male, unmatched_male = _validate_fuzzy_matches(matched_male, unmatched_male)
    matched_female, unmatched_female = _validate_fuzzy_matches(matched_female, unmatched_female)
    matched_male, unmatched_male = _dedup_matches(matched_male, unmatched_male)
    matched_female, unmatched_female = _dedup_matches(matched_female, unmatched_female)

    # Step 5: Write outputs
    print("\n" + "=" * 60)
    print("WRITING OUTPUTS")
    print("=" * 60)
    write_outputs(matched_male, matched_female, unmatched_male, unmatched_female, fifa_path)

    # Final summary
    total_sb = len(sb_players)
    total_matched = len(matched_male) + len(matched_female)
    total_unmatched = len(unmatched_male) + len(unmatched_female)
    print(f"\n{'=' * 60}")
    print(f"FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  StatsBomb players (unique per competition): {total_sb}")
    print(f"  Matched to FIFA:  {total_matched}  ({100*total_matched/max(total_sb,1):.1f}%)")
    print(f"  Unmatched:        {total_unmatched}  ({100*total_unmatched/max(total_sb,1):.1f}%)")


if __name__ == "__main__":
    main()
