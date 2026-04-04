# Phase 1: Data Preparation & Feature Encoding

Phase 1 loads StatsBomb events (and, by default, StatsBomb 360 freeze frames), filters to on-ball events with spatial context, encodes each event as a fixed-length numeric vector, and writes the feature matrix and metadata to disk. No embedding or similarity model is trained yet—that is planned for a later phase.

---

## Overview

| Step | What happens |
|------|----------------|
| 1. Load | Load competitions → matches → events; when `use_360=True`, restrict to matches with 360 data and to events that have a 360 frame; attach `freeze_frame` to each event. |
| 2. Summarise | Print event-type distribution and unique players/teams/matches. |
| 3. Encode | Convert each event row into a **126-dimensional** vector (event features + optional 360 spatial features + period), with coordinate normalisation; left/right roles are kept distinct by default (configurable via `feature.mirror_sides`). |
| 4. Save | Write `event_features.npy`, `event_metadata.parquet`, `feature_names.json`, and `data_stats.json` to `processed_data/`. |

---

## 1. Load

**Code:** `main.py` → `StatsBombDataLoader` (`src/data_preparation.py`).

- **Competitions:** Read `competitions.json`; optionally filter by `competition_ids` and `season_ids` (CLI: `--competition`, `--season`).
- **Matches:** For each competition/season, load `matches/{competition_id}/{season_id}.json`.  
  If **`data.use_360`** is `True` (default), keep only matches whose `match_id` has a file `three-sixty/{match_id}.json`.
- **Events (per match):**
  - Read `events/{match_id}.json`.
  - If `use_360=True`, read `three-sixty/{match_id}.json` and build a map `event_uuid` → `freeze_frame`.
  - Keep only events whose **event type** is in the configured on-ball list (Pass, Carry, Shot, Pressure, Dribble, etc.) and whose **location** is present and valid (list of length ≥ 2).
  - If `use_360=True`, keep only events whose **event `id`** appears as `event_uuid` in the 360 file; attach the corresponding **freeze frame** (list of other players’ positions) to the event.
- **Parsing:** Each kept event is parsed into a `ParsedEvent`: coordinates clamped to pitch bounds, pass/shot/carry fields extracted, duration and outcomes normalised. See `docs/DATA_QUALITY.md` for edge-case handling (missing location, OOB, negative duration, etc.).
- **Output:** A single DataFrame with one row per event; column names match `ParsedEvent` fields, including `freeze_frame` when 360 is used.

---

## 2. Summarise

Counts are printed to the console: event-type distribution, and unique `player_id`, `team_id`, `match_id`. No files are written in this step.

---

## 3. Encode

**Code:** `EventFeatureEncoder` (`src/feature_encoder.py`).

Each event row is turned into a **126-D** vector. By default the encoder does **not** mirror (left/right roles stay distinct). **Mirroring** is applied only when `feature.mirror_sides=True`: for events whose actor has a “Right …” position (e.g. Right Back, Right Wing), the y-axis is flipped (y′ = 80 − y) and the position label is remapped to the corresponding “Left …” role, so flank roles are comparable for player similarity.

### Feature groups (126 dimensions total)

| # | Group | Dims | Description |
|---|--------|-----|-------------|
| 1 | event_type | 14 | One-hot (Pass, Carry, Shot, Pressure, etc.) |
| 2 | location | 2 | (x, y) normalised to [0, 1] by pitch size |
| 3 | end_location | 3 | (end_x, end_y, has_end); 0 if no end location |
| 4 | delta | 2 | (dx, dy) = end − start (0 if no end) |
| 5 | dist_angle | 2 | Euclidean distance and signed angle of delta |
| 6 | play_pattern | 9 | One-hot (Regular Play, From Corner, etc.) |
| 7 | position | 26 | One-hot (sided: Left/Right distinct unless mirror_sides=True) |
| 8 | body_part | 7 | One-hot (Left/Right Foot, Head, etc.) |
| 9 | pass_outcome | 6 | One-hot (zeros for non-pass) |
| 10 | shot_outcome | 8 | One-hot (zeros for non-shot) |
| 11 | dribble_outcome | 2 | One-hot (zeros for non-dribble) |
| 12 | pass_type | 8 | One-hot |
| 13 | pass_height | 3 | One-hot |
| 14 | shot_type | 3 | One-hot |
| 15 | scalars | 9 | duration (norm), under_pressure, counterpress, pass_length, pass_angle, pass_switch, pass_cross, shot_xg, shot_first_time |
| 16 | pitch_zone | 9 | 3×3 grid one-hot (x = thirds, y = lanes) |
| 17 | spatial_360 | 9 | Teammate/opponent/keeper counts (norm), mean teammate/opponent positions (norm), min dist to teammate, min dist to opponent (norm); zeros when no 360 |
| 18 | period | 4 | One-hot match period (Period 1, Period 2, Extra Time 1, Extra Time 2); not masked at training |

All coordinates and relevant scalars are normalised/clipped so the matrix has no NaN/Inf; unknown categorical values yield an all-zero one-hot slice.

---

## 4. Save

**Directory:** `processed_data/` (config: `data.output_dir`).

| File | Content |
|------|--------|
| **event_features.npy** | NumPy array of shape `(n_events, 126)`, dtype float32. Row order matches metadata. |
| **event_metadata.parquet** | One row per event: `event_id`, `match_id`, `competition_id`, `season_id`, `player_id`, `player_name`, `team_id`, `team_name`, `position_name`, `event_type`, `period`, `minute`, `second`. No `freeze_frame` column. |
| **freeze_frames.pkl** | List of freeze-frame lists (when `use_360=True`), aligned by event index; used in Phase 3. |
| **feature_names.json** | List of 126 feature names (for slicing/debugging). |
| **data_stats.json** | Summary: `n_events`, `n_players`, `n_teams`, `n_matches`, `n_competitions`, `feature_dim`, `event_type_counts`, `mirror_sides`, `use_360`. |

**Representative scale (default open data + 360, no CLI filters):** on a typical checkout this produces about **737k** events, **323** matches, **7** competition–season pairs, and **~52k** possessions after Phase 2 filters — see `SYSTEM_DESIGN.md` § Dataset → Scale for exact figures from one measured run. Your `data_stats.json` is authoritative for your tree.

If any NaN/Inf are found in the feature matrix, they are replaced by 0 before saving and a warning is printed.

---

## How to run

```bash
cd GNN_StatsBomb
python main.py --mode prepare
```

Optional filters:

```bash
python main.py --mode prepare --competition 11 --season 1
```

Configuration (e.g. in `src/config.py`):

- **`data.use_360`** (default `True`): Use 360; only matches with `three-sixty/{match_id}.json` and only events with a 360 frame. Set to `False` to use all matches/events and zero out the 9 spatial_360 features.
- **`data.statsbomb_base_path`**: Path to the StatsBomb `data` directory.
- **`data.three_sixty_dir`**: Subdirectory name for 360 files (default `"three-sixty"`).
- **`data.output_dir`**: Where to write outputs (default `"./processed_data"`).
- **`feature.mirror_sides`** (default `False`): If `True`, right-side positions are mirrored to the left; when `False`, Left Wing / Right Wing etc. stay distinct.

---

## Relation to other phases

Phase 1 produces **event-level** features and metadata. These are consumed by:

- **Phase 2**: Groups events into possession sequences using `event_metadata.parquet`.
- **Phase 3**: Builds heterogeneous graphs using `event_features.npy`, `event_metadata.parquet`, and `freeze_frames.pkl`.
- **Phase 5/6**: Runtime masking zeroes out action/identity fields from the 126-D vector; only situational features survive.

---

## See also

- [../SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) — Phase 1 section (feature table, masking rationale).
- [PHASE2.md](PHASE2.md) — Possession construction.
- [PHASE3.md](PHASE3.md) — Graph construction.
- [DATA_QUALITY.md](DATA_QUALITY.md) — Edge cases, clamping, missing data, and integrity/domain notes.
