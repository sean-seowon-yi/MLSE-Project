# Data Quality & Edge Cases

This document describes how the GNN_StatsBomb pipeline handles missing data, extreme values, and edge cases.

## Data source: StatsBomb 360

The pipeline uses **StatsBomb 360** (not only the regular event stream). Only matches that have a `three-sixty/{match_id}.json` file are loaded, and only events that have a 360 frame are kept. Each such event is joined to its freeze frame (positions of other players at that moment). Spatial-context features (teammate/opponent counts, mean positions, min distances) are computed from the freeze frame and appended to the feature vector. When `use_360=False`, matches and events are unrestricted but spatial-context features are zeroed.

## Audit Summary (StatsBomb Open Data)

Based on a 50-match sample:

| Issue | Count | Handling |
|-------|--------|----------|
| Missing `location` | ~1.3k | Event skipped (not loaded). |
| `location` not a list or `len(loc) < 2` | 0 | Event skipped to avoid IndexError. |
| Coordinates outside [0,120]×[0,80] | 5 | Clamped to pitch bounds in **data_preparation** before storing. |
| Missing `player` / `position` | 67 | Only in event types we exclude (e.g. Referee Ball-Drop). We default to `player_id=-1`, `position_name="Unknown"`; "Unknown" is in the position vocabulary. |
| Negative `duration` | Rare | Clamped to `max(0, duration)` in both **data_preparation** and **feature_encoder**. |
| Pass `length` &lt; 0 or &gt; 100 m | 12 | Clamped to [0, 100] in **data_preparation**; normalised and clipped to [0, 1] in **feature_encoder**. |
| Shot `statsbomb_xg` outside [0,1] | 0 | Clamped to [0, 1] in **data_preparation** and **feature_encoder**. |
| Shot `end_location` with 3 elements (x,y,z) | Common | We use only `end_location[0]` and `end_location[1]`. |
| Position "Secondary Striker" | Present | Added to **config** `POSITIONS` and `POSITION_GROUPS["Forward"]`. |
| Position "Unknown" / missing | Possible | Added "Unknown" to `POSITIONS`; encoder uses it when position is missing or NaN. |

## Where It’s Handled

### `src/data_preparation.py`

- **360**: When `use_360=True`, filter matches to those with `three-sixty/{match_id}.json`; for each match load 360 frames and keep only events whose `id` appears as `event_uuid` in the 360 file; attach the corresponding `freeze_frame` to each parsed event.
- **Load**: Skip events with `location is None`, or `location` not a list/tuple, or `len(location) < 2`.
- **Parse**: Clamp `location_x`/`location_y` to [0, 120] and [0, 80]; clamp `end_location_x`/`end_location_y` and carry end to the same bounds; clamp `duration` to ≥ 0; clamp `pass_length` to [0, 100]; clamp `shot_xg` to [0, 1]; clamp carry end to pitch and require `len(carry_end) >= 2` before using.

### `src/feature_encoder.py`

- **Coordinates**: After normalising, clip `nx`, `ny`, `nex`, `ney` to [0, 1].
- **Scalars**: `duration / _MAX_DURATION` and normalised `pass_length` clipped to [0, 1]; `shot_xg` clipped to [0, 1].
- **Position**: If `position_name` is missing or NaN, use `"Unknown"`; `event_type` and `play_pattern` use `.get(..., default)` so missing keys don’t crash.
- **One-hot**: Unknown categorical values (not in config vocabularies) produce an all-zero one-hot vector; no crash.
- **360 spatial**: From each `freeze_frame` we compute 9 features (teammate/opponent/keeper counts, mean positions, min distances to nearest teammate/opponent). Freeze-frame locations are clamped to pitch; when `feature.mirror_sides` is True, y is flipped for right-side actors. Min distances are normalised by `feature.max_dist_for_norm` (default 50) and clipped to [0, 1]. If `freeze_frame` is missing or empty, these 9 values are zero.

### `src/config.py`

- **POSITIONS**: Includes `"Secondary Striker"` and `"Unknown"`.
- **POSITION_GROUPS**: `"Secondary Striker"` added under `"Forward"`.

## Integrity & domain (player similarity)

The pipeline targets **player similarity**: event features and 360 spatial context are defined so that (1) left and right roles (e.g. Left Wing vs Right Wing) are kept distinct by default to reflect preferred foot and tactical side, (2) spatial features reflect pressure/support/density from freeze frames, and (3) event types, positions, and pitch zones match football semantics. Downstream, events are grouped by `player_id` (in `event_metadata.parquet`) to build player-level representations for similarity.

## Result

- No NaN or Inf in the feature matrix.
- Normalised coordinates and relevant scalars stay in [0, 1] (or [−1, 1] for signed angles).
- Pipeline runs on the full StatsBomb open dataset without failing on the audited edge cases.

## See also

- **README.md** — Setup, usage, configuration.
- **docs/PHASE1.md** — Full Phase 1 pipeline (load → encode → save).
