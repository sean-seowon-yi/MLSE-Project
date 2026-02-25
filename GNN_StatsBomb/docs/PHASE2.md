# Phase 2: Possession Construction

Phase 2 groups events into **possession sequences** — continuous spells where one team controls the ball — and attaches per-possession labels and per-event timestamps used downstream for graph construction and time-delta edge attributes.

---

## Goal

- Group events by StatsBomb `possession_number` and `possession_team_id` within each match.
- Sort events within each possession by time.
- Compute possession-level labels: `ends_in_shot`, `ends_in_goal`, `total_xg`.
- Store per-event timestamps (seconds from match start) for Phase 3 temporal edge attributes.
- Filter out possessions that are too short or too long.

---

## Inputs

| Source | Description |
|--------|-------------|
| **event_metadata.parquet** | From Phase 1: one row per event with `match_id`, `player_id`, `event_type`, `period`, `minute`, `second`, etc. |
| **Event indices** | Global event indices aligned with `event_features.npy` and `event_metadata.parquet` row order. |

Possession membership comes from StatsBomb event fields: `possession_number` and `possession_team_id`.

---

## Output

| File | Content |
|------|--------|
| **possessions.pkl** | List of `Possession` objects. Each has: global event indices, per-event metadata, per-event timestamps (`timestamps_sec`), and labels. |

### Possession dataclass (conceptually)

- **event_indices**: global indices into the event feature matrix / metadata.
- **timestamps_sec**: seconds from match start per event (`minute × 60 + second`), used in Phase 3 for time-delta on temporal edges.
- **ends_in_shot** (bool): whether the possession contains a Shot event (in practice this approximates “ends in shot”).
- **ends_in_goal** (bool): whether the possession contains a goal.
- **total_xg** (float): sum of xG in the possession.

### Filtering

- **Min 2 events**: Single-event possessions have no temporal structure (no sequence to learn from).
- **Max 200 events**: Very long possessions are outliers; the cap keeps training batches manageable.

---

## Code

| Component | Location |
|-----------|----------|
| Possession definition & builder | `src/phase2_possession/possession_builder.py` |
| CLI entry | `main.py` → `--mode possession` |

---

## How to run

```bash
cd GNN_StatsBomb
python main.py --mode possession
```

Requires Phase 1 outputs in `processed_data/` (e.g. `event_metadata.parquet`).

---

## Relation to other phases

- **Phase 1**: Uses event metadata (and alignment with `event_features.npy`) to group by possession.
- **Phase 3**: Uses `Possession` list and `timestamps_sec` to build graphs and to set time-delta attributes on `(event, next, event)` and `(event, prev, event)` edges.

---

## See also

- [../SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) — Phase 2 section (rationale: why possession as unit, min/max event counts, labels).
- [PHASE1.md](PHASE1.md) — Data preparation outputs.
- [PHASE3.md](PHASE3.md) — Graph construction.
