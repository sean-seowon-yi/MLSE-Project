# Phase 3: Heterogeneous Graph Construction

Phase 3 converts each possession into a **heterogeneous graph** (`HeteroData`) that encodes the full situation: the sequence of actions, who performed them, and where every player was (from 360 freeze frames).

---

## Goal

- One **event** node per on-ball action in the possession (features: 122-D with Spatial_360 block **zeroed**).
- One **player** node per distinct actor or off-ball (360) player; features include position (or Unknown for off-ball), team flag, and spatial offset (dx, dy) from the ball.
- **Edges**: temporal (`next`/`prev` with time-delta attribute), actor-event (`acts_in`/`performed_by`), and context (`context_for` from 360).
- Output a list of `HeteroData` graphs for training and inference.

---

## Inputs

| Source | Description |
|--------|-------------|
| **possessions.pkl** | From Phase 2: list of `Possession` (event indices, timestamps_sec, labels). |
| **event_features.npy** | From Phase 1: (N × 122) event feature matrix. |
| **freeze_frames.pkl** | From Phase 1: list of freeze-frame lists, aligned by event index (for 360 context_for edges). |

---

## Output

| File | Content |
|------|--------|
| **possession_graphs.pkl** | List of `HeteroData` objects, one per possession. |

---

## Node types

| Type | Features | Description |
|------|----------|-------------|
| **event** | 122-D (Spatial_360 block zeroed) | One node per on-ball event. The GNN must use player nodes for spatial context, not the 360 summary in the event vector. |
| **player** | [position_idx, is_possession_team, dx, dy] | One per distinct player. **Actors**: real position, player_id; dx=0, dy=0. **Off-ball** (360): Unknown position, spatial offset from ball. |

---

## Edge types

| Relation | Edge attribute | Description |
|----------|----------------|-------------|
| (event, **next**, event) | 1-D time delta | Event t → event t+1; attribute = normalised time gap. |
| (event, **prev**, event) | 1-D time delta | Event t+1 → event t (reverse temporal). |
| (player, **acts_in**, event) | — | This player performed this event. |
| (event, **performed_by**, player) | — | Reverse of acts_in. |
| (player, **context_for**, event) | — | Off-ball player visible in 360 at this event. |

**Time-delta**: `min(Δt / 30, 1.0)` per edge, so the GNN gets tempo (e.g. quick counter vs slow buildup).

---

## Masking (Phase 3)

- **Spatial_360 zeroed** in event node features so the model learns spatial context from explicit player nodes and `context_for` edges, not from the 9-D summary in the event vector.
- Future-info and position masking are applied **at runtime** in Phase 5 (training) and Phase 6 (inference), not in the stored graphs.

---

## Code

| Component | Location |
|-----------|----------|
| Graph builder | `src/phase3_graph/graph_builder.py` |
| Masking (event-feature level) | `src/phase3_graph/masking.py` |
| CLI entry | `main.py` → `--mode graph` |

---

## How to run

```bash
cd GNN_StatsBomb
python main.py --mode graph
```

Requires Phase 1 and Phase 2 outputs in `processed_data/`.

---

## See also

- [../SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) — Phase 3 (rationale: heterogeneous graph, node/edge types, time-delta).
- [PHASE2.md](PHASE2.md) — Possession construction.
- [PHASE4.md](PHASE4.md) — Model that consumes these graphs.
