# Phase 3: Heterogeneous Graph Construction

Phase 3 converts each possession into a **heterogeneous graph** (`HeteroData`) that encodes the full situation: the sequence of actions, who performed them, and where every player was (from 360 freeze frames).

---

## Goal

- One **event** node per on-ball action in the possession (features: 126-D with Spatial_360 block **zeroed**).
- **Player** nodes: one node per **actor** `player_id` in the possession, plus **one node per off-ball freeze-frame slot per event** (teammate/opponent indices at that event); features include position (or Unknown for context nodes), team flag, and spatial offset (dx, dy) from the ball.
- **Edges**: temporal (`next`/`prev` with time-delta attribute), actor-event (`acts_in`/`performed_by`), and 360 context (`context_for` by default; optionally split into `context_for_tm` / `context_for_opp` via `--split_context_edges`).
- Output a list of `HeteroData` graphs for training and inference.

---

## Inputs

| Source | Description |
|--------|-------------|
| **possessions.pkl** | From Phase 2: list of `Possession` (event indices, timestamps_sec, labels). |
| **event_features.npy** | From Phase 1: (N × 126) event feature matrix. |
| **freeze_frames.pkl** | From Phase 1: list of freeze-frame lists, aligned by event index (for 360 context edges). |

---

## Output

| File | Content |
|------|--------|
| **possession_graphs.pkl** (or `possession_graphs_{tag}.pkl` with `--tag`) | List of `HeteroData` objects, one per possession. |

---

## Node types

| Type | Features | Description |
|------|----------|-------------|
| **event** | 126-D (Spatial_360 block zeroed) | One node per on-ball event. The GNN must use player nodes for spatial context, not the 360 summary in the event vector. |
| **player** | [position_idx, is_possession_team, dx, dy] | **Actors:** one node per distinct `player_id` (`("actor", pid)`). **Off-ball 360:** one node per slot per event (`("tm"/"opp", local_event_idx, k)` in `graph_builder.py`); Unknown position; `(dx, dy)` from ball at that event. |

---

## Edge types

| Relation | Edge attribute | Description |
|----------|----------------|-------------|
| (event, **next**, event) | 1-D time delta | Event t → event t+1; attribute = normalised time gap. |
| (event, **prev**, event) | 1-D time delta | Event t+1 → event t (reverse temporal). |
| (player, **acts_in**, event) | — | This player performed this event. |
| (event, **performed_by**, player) | — | Reverse of acts_in. |
| (player, **context_for**, event) | — | Off-ball player visible in 360 at this event. **Default** when `split_context_edges=False`. |
| (player, **context_for_tm**, event) | — | **Teammate** off-ball player (passing options, support). Used when `--split_context_edges` is set. |
| (player, **context_for_opp**, event) | — | **Opponent** off-ball player (defensive pressure, blocks). Used when `--split_context_edges` is set. |

> **Ablation:** Pass `--split_context_edges` to split the single `context_for` edge into `context_for_tm` and `context_for_opp`. The default (`False`) preserves backward compatibility with existing checkpoints.

**Time-delta**: `min(Δt / 30, 1.0)` per edge, so the GNN gets tempo (e.g. quick counter vs slow buildup).

---

## Masking (Phase 3)

- **Spatial_360 zeroed** in event node features so the model learns spatial context from explicit player nodes and context edges, not from the 9-D summary in the event vector.
- Future-info and position masking are applied **at runtime** in Phase 5 (training) and Phase 6 (inference), not in the stored graphs.

---

## Code

| Component | Location |
|-----------|----------|
| Graph builder | `src/phase3_graph/graph_builder.py` |
| Masking (event-feature level) | `src/phase3_graph/masking.py` |
| CLI entry | `main.py` → `--mode build_graphs` |

---

## Graph/config validation

When loading graphs, `validate_graph_config_match()` (in `src/config.py`) checks that the loaded graph file matches the active `--split_context_edges` setting. This prevents silently loading graphs with a single `context_for` edge when the model expects split edges (or vice versa), which would cause incorrect behaviour at training or inference time.

When `--mode build_graphs` is run and a graph file already exists, the builder checks for architecture mismatch and refuses to overwrite graphs with a different edge schema. Use a different `--tag` to create a separate graph file.

---

## How to run

```bash
cd GNN_StatsBomb

# Build graphs (baseline: unified context_for)
python main.py --mode build_graphs

# Build graphs (split teammate/opponent context edges)
python main.py --mode build_graphs --tag split_ctx --split_context_edges
```

Requires Phase 1 and Phase 2 outputs in `processed_data/`. With `--tag`, the output is `possession_graphs_{tag}.pkl`.

---

## See also

- [../SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) — Phase 3 (rationale: heterogeneous graph, node/edge types, time-delta).
- [PHASE2.md](PHASE2.md) — Possession construction.
- [PHASE4.md](PHASE4.md) — Model that consumes these graphs.
