# Phase 7: Situation-Level Analysis

Phase 7 validates **player similarity** in concrete situations: for a query player and their top similar players, it compares **predicted action distributions** in the same game states (same h_event, different z_p via FiLM).

---

## Goal

- For a query player: get top-k similar players from Phase 6.
- Sample real events from the query player’s possessions; for each, get the **situation encoding** h_event from the GNN.
- For each such situation, substitute each candidate player’s z_p, run FiLM + action heads, and compare predicted action type, direction, and length.
- Produce **reports** (text) and **plots** (bar charts, PCA of embeddings) to interpret similarity.

---

## Configuration

Analysis behaviour is controlled by `ReportConfig` in `report_builder.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_random_queries` | 5 | Number of query players to analyse when `query_player_ids` is not set. |
| `random_seed` | 42 | Seed for query and event sampling. |
| `query_player_ids` | None | If set, only these player IDs are used as queries (overrides random sampling). |
| `situations_per_query` | 5 | Max number of real game events sampled per query player for situation comparison. |
| `top_k_neighbours` | 5 | Number of nearest neighbours to include in reports and plots. |

---

## Outputs

All artifacts are written under `embeddings/{tag}/analysis/` (or `embeddings/baseline/analysis/` for the baseline run).

- **Text reports**: `report_<player_id>.txt` — per-situation predicted action type, direction, and length distributions for query and candidates.
- **Bar charts (per situation)**:
  - `situation_<player_id>_s<N>_actions.png` — grouped bars comparing action-type probabilities across players.
  - `situation_<player_id>_s<N>_direction.png` — angle-bin (direction) probabilities across players.
- **PCA plots**:
  - **Global** (all players, same 2D coordinates):  
    `embeddings_pca.png` (coloured by position **group**: GK / Defender / Midfielder / Forward),  
    `embeddings_pca_subgroup.png` (coloured by **subgroup**: e.g. Center Back, Full Back, Defensive Mid, Central Mid, Wide Mid, Forward, etc.),  
    `embeddings_pca_position.png` (coloured by full **position** name, all 26+).
  - **Per-query neighbourhood**: `pca_neighbourhood_<player_id>.png` — same 2D space with query and top-k neighbours highlighted and labelled.

Subgroups are defined in `src/config.py` via `POSITION_SUBGROUPS` (mapping from each position name to one of ~8 categories).

---

## Code

| Component | Location |
|-----------|----------|
| Report orchestration | `src/phase7_analysis/report_builder.py` |
| Counterfactual action prediction (FiLM + heads) | `src/phase7_analysis/situation_comparison.py` |
| Embedding PCA visualisation (group, subgroup, position, neighbourhood) | `src/phase7_analysis/embedding_viz.py` |
| CLI entry | `main.py` → `--mode analyze` |

---

## How to run

```bash
cd GNN_StatsBomb

# Baseline analysis
python main.py --mode analyze

# With experiment tag
python main.py --mode analyze --tag split_ctx --split_context_edges
```

Requires Phase 6 outputs (embeddings, player_info, embedding_manifest.json) and a trained checkpoint; reads possession graphs and event features for h_event and event sampling. The manifest is validated to ensure embeddings match the current checkpoint. Query players are chosen at random (or via config); see `ReportConfig` to fix specific player IDs.

**Gender-aware analysis.** When finding nearest neighbours for a query player, candidates are automatically filtered to match the query player's gender (derived from competition metadata). This ensures that female players are only compared to females and male players to males.

---

## See also

- [../SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) — Phase 7 summary.
- [PHASE6.md](PHASE6.md) — How z_p and similarity list are produced.
- [PHASE4.md](PHASE4.md) — FiLM and action heads used for counterfactual predictions.
