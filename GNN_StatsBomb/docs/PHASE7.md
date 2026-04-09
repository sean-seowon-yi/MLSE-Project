# Phase 7: Situation-Level Analysis

Phase 7 validates **player similarity** in concrete situations: for a query player and their top similar players, it compares **predicted action distributions** in the same game states (same `h_event`, different `z_p` via FiLM).

---

## Counterfactual-safe situation encoding (`h_event`)

**Problem (historical):** If `(player, acts_in, event)` message passing is active, the actor’s identity (via player-node features) can flow **into** `h_event` before FiLM. Swapping another player’s `z_p` then mixes “what the situation looks like if *they* were the actor in the graph” with “what they would do if only FiLM changed” — a **counterfactual leak**.

**What the code does:** `SituationComparator` calls `PlayerSimilarityModel.encode_possession_counterfactual` (see `src/phase4_model/model.py`). That forward **removes** `acts_in` edges only; temporal edges, `performed_by`, and 360 `context_for*` edges stay so the graph remains usable. Policy diagnostic and possession animation use the same pattern (diagnostic directly; animation via `SituationComparator`).

**Scope:** This does **not** change **training**, which still uses the full graph (including `acts_in`) through `forward` / `encode_possession` — appropriate for imitation learning.

**Residual nuance (doc sync):** Keeping `(event, performed_by, player)` means the actor node is still in the computation graph; only the direct **player → event** actor channel is removed. Further hardening (e.g. stricter actor-free encoders) is optional and not implemented. See `docs/FUTURE_IMPROVEMENTS.md` §8.

---

## Goal

- For a query player: get top-k similar players from Phase 6.
- Sample real events from the query player’s possessions; for each, get **`h_event`** from the GNN via **`encode_possession_counterfactual`** (not the training-time full graph).
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

Artifacts are written under **`embeddings/{tag}/analysis/`** when you run `--mode analyze` alone, or under **`evaluations/{tag}/analysis/`** when Phase 7 is invoked from **`full_eval`**.

- **Text reports**: `report_<player_id>.txt` — per-situation predicted action type, direction, and length distributions for query and candidates.
- **Bar charts (per situation)**:
  - `situation_<player_id>_s<N>_actions.png` — grouped bars comparing action-type probabilities across players.
  - `situation_<player_id>_s<N>_direction.png` — angle-bin (direction) probabilities across players.
- **PCA plots** (2D projection with PCA):
  - **Global**: `embeddings_pca.png` (position **group**), `embeddings_pca_subgroup.png` (**subgroup**), `embeddings_pca_position.png` (full **position** name).
  - **Per-query neighbourhood**: `pca_neighbourhood_<player_id>.png` — query and top-k neighbours highlighted and fully labelled.
- **t-SNE plots** (same layout as PCA; skipped if too few players for a stable perplexity):
  - **Global**: `embeddings_tsne.png`, `embeddings_tsne_subgroup.png`, `embeddings_tsne_position.png`.
  - **Per-query neighbourhood**: `tsne_neighbourhood_<player_id>.png`.

Subgroups are defined in `src/config.py` via `POSITION_SUBGROUPS` (mapping from each position name to one of ~8 categories).

---

## Possession animation (`--mode possession_animation`)

Separate from `analyze`, this mode renders **MP4/GIF** videos of one or more possession graphs: pitch, ball trail (event locations), freeze-frame players, and predicted actions for **two players** (counterfactual) or **dynamic substitute** mode (on-ball actor vs top-1 cosine neighbour). Supports ground-truth vs prediction labelling for the actor, multi-segment possession chains, and smooth export settings. Requires `--pipeline`, trained checkpoint, embeddings, and graphs. See `main.py` docstring and `src/phase7_analysis/possession_animation.py`.

## Code

| Component | Location |
|-----------|----------|
| Report orchestration | `src/phase7_analysis/report_builder.py` |
| Counterfactual action prediction (FiLM + heads) | `src/phase7_analysis/situation_comparison.py` |
| Embedding PCA + t-SNE visualisation (group, subgroup, position, neighbourhood) | `src/phase7_analysis/embedding_viz.py` |
| Possession animation (MP4/GIF) | `src/phase7_analysis/possession_animation.py` |
| CLI entry | `main.py` → `--mode analyze` or `--mode possession_animation` |

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
