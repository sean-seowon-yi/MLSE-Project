# Phase 7: Situation-Level Analysis

Phase 7 validates **player similarity** in concrete situations: for a query player and their top similar players, it compares **predicted action distributions** in the same game states (same h_event, different z_p via FiLM).

---

## Goal

- For a query player: get top-k similar players from Phase 6.
- Sample real events from the query player’s possessions; for each, get the **situation encoding** h_event from the GNN.
- For each such situation, substitute each candidate player’s z_p, run FiLM + action heads, and compare predicted action type, direction, and length.
- Produce **reports** (text) and **plots** (bar charts, PCA of embeddings) to interpret similarity.

---

## Outputs

- **Text reports**: Per-situation predicted action type, direction, and length distributions for query and candidates.
- **Bar charts**: Grouped bars comparing action probabilities across players per situation.
- **PCA plots**: 2D embedding space with query and neighbours highlighted.

Artifacts are written under `embeddings/analysis/` (e.g. `report_<player_id>.txt`, situation-specific PNGs).

---

## Code

| Component | Location |
|-----------|----------|
| Report orchestration | `src/phase7_analysis/report_builder.py` |
| Counterfactual action prediction (FiLM + heads) | `src/phase7_analysis/situation_comparison.py` |
| Embedding PCA visualisation | `src/phase7_analysis/embedding_viz.py` |
| CLI entry | `main.py` → `--mode analyze` (often with player_id or config) |

---

## How to run

Typically:

```bash
cd GNN_StatsBomb
python main.py --mode analyze
```

(Exact CLI may take a player_id or config for which player to report on; see `main.py` and `report_builder.py`.)

Requires Phase 6 outputs (embeddings, player_info) and a trained checkpoint; reads possession graphs for h_event and event sampling.

---

## See also

- [../SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) — Phase 7 summary.
- [PHASE6.md](PHASE6.md) — How z_p and similarity list are produced.
- [PHASE4.md](PHASE4.md) — FiLM and action heads used for counterfactual predictions.
