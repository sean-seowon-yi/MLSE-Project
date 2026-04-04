# Phase 6: Inference, Evaluation & Similarity Search

Phase 6 generates a **global player embedding** `z_p` per player from all their possessions (with the same future-info masking as training), supports **similarity search** by cosine similarity, and provides **embedding-quality diagnostics**: pseudo ground-truth evaluation, self-consistency, policy distance analysis, plus **embedding-only** metrics (position retrieval, split-half for heuristics, qualitative neighbours).

---

## Goal

- **6A**: For each player with enough possessions, compute a single 64-D embedding `z_p`.
- **6B**: Given a query player_id, rank other players by cosine similarity to `z_p`.
- **6C**: Evaluate embeddings against externally sourced player-similarity pairs (ground truth).
- **6D**: Test embedding stability via self-consistency (competition-split + random-half).
- **6E**: Validate that cosine similarity corresponds to behavioral similarity (policy diagnostic).
- **6F**: Optional embedding-only checks (position-group retrieval, split-half for heuristic embeddings, qualitative neighbour tables).

---

## 6A: Embedding generation (`--mode inference`)

1. Load possession graphs and trained model (`best_model.pt`).
2. For each graph: apply same future-info masking as training → run GNN → collect `h_player` for each actor player (one embedding per possession per player after deduplication).
3. Per player: attention-pool all their per-possession `h_player` into one **z_p**.
4. **Minimum 50 possessions** per player; players below this are excluded.

### Inference split

The `--inference_split` flag controls which graphs are used:

| Value | Description |
|-------|-------------|
| `all` (default) | Full corpus of possession graphs |
| `train` | Training split only (same match-level split, seed=42) |
| `val` | Validation split only |
| `test` | Test split only |

This allows generating embeddings from specific subsets for controlled evaluation.

### Provenance system

Every embedding generation writes an `embedding_manifest.json` alongside the embedding files, recording:

- Checkpoint path and modification timestamp
- Graphs filename and `split_context_edges` flag
- Experiment tag and inference split

Downstream modes (`search`, `ground_truth`, `policy_diagnostic`) **validate the manifest** before running, catching:

- Stale embeddings (checkpoint retrained but embeddings not regenerated)
- Graph/checkpoint architecture mismatches
- Cross-tag contamination

### Outputs

| File | Content |
|------|---------|
| `player_embeddings.npy` | (n_players × 64) embedding matrix |
| `player_info.parquet` | Player metadata: player_id, name, position (most-frequent), team, n_possessions |
| `embedding_manifest.json` | Provenance metadata for downstream validation |

---

## 6B: Similarity search (`--mode search`)

1. Validate embedding provenance against the current checkpoint.
2. Compute pairwise cosine similarity between all `z_p` vectors.
3. Rank by similarity, with optional filters (position group, min possessions, exclude same team).
4. Return top-k matches (default k=10).

```bash
python main.py --mode search --player_id 5503
```

---

## 6C: Pseudo ground-truth evaluation (`--mode ground_truth`)

Evaluates the embedding space against **curated** player-similarity pairs (research-backed expectations). The current pair list, tiering, and **rationale for each pair** are documented in **[pseudo_ground_truth.md](pseudo_ground_truth.md)** (design only; that file intentionally omits metric results).

For each pair (A, B):

1. Compute cosine similarity between `z_A` and `z_B`.
2. Find B's rank in A's nearest-neighbour list (and vice versa), with **gender-aware** filtering on both directions.

Aggregate metrics: mean/median rank, hit@5/10/20/50, per-tier breakdowns, and bootstrap 95% confidence intervals on mean rank and hit@10.

### Outputs

| File | Content |
|------|---------|
| `ground_truth_report.txt` | Human-readable per-pair results with verdict |
| `ground_truth_results.json` | Structured results with run provenance appended |
| `gt_*.png` | Visual summaries (e.g. rank overview, hit@k, similarity vs rank) |

---

## 6D: Self-consistency evaluation (`--mode self_consistency`)

Tests whether the model assigns stable player identity by checking if the same player, observed in different data subsets, retrieves themselves as the closest match.

### Competition-split test

For players appearing in ≥2 competitions with sufficient possessions in each:

1. Pool `h_player` embeddings per competition independently.
2. Check self-retrieval rank (query one competition's embedding, retrieve from the other).
3. Non-testable players whose total possessions meet the threshold are added to the retrieval gallery as distractors, so ranks reflect the full inference-time search population.

### Random-half test

For all players with ≥100 possessions (regardless of competition count):

1. Randomly split possessions 50/50.
2. Pool each half independently.
3. Check self-retrieval rank.

Tests embedding stability without the confound of competition context shift.

### Metrics reported

Self-cosine (mean/median/std), cross-cosine, cosine margin, self-retrieval ranks (mean/median), hit@1/5/10/20/50, and per-position-group breakdowns.

### Outputs

| File | Content |
|------|---------|
| `self_consistency_report.txt` | Per-test summary with best/worst retrievals |
| `self_consistency_results.json` | Structured results with run provenance appended |
| `sc_*.png` | Plots (e.g. hit@k, rank CDF, position breakdown) |

---

## 6E: Policy diagnostic (`--mode policy_diagnostic`)

Three diagnostics testing whether cosine similarity corresponds to actual behavioral similarity:

### 1. Policy distance correlation

- Sample canonical game situations (stratified by actor position group, default 200).
- Compute predicted action distributions for all players via FiLM conditioning.
- Measure pairwise Jensen-Shannon divergence within each position group.
- Correlate with pairwise cosine distance using Spearman ρ.
- Statistical significance via Mantel-style row-permutation tests (not parametric), with Holm-Bonferroni correction across group-level tests.

Interpretation: ρ > 0.5 strong, ρ 0.3–0.5 moderate, ρ < 0.3 weak.

### 2. Substitute quality

- For a set of query players (~half top-possession, ~half random per position group):
  - Compare mean JS divergence of top-K cosine neighbours vs K random same-group players.
  - A ratio > 1 means cosine retrieval finds better behavioral matches than chance.
- Reports bootstrap 95% CI on the aggregate ratio.

Interpretation: ratio > 1.5 strong, 1.1–1.5 mild, ~1.0 no advantage.

### 3. FiLM sensitivity

- For each player, replace their `z_p` with a random same-group player's `z_p` (guaranteed derangement).
- Measure JS divergence against the correct predictions.
- High JS means FiLM is load-bearing and `z_p` genuinely modulates predictions.

Interpretation: JS > 0.1 strong effect, 0.01–0.1 moderate, < 0.01 weak.

### Outputs

| File | Content |
|------|---------|
| `policy_diagnostic_report.txt` | Three-section report with per-group breakdowns |
| `policy_diagnostic_results.json` | Structured results with run provenance appended |
| `pd_*.png` | Diagnostic figures (correlation by group, substitute quality, FiLM sensitivity, etc.) |

---

## 6F: Embedding-space diagnostics (`embedding_eval.py`)

These run automatically inside **`full_eval`** / **`eval_heuristics`** and can be run standalone (`--mode position_retrieval`, `split_half`, `qualitative_neighbors`):

| Mode | What it measures | Notes |
|------|------------------|-------|
| **position_retrieval** | Fraction of top‑K neighbours in the same coarse position group as the query | Works on any embedding matrix (GNN or heuristic). |
| **split_half** | Split each player’s events in half, rebuild embedding with an `embed_fn`, self-retrieval rank | **GNN:** skipped (`embed_fn=None`). **Heuristics** (`mean_features`, `action_profile`): uses event-based `embed_fn`. **FIFA heuristic:** skipped. |
| **qualitative_neighbors** | Top‑K neighbours for a fixed list of notable players; table + plot | Good for sanity checks and demos. |

Outputs per mode: JSON, text report, and PNG where applicable.

---

## Code

| Component | Location |
|-----------|----------|
| Embedding generator (batched, pooling) | `src/phase6_inference/embedding_generator.py` |
| Similarity search (cosine, filters) | `src/phase6_inference/similarity_search.py` |
| Ground-truth pair evaluation | `src/phase6_inference/ground_truth.py` |
| Self-consistency evaluator | `src/phase6_inference/self_consistency.py` |
| Policy diagnostic (JS correlation, substitute quality, FiLM sensitivity) | `src/phase6_inference/policy_diagnostic.py` |
| Position retrieval, split-half, qualitative neighbours | `src/phase6_inference/embedding_eval.py` |
| Embedding provenance (manifest build/validate) | `src/phase6_inference/provenance.py` |
| CLI entry | `main.py` → `--mode inference` / `search` / `ground_truth` / `self_consistency` / `policy_diagnostic` / `position_retrieval` / `split_half` / `qualitative_neighbors` |
| Name → `player_id` (accent + fuzzy) | `find_player.py` (project root under `GNN_StatsBomb/`) |

---

## How to run

```bash
cd GNN_StatsBomb

# Generate embeddings (full corpus)
python main.py --mode inference

# Generate embeddings (test split only)
python main.py --mode inference --inference_split test

# Search for similar players
python main.py --mode search --player_id 5503

# Run all evaluation diagnostics
python main.py --mode ground_truth
python main.py --mode self_consistency
python main.py --mode policy_diagnostic

# With experiment tag
python main.py --mode inference --tag split_ctx --split_context_edges
python main.py --mode ground_truth --tag split_ctx --split_context_edges

# Position-ablated + split-ctx + EMA alignment
python main.py --mode inference --tag pos_ablated_split_ctx_ema --split_context_edges --ablate_position
python main.py --mode ground_truth --tag pos_ablated_split_ctx_ema --split_context_edges --ablate_position
```

Requires Phase 1–3 outputs and a trained checkpoint (Phase 5). Embedding outputs go to `embeddings/baseline/` (or `embeddings/{tag}/`).

### Gender-aware evaluation

All similarity-based modes (`search`, `ground_truth`, `self_consistency`, `policy_diagnostic`) automatically filter candidates by gender. Gender is derived from `competitions.json` → `competition_gender` via the `build_gender_map()` utility in `similarity_search.py`. Male query players only see male candidates; female queries only see female candidates. No special flag is needed.

### FIFA stat comparison (`--mode fifa_comparison`)

Validates the GNN similarity system against external FIFA/EA Sports FC player attributes. Requires pre-matched FIFA CSVs in `FIFA_data/` (generated by `match_fifa_players.py`).

- Samples 10 male + 10 female players (stratified by position, random each run), plus famous players when available.
- For each, finds the top-1 same-gender substitute via GNN cosine similarity that also has FIFA data.
- Compares main and sub-attributes; multiple PNG summaries (radar grid with GK-aware axes, similarity vs stat distance, breakdowns, summary-style charts, sub-attribute heatmaps).

### Unified evaluation (`--mode full_eval`, `--mode full_eval_all`)

Run all post-training evaluations for one or all pipeline variants:

```bash
python main.py --mode full_eval                             # current tag
python main.py --mode full_eval_all --eval_output_dir ./evaluations  # all 11 registered pipelines
```

**Eleven steps** per pipeline: `inference`, `test_metrics`, `ground_truth`, `position_retrieval`, `split_half` (skipped for GNN), `qualitative_neighbors`, `self_consistency`, `policy_diagnostic`, `empirical_behavioral`, `analysis`, `fifa_comparison`. Outputs: `evaluations/{pipeline_name}/{step_name}/`. Heuristic orchestration: `generate_heuristics`, `eval_heuristics`, `full_eval_all_with_heuristics`.

---

## See also

- [../SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) — Phase 6 (provenance, ground truth, self-consistency, policy diagnostic).
- [PHASE5.md](PHASE5.md) — Training (same masking and pooling logic).
- [PHASE7.md](PHASE7.md) — Analysis uses these embeddings and the model for situation-level comparison.
