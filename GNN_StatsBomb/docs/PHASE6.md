# Phase 6: Inference, Evaluation & Similarity Search

Phase 6 generates a **global player embedding** `z_p` per player from all their possessions (with the same future-info masking as training), supports **similarity search** by cosine similarity, and provides three **embedding-quality diagnostics**: pseudo ground-truth evaluation, self-consistency testing, and policy distance analysis.

---

## Goal

- **6A**: For each player with enough possessions, compute a single 64-D embedding `z_p`.
- **6B**: Given a query player_id, rank other players by cosine similarity to `z_p`.
- **6C**: Evaluate embeddings against externally sourced player-similarity pairs (ground truth).
- **6D**: Test embedding stability via self-consistency (competition-split + random-half).
- **6E**: Validate that cosine similarity corresponds to behavioral similarity (policy diagnostic).

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

Evaluates the embedding space against externally sourced player-similarity pairs drawn from public StatsBomb analysis articles. Ten pairs are defined across three tiers:

| Tier | Description | Examples |
|------|-------------|---------|
| 1 | Strong directional expectation | Miedema ↔ Caldentey, TAA ↔ Hakimi, Alba ↔ Robertson |
| 2 | Good directional expectation | Kroos ↔ Enzo Fernandez, TAA ↔ Maehle, Sancho ↔ Vargas |
| 3 | Weak / conditional expectation | Kane ↔ Leao, Kane ↔ Felix, Uduokhai ↔ Souttar |

For each pair (A, B):

1. Compute cosine similarity between `z_A` and `z_B`.
2. Find B's rank in A's nearest-neighbour list (and vice versa).

Aggregate metrics: mean/median rank, hit@5/10/20/50, per-tier breakdowns, and bootstrap 95% confidence intervals on mean rank and hit@10.

### Outputs

| File | Content |
|------|---------|
| `ground_truth_report.txt` | Human-readable per-pair results with verdict |
| `ground_truth_results.json` | Structured results with run provenance appended |

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

---

## Code

| Component | Location |
|-----------|----------|
| Embedding generator (batched, pooling) | `src/phase6_inference/embedding_generator.py` |
| Similarity search (cosine, filters) | `src/phase6_inference/similarity_search.py` |
| Ground-truth pair evaluation | `src/phase6_inference/ground_truth.py` |
| Self-consistency evaluator | `src/phase6_inference/self_consistency.py` |
| Policy diagnostic (JS correlation, substitute quality, FiLM sensitivity) | `src/phase6_inference/policy_diagnostic.py` |
| Embedding provenance (manifest build/validate) | `src/phase6_inference/provenance.py` |
| CLI entry | `main.py` → `--mode inference` / `search` / `ground_truth` / `self_consistency` / `policy_diagnostic` |

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
```

Requires Phase 1–3 outputs and a trained checkpoint (Phase 5). Embedding outputs go to `embeddings/baseline/` (or `embeddings/{tag}/`).

### Gender-aware evaluation

All similarity-based modes (`search`, `ground_truth`, `self_consistency`, `policy_diagnostic`) automatically filter candidates by gender. Gender is derived from `competitions.json` → `competition_gender` via the `build_gender_map()` utility in `similarity_search.py`. Male query players only see male candidates; female queries only see female candidates. No special flag is needed.

### FIFA stat comparison (`--mode fifa_comparison`)

Validates the GNN similarity system against external FIFA/EA Sports FC player attributes. Requires pre-matched FIFA CSVs in `FIFA_data/` (generated by `match_fifa_players.py`).

- Samples 10 male + 10 female players (stratified by position, random each run).
- For each, finds the top-1 same-gender substitute via GNN cosine similarity that also has FIFA data.
- Compares 6 main stats and 34 sub-attributes; computes Spearman correlation.
- Generates 4 visualizations (radar charts, scatter, stat breakdown, summary dashboard).

### Unified evaluation (`--mode full_eval`, `--mode full_eval_all`)

Run all post-training evaluations for one or all pipeline variants:

```bash
python main.py --mode full_eval                             # current tag
python main.py --mode full_eval_all --eval_output_dir ./evaluations  # all 4 variants
```

Outputs are organized under `evaluations/{pipeline_name}/{step_name}/`.

---

## See also

- [../SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) — Phase 6 (provenance, ground truth, self-consistency, policy diagnostic).
- [PHASE5.md](PHASE5.md) — Training (same masking and pooling logic).
- [PHASE7.md](PHASE7.md) — Analysis uses these embeddings and the model for situation-level comparison.
