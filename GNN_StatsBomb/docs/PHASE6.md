# Phase 6: Inference & Similarity Search

Phase 6 generates a **global player embedding** `z_p` per player from all their possessions (with the same future-info masking as training), then supports **similarity search** by cosine similarity in embedding space.

---

## Goal

- **6A**: For each player with enough possessions, compute a single 64-D embedding `z_p` by running the GNN on each possession (masked), gathering `h_player` per possession, and attention-pooling to one vector per player.
- **6B**: Given a query player_id, rank other players by cosine similarity to `z_p`, with optional filters (position, min possessions, exclude same team).

---

## 6A: Embedding generation

1. Load possession graphs and trained model.
2. For each graph: apply same future-info masking as training → run GNN → collect `h_player` for each player (actor nodes; one embedding per possession per player after deduplication).
3. Per player: attention-pool all their per-possession `h_player` into one **z_p**.
4. **Minimum 50 possessions** per player; players below this are excluded from the embedding table.

**Outputs**:

- **player_embeddings.npy**: (n_players × 64).
- **player_info.parquet**: player_id, name, position, team, etc., aligned with the embedding rows.

Processing uses mini-batch inference (`Batch.from_data_list`) for speed.

---

## 6B: Similarity search

- Given query player_id, take their `z_p`, compute cosine similarity to all other `z_p` in the table.
- Rank by similarity; optional filters: position group, min possessions, exclude same team.
- Return top-k similar players (and optionally similarity scores).

---

## Code

| Component | Location |
|-----------|----------|
| Embedding generator (batched, pooling) | `src/phase6_inference/embedding_generator.py` |
| Similarity search (cosine, filters) | `src/phase6_inference/similarity_search.py` |
| CLI entry | `main.py` → `--mode inference` (and search via API or script) |

---

## How to run

```bash
cd GNN_StatsBomb
python main.py --mode inference
```

Requires Phase 1–3 outputs and a trained checkpoint (Phase 5). Writes to `embeddings/` (e.g. `player_embeddings.npy`, `player_info.parquet`).

---

## See also

- [../SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) — Phase 6 (min_samples_per_player, outputs).
- [PHASE5.md](PHASE5.md) — Training (same masking and pooling logic).
- [PHASE7.md](PHASE7.md) — Analysis uses these embeddings and the model for situation-level comparison.
