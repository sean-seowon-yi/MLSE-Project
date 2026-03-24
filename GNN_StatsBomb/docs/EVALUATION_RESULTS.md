# Evaluation Results — Baseline vs Split-Context Model

> **Model**: Default baseline (unified context edges, no tag)
> **Embedding dimension**: 64-D
> **Players in embedding space**: 1,633
> **Possession graphs**: 51,778

---

## 1. Self-Consistency Evaluation

The self-consistency evaluation tests whether the model assigns **stable identity** — i.e., whether the same player, observed in different contexts, is recognised as their own closest match. Two complementary tests are run over the full set of possession graphs.

Both tests use **open-set retrieval**: non-testable players whose total possessions meet the inference threshold (50) are pooled and added to the retrieval gallery as distractors. This means ranks are measured against the full inference population of **1,633 players**, not just the testable subset.

### 1.1 Competition-Split Test

Players appearing in **two or more competitions** with at least 50 possessions each are split by competition. Embeddings from each competition half are pooled independently via the trained AttentionPooling, then self-retrieval rank is measured against the full gallery.

| Metric | Value |
|--------|-------|
| Testable players | 311 |
| Gallery size | 1,633 (311 testable + 1,322 distractors) |
| Self-cosine (mean / median / std) | 0.8273 / 0.8636 / 0.1588 |
| Cross-cosine (mean) | 0.3003 |
| **Cosine margin (self − cross)** | **+0.5270** |
| Mean rank | 185.0 / 1,633 (top 11.3%) |
| Median rank | 108 / 1,633 (top 6.6%) |
| Hit@1 | 2.2% |
| Hit@5 | 8.5% |
| Hit@10 | 13.0% |
| Hit@20 | 20.3% |
| Hit@50 | 33.3% |

#### By position group (competition-split)

| Position Group | n | Self-cosine | Mean Rank | Median Rank | Hit@1 | Hit@5 | Hit@10 |
|----------------|---|-------------|-----------|-------------|-------|-------|--------|
| Defender | 124 | 0.8273 | 164.6 | 103 | 1.2% | 8.5% | 11.7% |
| Forward | 71 | 0.8480 | 172.6 | 122 | 0.7% | 3.5% | 9.2% |
| Goalkeeper | 18 | 0.9973 | 28.6 | 18 | 16.7% | 25.0% | 36.1% |
| Midfielder | 98 | 0.7809 | 248.5 | 155 | 2.0% | 9.2% | 13.3% |

#### Notable competition-split retrievals

**Best self-retrievals (avg rank):**

| Player | Self-cosine | Rank (A→B) | Rank (B→A) | Avg |
|--------|-------------|------------|------------|-----|
| Hugo Lloris | 0.9988 | 1 | 1 | 1.0 |
| Antonio Rüdiger | 0.9720 | 1 | 1 | 1.0 |
| Thibaut Courtois | 0.9985 | 1 | 1 | 1.0 |
| Ingrid Filippa Angeldal | 0.9827 | 1 | 1 | 1.0 |
| Mario Pašalić | 0.9215 | 1 | 2 | 1.5 |
| Rúben Dias | 0.9723 | 2 | 2 | 2.0 |
| Pauline Peyraud Magnin | 0.9989 | 1 | 3 | 2.0 |
| Jurriën Timber | 0.9573 | 5 | 1 | 3.0 |
| Irene Paredes Hernandez | 0.9772 | 4 | 2 | 3.0 |

**Worst self-retrievals (avg rank):**

| Player | Self-cosine | Rank (A→B) | Rank (B→A) | Avg |
|--------|-------------|------------|------------|-----|
| Sergej Milinković-Savić | 0.1650 | 1,127 | 1,200 | 1,163.5 |
| Stephanie van der Gragt | −0.3230 | 1,350 | 1,407 | 1,378.5 |

---

### 1.2 Random-Half Test

All players with at least 100 possessions (regardless of competition count) are randomly split 50/50. Each half is pooled independently and self-retrieval rank is measured against the full gallery. This tests pure embedding stability without the confound of competition context shift.

| Metric | Value |
|--------|-------|
| Testable players | 1,083 |
| Gallery size | 1,633 (1,083 testable + 550 distractors) |
| Self-cosine (mean / median / std) | 0.8802 / 0.9009 / 0.1007 |
| Cross-cosine (mean) | 0.3024 |
| **Cosine margin (self − cross)** | **+0.5778** |
| Mean rank | 88.0 / 1,633 (top 5.4%) |
| Median rank | 46 / 1,633 (top 2.8%) |
| Hit@1 | 4.8% |
| Hit@5 | 13.9% |
| Hit@10 | 20.7% |
| Hit@20 | 31.4% |
| Hit@50 | 53.5% |

#### By position group (random-half)

| Position Group | n | Self-cosine | Mean Rank | Median Rank | Hit@1 | Hit@5 | Hit@10 |
|----------------|---|-------------|-----------|-------------|-------|-------|--------|
| Defender | 451 | 0.8873 | 81.9 | 50 | 2.2% | 9.8% | 17.1% |
| Forward | 217 | 0.8817 | 92.9 | 58 | 2.5% | 11.5% | 17.1% |
| Goalkeeper | 51 | 0.9986 | 5.0 | 2 | 45.1% | 72.5% | 85.3% |
| Midfielder | 364 | 0.8540 | 104.3 | 44 | 3.7% | 12.1% | 18.3% |

### 1.3 Interpretation

**The model demonstrably captures player identity.** The cosine margin of +0.53 to +0.58 is large — self-similarity is far above random cross-player similarity. Median ranks in the top 3–7% of the full 1,633-player gallery confirm the signal is real and consistent across both test variants.

**Open-set retrieval is the more realistic benchmark.** By including all inference-eligible players as distractors, the self-retrieval task matches the actual deployment scenario where a query player must be found among all 1,633 candidates. The percentile rankings (top 5–7% by median) are consistent with what was observed in the earlier closed-set evaluation, confirming that the model's quality signal is not an artifact of a small candidate pool.

**Goalkeeper results are inflated by positional distinctiveness.** With only 18 goalkeepers in the competition-split (51 in random-half), goalkeepers are easy to cluster away from outfield players. In the open-set gallery of 1,633, goalkeepers achieve mean rank 28.6 (competition-split) and 5.0 (random-half). The random-half result (Hit@1 = 45.1% among 1,633) demonstrates genuine individual identity learning beyond positional clustering, but the inherently low-variance goalkeeper action profile still makes this the easiest subgroup.

**Outfield positions are the more honest signal.** Defenders and forwards achieve mean rank 82–173 out of 1,633. Midfielders are the hardest group (mean rank 104–249), likely because midfield roles are the most tactically varied.

**Hit@1 is low (2–5%) across both tests.** The model gets players to the right neighbourhood but rarely pins down exact identity as the top-1 match. This is expected for a baseline with no player-aware contrastive sampling.

---

## 2. Pseudo Ground-Truth Evaluation

Ten player-similarity pairs were curated from public StatsBomb articles and recruitment case studies. For each pair (A, B), we compute the cosine similarity and the rank at which B appears in A's nearest-neighbour list (and vice versa), across all 1,633 players in the embedding space.

### 2.1 Per-Pair Results

| Tier | Player A | Player B | Cosine | Rank (A→B) | Rank (B→A) | Avg Rank |
|------|----------|----------|--------|------------|------------|----------|
| 1 | Vivianne Miedema | Caldentey | 0.6964 | 431 | 608 | 519.5 |
| 1 | Trent Alexander-Arnold | Achraf Hakimi | 0.9104 | 27 | 116 | 71.5 |
| 1 | Jordi Alba | Andrew Robertson | 0.8074 | 127 | 63 | 95.0 |
| 2 | Toni Kroos | Enzo Fernandez | 0.9427 | 35 | 76 | 55.5 |
| 2 | Trent Alexander-Arnold | Joakim Maehle | 0.7539 | 171 | 320 | 245.5 |
| 2 | Jadon Sancho | Ruben Vargas | 0.8442 | 71 | 258 | 164.5 |
| 2 | Jadon Sancho | Christoph Baumgartner | 0.7413 | 187 | 572 | 379.5 |
| 3 | Harry Kane | Rafael Leao | 0.8342 | 303 | 147 | 225.0 |
| 3 | Harry Kane | Joao Felix | 0.8329 | 307 | 158 | 232.5 |
| 3 | Felix Uduokhai | Harry Souttar | 0.8153 | 220 | 193 | 206.5 |

### 2.2 Aggregate Statistics

| Metric | All Pairs | Tier 1 | Tier 2 | Tier 3 |
|--------|-----------|--------|--------|--------|
| Pairs evaluated | 10 | 3 | 4 | 3 |
| Mean cosine | 0.8179 | — | — | — |
| Mean rank | 219.5 | 228.7 | 211.2 | 221.3 |
| Median rank | 179 | 121 | 179 | 206 |
| Hit@5 | 0.0% | 0.0% | 0.0% | 0.0% |
| Hit@10 | 0.0% | 0.0% | 0.0% | 0.0% |
| Hit@20 | 0.0% | 0.0% | 0.0% | 0.0% |
| Hit@50 | 10.0% | — | — | — |

### 2.3 Interpretation

**The pseudo ground-truth results are poor for fine-grained similarity.** No pair achieves a rank within the top 20 in either direction. The mean rank of 219.5 out of 1,633 players (top 13.4%) is better than random (expected ~817) but far from the top-5 retrieval that StatsBomb's own system achieves.

**Important caveats on this evaluation:**

1. **Domain mismatch.** StatsBomb's similarity metrics use a proprietary feature set (radar attributes, weighted skill scores) that differs fundamentally from our event-sequence GNN approach. The two systems are not solving the same problem.
2. **Data coverage mismatch.** StatsBomb's articles use club-level data spanning full seasons; our dataset covers only international tournaments (Euro 2020/2024, World Cup 2022, Women's World Cup 2023, Women's Euro 2022). A player's international tournament profile can differ substantially from their club profile.
3. **Low possession counts.** Several players (Sancho: 72, Uduokhai: 80) are near the 50-possession minimum, limiting embedding reliability.
4. **Tier 3 pairs are inherently weak.** The Kane–Leao/Felix and Uduokhai–Souttar pairs are conditional on altered similarity weighting or transitive inference, making them unlikely to match in any embedding space.

**Best-performing pair:** Kroos → Enzo Fernandez (rank 35, cosine 0.9427) is the closest to a successful retrieval, consistent with both being metronomic deep-lying midfielders.

**The self-consistency evaluation is a more reliable indicator of model quality** than this pseudo ground-truth, because it tests the model's own internal consistency without relying on external labels derived from a different methodology and data source.

---

## 3. Summary

| Evaluation | Key Metric | Value | Interpretation |
|------------|-----------|-------|----------------|
| Competition-split self-consistency | Median rank | 108 / 1,633 (top 6.6%) | Strong identity signal across tournaments |
| Competition-split self-consistency | Hit@50 | 33.3% | 1 in 3 players retrieved within top 50 of 1,633 |
| Competition-split self-consistency | Cosine margin | +0.527 | Large separation between self and cross |
| Random-half self-consistency | Median rank | 46 / 1,633 (top 2.8%) | Stable embeddings under random split |
| Random-half self-consistency | Hit@50 | 53.5% | Majority of players retrieved within top 50 of 1,633 |
| Pseudo ground-truth | Mean rank | 219.5 / 1,633 | Better than random, far from top-K |
| Pseudo ground-truth | Hit@20 | 0.0% | No pair retrieved in top 20 |

The baseline model successfully learns player identity from raw event sequences — the self-consistency tests confirm this with large cosine margins and median ranks in the top 3–7% of the full 1,633-player gallery. The pseudo ground-truth underperforms because it measures cross-methodology agreement rather than intrinsic model quality. See Section 4 for a four-model ablation study comparing split-context edges, player-aware sampling, and their combination against this baseline.

---

## 4. Four-Model Ablation Study

Four model variants were trained and evaluated under identical conditions (64-D embeddings, 1,633-player gallery, early stopping with patience 15):

| Label | CLI flags | Description |
|-------|-----------|-------------|
| **Baseline** | *(none)* | Unified context edges, random batch sampling |
| **Split-ctx** | `--split_context_edges --tag split_ctx` | Teammate/opponent edge types, random batch sampling |
| **Player-samp** | `--player_sampling --tag player_samp` | Unified context edges, player-aware batch sampler (K=16, M=6) |
| **Combined** | `--split_context_edges --player_sampling --tag split_ctx_ps` | Both changes active |

#### Training metadata

| | Baseline | Split-ctx | Player-samp | Combined |
|---|----------|-----------|-------------|----------|
| Epochs trained | 78 | 88 | 60 | 67 |
| Best val loss | 1.896 | 1.946 | 1.577 | 1.622 |
| Final LR | 6.25e-5 | 3.125e-5 | 1.25e-4 | 6.25e-5 |

The player-sampling models converge faster (60–67 epochs vs 78–88) and reach substantially lower validation loss (1.577–1.622 vs 1.896–1.946), confirming the sampler provides stronger gradient signal to the contrastive head.

### 4.1 Self-Consistency (Open-Set) Comparison

#### Competition-split (311 testable, gallery 1,633)

| Metric | Baseline | Split-ctx | Player-samp | Combined |
|--------|----------|-----------|-------------|----------|
| Self-cosine (mean) | 0.8273 | 0.8145 | 0.5610 | 0.5200 |
| Cosine margin | +0.5270 | +0.5363 | +0.5106 | +0.4820 |
| Mean rank | **185.0** | 187.2 | 219.5 | 234.8 |
| Median rank | **108** | 109 | 126 | 137 |
| Hit@1 | **2.2%** | 1.6% | 1.0% | 0.5% |
| Hit@5 | **8.5%** | 7.1% | 3.9% | 3.5% |
| Hit@10 | **13.0%** | 10.8% | 7.2% | 6.3% |
| Hit@20 | **20.3%** | 17.0% | 12.2% | 9.0% |
| Hit@50 | **33.3%** | 31.0% | 27.3% | 21.1% |

#### Position-group breakdown (competition-split)

| Position | n | Baseline MR | Split-ctx MR | P-samp MR | Combined MR | Baseline H@10 | Split-ctx H@10 | P-samp H@10 | Combined H@10 |
|----------|---|-------------|-------------|-----------|-------------|---------------|----------------|-------------|----------------|
| Defender | 124 | **164.6** | 186.3 | 217.4 | 246.4 | **11.7%** | 10.5% | 6.1% | 5.2% |
| Forward | 71 | **172.6** | 184.1 | 174.8 | 188.2 | **9.2%** | 6.3% | 7.8% | 10.6% |
| Goalkeeper | 18 | 28.6 | **25.8** | 47.8 | 60.8 | 36.1% | **41.7%** | 19.4% | 2.8% |
| Midfielder | 98 | 248.5 | **220.2** | 286.0 | 285.9 | **13.3%** | 8.7% | 6.1% | 5.1% |

#### Random-half (1,083 testable, gallery 1,633)

| Metric | Baseline | Split-ctx | Player-samp | Combined |
|--------|----------|-----------|-------------|----------|
| Self-cosine (mean) | 0.8802 | 0.8681 | 0.6623 | 0.6439 |
| Cosine margin | +0.5778 | +0.5877 | **+0.6065** | +0.6011 |
| Mean rank | **88.0** | 94.0 | 123.8 | 117.1 |
| Median rank | **46** | 50 | 69 | 65 |
| Hit@1 | 4.8% | **5.9%** | 3.0% | 2.9% |
| Hit@5 | 13.9% | **14.4%** | 9.0% | 9.1% |
| Hit@10 | 20.7% | **20.9%** | 13.8% | 14.7% |
| Hit@20 | **31.4%** | 30.8% | 21.2% | 22.9% |
| Hit@50 | **53.5%** | 50.0% | 40.7% | 41.4% |

#### Position-group breakdown (random-half)

| Position | n | Baseline MR | Split-ctx MR | P-samp MR | Combined MR | Baseline H@10 | Split-ctx H@10 | P-samp H@10 | Combined H@10 |
|----------|---|-------------|-------------|-----------|-------------|---------------|----------------|-------------|----------------|
| Defender | 451 | **81.9** | 86.7 | 106.6 | 106.9 | 17.1% | **18.6%** | 13.6% | 13.8% |
| Forward | 217 | 92.9 | 96.2 | 119.1 | **105.4** | **17.1%** | 12.0% | 13.6% | **18.0%** |
| Goalkeeper | 51 | 5.0 | **4.8** | 60.7 | 55.5 | **85.3%** | **91.2%** | 5.9% | 11.8% |
| Midfielder | 364 | **104.3** | 114.1 | 156.8 | 145.2 | 18.3% | **19.1%** | 15.1% | 14.3% |

### 4.2 Pseudo Ground-Truth Comparison

| Metric | Baseline | Split-ctx | Player-samp | Combined |
|--------|----------|-----------|-------------|----------|
| Mean cosine | **0.8179** | 0.7847 | 0.4466 | 0.2434 |
| Mean rank | **219.5** | 236.1 | 291.1 | 565.1 |
| Median rank | **179** | 208 | 292 | 518 |
| Hit@50 | 10.0% | **15.0%** | 5.0% | 0.0% |

#### Tier-level mean rank (lower is better)

| Tier | Baseline | Split-ctx | Player-samp | Combined |
|------|----------|-----------|-------------|----------|
| Tier 1 | 228.7 | **101.8** | 216.2 | 518.0 |
| Tier 2 | **211.2** | 330.8 | 292.1 | 720.4 |
| Tier 3 | **221.3** | 244.2 | 364.7 | 405.3 |

#### Per-pair comparison (avg rank, lower is better)

| Tier | Player A | Player B | Baseline | Split-ctx | Player-samp | Combined |
|------|----------|----------|----------|-----------|-------------|----------|
| 1 | Miedema | Caldentey | 519.5 | **144.5** | 439.0 | 490.0 |
| 1 | TAA | Hakimi | **71.5** | 90.0 | 117.5 | 973.0 |
| 1 | Alba | Robertson | 95.0 | **71.0** | 92.0 | 91.0 |
| 2 | Kroos | Enzo Fernandez | **55.5** | 54.0 | 206.5 | 191.5 |
| 2 | TAA | Maehle | **245.5** | 360.5 | 360.0 | 1552.0 |
| 2 | Sancho | Vargas | **164.5** | 415.0 | 80.0 | 616.0 |
| 2 | Sancho | Baumgartner | **379.5** | 493.5 | 522.0 | 522.0 |
| 3 | Kane | Leao | **225.0** | 269.5 | 515.5 | 760.0 |
| 3 | Kane | Felix | **232.5** | 248.0 | 177.5 | 180.5 |
| 3 | Uduokhai | Souttar | **206.5** | 215.0 | 401.0 | 275.5 |

### 4.3 Findings

1. **Baseline is the strongest model on aggregate retrieval.** Across both self-consistency tests, the baseline achieves the best mean and median ranks, and the best Hit@K at almost every threshold. The ranking is consistently: Baseline > Split-ctx > Combined > Player-samp on competition-split, while Combined slightly edges Player-samp on random-half.

2. **Player-aware sampling lowers training loss but hurts evaluation.** The player-sampling models reach substantially lower val loss (1.577–1.622 vs 1.896–1.946) and converge faster (60–67 vs 78–88 epochs). However, this does *not* translate to better retrieval. Mean self-cosine drops from 0.83–0.88 to 0.52–0.66, and Hit@10 is roughly halved. The sampler appears to overfit the contrastive objective at the expense of general embedding quality.

3. **Player-sampling collapses goalkeeper identity.** The most dramatic regression is for goalkeepers: baseline achieves Hit@10 of 36.1% (competition-split) and 85.3% (random-half), while player-samp drops to 19.4% and 5.9%. The player-aware batching concentrates training signal on players with many possessions, potentially underrepresenting the small goalkeeper cohort and destroying their distinctive embeddings.

4. **Cosine margin is not predictive of retrieval quality.** Player-samp and Combined achieve higher random-half cosine margins (+0.6065 and +0.6011) than baseline (+0.5778), yet have worse ranks. The sampler pushes same-player embeddings apart from cross-player ones in cosine space, but the resulting embedding geometry is less discriminative for nearest-neighbour retrieval.

5. **Split-context edges are a mild, mixed change.** Relative to baseline, split-ctx yields small regressions on most metrics but improves goalkeepers, random-half Hit@1/5/10, and Tier 1 ground-truth pairs. It is the only variant that improves any ground-truth tier.

6. **Ground-truth degrades sharply with player-sampling.** Mean rank worsens from 219.5 (baseline) to 291.1 (player-samp) to 565.1 (combined). Cosine similarities between known-similar pairs plummet (0.82 → 0.45 → 0.24), with some pairs going negative (TAA–Maehle: -0.36 in combined). The sampler's emphasis on same-player attraction evidently disrupts the cross-player similarity structure.

7. **Combining both changes is worse than either alone.** The combined model underperforms both standalone variants on nearly every metric, suggesting the two modifications interact negatively — player-aware sampling with split context edges may amplify the embedding distortion seen in each.

### 4.4 Implications for Future Work

The player-aware sampling hypothesis — that guaranteeing positive pairs in every batch would improve identity retrieval — is **not supported** by these results. While the approach successfully optimises the contrastive loss (lower val loss, faster convergence), the resulting embeddings lose general discriminative power. Possible explanations and next steps:

- **Temperature/loss-weight rebalancing.** The InfoNCE temperature (0.05) was tuned for the baseline's sparse positive pairs. With guaranteed K×M positives per batch, the effective gradient may be too concentrated, collapsing the embedding space around player-identity axes while sacrificing positional and stylistic structure. A higher temperature or reduced contrastive loss weight may help.
- **Auxiliary loss underweighted.** The uniformity and classification losses may need higher relative weight to counterbalance the stronger contrastive signal from the player-aware sampler.
- **K and M hyperparameters.** K=16 players × M=6 possessions yields 96-sample batches with very high positive-pair density. A lower M (e.g., 2–3) or higher K might preserve more diversity.
- **Evaluation mismatch.** The self-consistency evaluation pools *all* possessions per split, including contexts where the player may be marginal. The sampler trains on actor-possessions only. This mismatch could explain the cosine-margin/rank divergence.
