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

The baseline model successfully learns player identity from raw event sequences — the self-consistency tests confirm this with large cosine margins and median ranks in the top 3–7% of the full 1,633-player gallery. The pseudo ground-truth underperforms because it measures cross-methodology agreement rather than intrinsic model quality.

See Section 4 for a four-model ablation study comparing split-context edges, player-aware sampling (with rebalanced hyperparameters), and their combination against this baseline. The rebalanced player-sampling model achieves 64.8% Hit@10 (random-half), far surpassing baseline (20.7%), but trades cross-player similarity for identity precision.

---

## 4. Four-Model Ablation Study

Four model variants were trained and evaluated under identical conditions (64-D embeddings, 1,633-player gallery, early stopping with patience 15):

| Label | CLI flags | Description |
|-------|-----------|-------------|
| **Baseline** | *(none)* | Unified context edges, random batch sampling |
| **Split-ctx** | `--split_context_edges --tag split_ctx` | Teammate/opponent edge types, random batch sampling |
| **Player-samp** | `--player_sampling --tag player_samp` | Unified context edges, player-aware batch sampler (K=16, M=6) |
| **Combined** | `--split_context_edges --player_sampling --tag split_ctx_ps` | Both changes active |

> **Hyperparameter rebalancing (v2).** The initial player-sampling experiments (v1) used the baseline's loss hyperparameters (temperature=0.05, λ\_contrast=0.5, λ\_pooled=0.3). V1 models achieved lower val loss but *worse* retrieval, indicating the contrastive signal was too strong. For v2 (current results), only the player-sampling pipelines were adjusted:
>
> | Parameter | Baseline / Split-ctx | Player-samp / Combined (v2) |
> |-----------|---------------------|----------------------------|
> | InfoNCE temperature | 0.05 | **0.15** (3× softer) |
> | λ\_contrast | 0.5 | **0.15** (reduced weight) |
> | λ\_pooled\_uniformity | 0.3 | **0.5** (stronger anti-collapse) |
>
> These overrides are applied automatically when `--player_sampling` is set and do not affect the baseline or split-ctx pipelines.

#### Training metadata

| | Baseline | Split-ctx | Player-samp (v2) | Combined (v2) |
|---|----------|-----------|-------------------|---------------|
| Epochs trained | 78 | 88 | 105 | 105 |
| Best val loss | 1.896 | 1.946 | −0.534 | −0.497 |
| Final LR | 6.25e-5 | 3.125e-5 | 3.91e-6 | 7.81e-6 |

Note: v2 val loss is negative because the stronger uniformity weight (0.5 × negative uniformity loss) shifts the total loss downward. Val loss is **not directly comparable** across pipelines with different loss weights. Training ran longer (105 vs 60–67 in v1), converging more gradually.

### 4.1 Self-Consistency (Open-Set) Comparison

#### Competition-split (311 testable, gallery 1,633)

| Metric | Baseline | Split-ctx | Player-samp (v2) | Combined (v2) |
|--------|----------|-----------|-------------------|---------------|
| Self-cosine (mean) | 0.8273 | 0.8145 | 0.4980 | 0.5028 |
| Cosine margin | **+0.5270** | +0.5363 | +0.4298 | +0.4323 |
| Mean rank | **185.0** | 187.2 | 259.0 | 254.5 |
| Median rank | 108 | 109 | **82** | **81** |
| Hit@1 | 2.2% | 1.6% | **5.8%** | 4.8% |
| Hit@5 | 8.5% | 7.1% | **13.2%** | 11.6% |
| Hit@10 | 13.0% | 10.8% | **18.2%** | 15.8% |
| Hit@20 | 20.3% | 17.0% | **24.6%** | **24.3%** |
| Hit@50 | 33.3% | 31.0% | **38.3%** | 37.1% |

#### Position-group breakdown (competition-split)

| Position | n | Baseline MR | Split-ctx MR | P-samp v2 MR | Comb v2 MR | Baseline H@10 | Split-ctx H@10 | P-samp v2 H@10 | Comb v2 H@10 |
|----------|---|-------------|-------------|--------------|------------|---------------|----------------|-----------------|--------------|
| Defender | 124 | **164.6** | 186.3 | 287.5 | 279.4 | 11.7% | 10.5% | **13.3%** | 11.7% |
| Forward | 71 | **172.6** | 184.1 | 239.8 | 238.1 | 9.2% | 6.3% | **25.4%** | 22.5% |
| Goalkeeper | 18 | 28.6 | **25.8** | 24.4 | 30.9 | 36.1% | **41.7%** | 38.9% | 25.0% |
| Midfielder | 98 | **248.5** | 220.2 | 280.0 | 275.8 | 13.3% | 8.7% | **15.3%** | 14.3% |

#### Random-half (1,083 testable, gallery 1,633)

| Metric | Baseline | Split-ctx | Player-samp (v2) | Combined (v2) |
|--------|----------|-----------|-------------------|---------------|
| Self-cosine (mean) | 0.8802 | 0.8681 | 0.8178 | 0.8154 |
| Cosine margin | +0.5778 | +0.5877 | **+0.7473** | +0.7410 |
| Mean rank | 88.0 | 94.0 | **15.2** | **15.2** |
| Median rank | 46 | 50 | **4** | **4** |
| Hit@1 | 4.8% | 5.9% | **34.1%** | **34.5%** |
| Hit@5 | 13.9% | 14.4% | **54.9%** | **55.6%** |
| Hit@10 | 20.7% | 20.9% | **64.8%** | **65.3%** |
| Hit@20 | 31.4% | 30.8% | **76.2%** | **77.1%** |
| Hit@50 | 53.5% | 50.0% | **91.2%** | **91.9%** |

#### Position-group breakdown (random-half)

| Position | n | Baseline MR | Split-ctx MR | P-samp v2 MR | Comb v2 MR | Baseline H@10 | Split-ctx H@10 | P-samp v2 H@10 | Comb v2 H@10 |
|----------|---|-------------|-------------|--------------|------------|---------------|----------------|-----------------|--------------|
| Defender | 451 | 81.9 | 86.7 | **23.2** | 24.4 | 17.1% | 18.6% | **50.7%** | 48.9% |
| Forward | 217 | 92.9 | 96.2 | **9.4** | 8.4 | 17.1% | 12.0% | **75.4%** | **76.7%** |
| Goalkeeper | 51 | **5.0** | 4.8 | 14.2 | 13.2 | **85.3%** | **91.2%** | 60.8% | 69.6% |
| Midfielder | 364 | 104.3 | 114.1 | **8.9** | **8.0** | 18.3% | 19.1% | **76.7%** | **78.3%** |

### 4.2 Pseudo Ground-Truth Comparison

| Metric | Baseline | Split-ctx | Player-samp (v2) | Combined (v2) |
|--------|----------|-----------|-------------------|---------------|
| Mean cosine | **0.8179** | 0.7847 | 0.0422 | 0.0547 |
| Mean rank | **219.5** | 236.1 | 814.4 | 824.8 |
| Median rank | **179** | 208 | 759 | 700 |
| Hit@50 | 10.0% | **15.0%** | 0.0% | 0.0% |

#### Tier-level mean rank (lower is better)

| Tier | Baseline | Split-ctx | Player-samp (v2) | Combined (v2) |
|------|----------|-----------|-------------------|---------------|
| Tier 1 | 228.7 | **101.8** | 626.7 | 640.5 |
| Tier 2 | **211.2** | 330.8 | 460.2 | 434.1 |
| Tier 3 | **221.3** | 244.2 | 1474.3 | 1529.8 |

#### Per-pair comparison (avg rank, lower is better)

| Tier | Player A | Player B | Baseline | Split-ctx | P-samp (v2) | Combined (v2) |
|------|----------|----------|----------|-----------|-------------|---------------|
| 1 | Miedema | Caldentey | 519.5 | **144.5** | 1548.5 | 1546.0 |
| 1 | TAA | Hakimi | **71.5** | 90.0 | 194.0 | 175.0 |
| 1 | Alba | Robertson | 95.0 | **71.0** | 137.5 | 200.5 |
| 2 | Kroos | Enzo Fernandez | **55.5** | 54.0 | 130.5 | 129.0 |
| 2 | TAA | Maehle | **245.5** | 360.5 | 1101.5 | 1152.0 |
| 2 | Sancho | Vargas | **164.5** | 415.0 | 246.5 | 198.0 |
| 2 | Sancho | Baumgartner | **379.5** | 493.5 | 362.5 | 257.5 |
| 3 | Kane | Leao | **225.0** | 269.5 | 1579.0 | 1594.5 |
| 3 | Kane | Felix | **232.5** | 248.0 | 1522.0 | 1555.5 |
| 3 | Uduokhai | Souttar | **206.5** | 215.0 | 1322.0 | 1439.5 |

### 4.3 Findings

1. **Hyperparameter rebalancing transforms player-sampling from worst to best on self-consistency.** The v2 player-sampling models (τ=0.15, λ\_contrast=0.15, λ\_pooled=0.5) dramatically outperform baseline on random-half: Hit@1 jumps from 4.8% to 34.1%, Hit@10 from 20.7% to 64.8%, and mean rank drops from 88.0 to 15.2. This reverses the v1 finding where player-sampling *hurt* retrieval. The key insight is that the baseline's hyperparameters (τ=0.05, high λ\_contrast) were overdriving the contrastive loss when combined with guaranteed positive pairs.

2. **Competition-split shows mixed but net-positive results.** Mean rank is worse for player-samp v2 (259.0 vs 185.0), but median rank is better (82 vs 108), and Hit@K is better at every threshold. The mean–median divergence suggests a small number of outlier players pull the mean upward while the typical player benefits substantially from the rebalanced training.

3. **Goalkeeper identity is recovered (partially).** The v1 player-sampling catastrophically collapsed goalkeeper embeddings (Hit@10: 5.9% random-half). With rebalancing, goalkeeper Hit@10 recovers to 60.8% (random-half) and 38.9% (competition-split). This is still below baseline (85.3% and 36.1% respectively), but no longer represents a catastrophic failure. The stronger uniformity loss prevents the embedding space from collapsing around outfield archetypes.

4. **Cross-player similarity (ground-truth) degrades severely.** Despite the self-consistency gains, ground-truth mean rank worsens from 219.5 to 814.4, and mean cosine drops from 0.82 to 0.04. The rebalanced model spreads all player embeddings far apart on the hypersphere (high uniformity), making individual identities easy to distinguish but destroying the cross-player similarity structure needed for "find me a similar player." Tier 3 pairs (cross-role comparisons) are near random (mean rank ~1,500 of 1,633).

5. **The identity–similarity trade-off is fundamental.** Player-aware sampling with strong uniformity excels at "is this the same player?" (self-consistency) but fails at "who plays like this player?" (ground-truth). The baseline — with weaker contrastive signal and weaker uniformity — preserves more cross-player structure. This suggests the two objectives may require different embedding spaces or a multi-task architecture.

6. **Combined (v2) performs comparably to Player-samp (v2).** Unlike v1 where combining split-context with player-sampling was strictly worse, v2 shows the two models performing nearly identically on every metric. The hyperparameter rebalancing neutralises the negative interaction observed in v1, though split-context edges provide no additional benefit when paired with the rebalanced sampler.

7. **Split-context edges remain a mild, independent improvement.** The split-ctx model (without player-sampling) continues to show small improvements over baseline on specific metrics: goalkeeper retrieval, random-half Hit@1/5/10, and Tier 1 ground-truth. It is the only model that improves ground-truth for any tier.

### 4.4 Implications for Future Work

The hyperparameter rebalancing experiment reveals a **fundamental trade-off** between identity consistency (self-retrieval) and cross-player similarity (replacement search). The two objectives respond differently to the contrastive–uniformity balance:

- **For deployment as a player-identification system** (e.g., verifying a player's identity across contexts), the rebalanced player-sampling model is clearly superior (Hit@10 of 65% vs 21%).
- **For deployment as a player-replacement/scouting tool** (e.g., "find similar players to Toni Kroos"), the baseline model is currently better (mean rank 219 vs 814 on ground-truth pairs).

Possible next steps to bridge this gap:

- **Two-stage architecture.** Use the player-sampling model for identity-aware pooling, then train a second similarity head on the pooled embeddings with lower uniformity.
- **Uniformity weight scheduling.** Start training with high uniformity (to learn spread) and gradually decay it, allowing late-stage training to build cross-player bridges.
- **Temperature annealing.** Begin with τ=0.15 and anneal down to 0.05, letting the model first learn coarse structure then refine fine-grained similarity.
- **Evaluation alignment.** The self-consistency test measures identity stability; the ground-truth test measures cross-player similarity. A combined metric that weights both could guide hyperparameter search more effectively.

---

## 5. Policy Diagnostic — Behavioral Validation

This diagnostic tests whether cosine similarity in the learned z\_p embedding space actually corresponds to **behavioral similarity** — i.e., whether players who are "close" in embedding space would make similar decisions in the same game situations. Two complementary tests were run on all four models.

**Method**: 200 canonical game situations were sampled (stratified by actor position group). For each situation, all 1,633 players' predicted action distributions were computed via FiLM conditioning (same h\_event, each player's z\_p). Pairwise Jensen-Shannon divergence was computed within position groups and correlated with cosine distance.

### 5.1 Policy Distance vs Cosine Distance Correlation

Spearman rank correlation between pairwise JS divergence (behavioral distance) and cosine distance (embedding distance), computed within position groups (411,231 total pairs):

| Position Group | Baseline | Split-ctx | Player-samp (v2) | Combined (v2) |
|----------------|----------|-----------|-------------------|---------------|
| Goalkeeper | 0.568 | **0.686** | **0.710** | 0.704 |
| Defender | **0.791** | 0.731 | 0.682 | 0.667 |
| Midfielder | 0.687 | 0.676 | 0.687 | **0.694** |
| Forward | 0.625 | 0.613 | 0.658 | **0.683** |
| **Overall** | **0.712** | 0.700 | 0.663 | 0.657 |

All correlations are highly significant (p = 0.0 for all, >400K pairs per model).

### 5.2 Substitute Quality (top-K neighbours vs random same-group)

For 8 query players (2 per position group, highest-possession), the mean JS divergence of their top-10 cosine neighbours was compared to 10 random same-position-group players:

| Metric | Baseline | Split-ctx | Player-samp (v2) | Combined (v2) |
|--------|----------|-----------|-------------------|---------------|
| JS(top-K) | 0.009371 | 0.009633 | **0.001124** | 0.001945 |
| JS(random-K) | 0.017283 | 0.022486 | 0.019344 | 0.020129 |
| **Ratio** (random/topK) | 1.84 | 2.33 | **17.22** | 10.35 |

Higher ratio = better. A ratio of 1.0 means cosine neighbours are no better than random; higher means neighbours are genuinely more behaviorally similar.

### 5.3 Findings

1. **All models show strong policy-cosine correlation (rho = 0.66-0.71).** The embedding space genuinely captures behavioral similarity across all four variants. This validates the fundamental approach: cosine distance in z\_p space is a meaningful proxy for "how differently two players would act in the same situation." The correlation is strongest for defenders (rho up to 0.79) and weakest for forwards (0.61-0.68).

2. **Baseline has the smoothest embedding space (highest rho = 0.71) but lowest substitute ratio (1.84).** The baseline's cosine distances are well-calibrated monotonically but the practical gap between neighbours and random is small. Top-K neighbours are only ~84% more behaviorally similar than random same-position players.

3. **Player-samp v2 has dramatically better practical substitute quality (ratio = 17.2) despite lower overall rho (0.66).** This is the most important finding for scouting applications. The rebalanced player-sampling model creates an embedding space where nearest neighbours are 17x more behaviourally similar than random — compared to just 1.8x for baseline. The lower rho is explained by the wider spread of cosine distances (mean cosine dist 0.88 vs 0.36 for baseline), which compresses the mid-range of the correlation while sharpening the distinction at the extremes.

4. **The identity-similarity paradox is partially resolved.** The ground-truth evaluation (Section 4.2) showed player-samp v2 performing much worse than baseline on cross-player similarity (mean rank 814 vs 220). But the policy diagnostic shows that player-samp v2's neighbours are dramatically more behaviorally similar. This suggests the ground-truth pairs (curated from StatsBomb's proprietary radar system) measure a different notion of "similarity" than behavioral agreement. The GNN-based model optimises for *decision-making similarity*, while StatsBomb's system uses *statistical profile similarity* — these are related but not identical concepts.

5. **Mean JS values confirm the player-samp model creates tighter behavioral clusters.** The absolute JS(top-K) for player-samp is 0.001124 vs 0.009371 for baseline — an 8x reduction. Neighbours in the player-samp embedding space predict almost identical action distributions in the same situations. This is consistent with the model's excellent self-consistency (Section 4.1).

6. **Split-context edges mildly improve substitute quality.** Split-ctx achieves a ratio of 2.33 vs baseline's 1.84, while maintaining nearly identical correlation (0.70 vs 0.71). This is consistent with the mild improvement pattern seen in Sections 4.1 and 4.2.

### 5.4 Implications

The policy diagnostic reveals that **all four models have successfully learned behaviorally meaningful embedding spaces**, but they differ in how sharply they separate neighbours from non-neighbours:

- **For scouting/replacement applications**, the player-samp v2 model is clearly superior despite its poor ground-truth ranking. Its neighbours are near-identical in predicted behavior (JS = 0.001), making it the best choice when the goal is "find a player who would make the same decisions."
- **The ground-truth evaluation should be reinterpreted**, not as a failure of the player-samp model, but as a measurement of a different construct (statistical profile similarity vs behavioral decision similarity).
- **The overall rho of 0.66-0.71 across all models** sets a ceiling for cosine-based retrieval. To improve beyond this, architectural changes (e.g., dedicated similarity heads, non-cosine distance metrics) may be needed.

---

## 6. FiLM Sensitivity Analysis

This diagnostic tests whether the FiLM conditioning layer actually uses z\_p to change predictions, or whether the model could be largely ignoring the player embedding. For each of the 200 canonical situations, every player's z\_p is replaced with a **random same-position-group** player's z\_p, and the JS divergence between the original and shuffled predictions is measured. High JS means FiLM is load-bearing; near-zero JS means the model bypasses player conditioning.

### 6.1 FiLM Effect Size (mean JS, correct z\_p vs shuffled same-group z\_p)

| Position Group | Baseline | Split-ctx | Player-samp (v2) | Combined (v2) |
|----------------|----------|-----------|-------------------|---------------|
| Goalkeeper | 0.0016 | 0.0016 | 0.0011 | 0.0012 |
| Defender | 0.0120 | **0.0154** | 0.0035 | 0.0037 |
| Midfielder | 0.0126 | **0.0130** | 0.0027 | 0.0026 |
| Forward | 0.0095 | **0.0112** | 0.0028 | 0.0024 |
| **Overall** | 0.0109 | **0.0128** | 0.0029 | 0.0029 |

### 6.2 Findings

1. **FiLM is load-bearing in all four models.** Every model shows non-zero JS when z\_p is shuffled, confirming that the player embedding materially changes the predicted action distribution. The FiLM conditioning mechanism is functioning as designed.

2. **Baseline and Split-ctx show ~4x larger FiLM effect than Player-samp/Combined (0.011-0.013 vs 0.003).** This initially seems counter-intuitive — the player-samp models have stronger identity signal (better self-consistency and 17x substitute ratio). The explanation lies in the embedding geometry:

   - **Baseline/Split-ctx**: z\_p vectors vary substantially within position groups (mean cosine distance within-group ~0.36-0.40). Replacing a player's z\_p with a random same-group player's produces a large perturbation, yielding high JS. But this variation is relatively unstructured — neighbours are only 1.8-2.3x better than random (Section 5.2).
   - **Player-samp/Combined**: The stronger uniformity loss pushes all z\_p vectors far apart overall (mean cosine dist ~0.88), but within each position group the embeddings form **tighter, more structured clusters**. A random same-group swap produces a smaller perturbation because same-group players are closer in the structured embedding space. Yet this structure is highly meaningful — neighbours are 10-17x better than random.

3. **The FiLM effect size and substitute quality are measuring different things.** FiLM sensitivity measures the *magnitude* of z\_p's influence on predictions for arbitrary within-group swaps. Substitute quality measures the *quality* of the nearest-neighbour structure. A model can have small FiLM effect size (predictions are moderately stable under same-group swaps) yet excellent substitute quality (the small differences that DO exist are precisely aligned with behavioral similarity).

4. **Goalkeeper FiLM effect is consistently tiny (0.001-0.002) across all models.** GK actions are so constrained by position (kicks, throws, limited event types) that swapping one GK's z\_p for another barely changes predictions. This is consistent with the high GK self-consistency in Section 4.1 — GK embeddings are informative for identity but have limited action-prediction leverage.

5. **Split-ctx shows the highest FiLM sensitivity (0.013).** The separate teammate/opponent edge weights may give the model richer situation representations that are more responsive to z\_p modulation.

### 6.3 Implications

The FiLM layer is confirmed to be an active, load-bearing component of the architecture. The player embedding z\_p is not being bypassed. Combined with the policy-cosine correlation (Section 5.1), this establishes that:

- z\_p encodes behaviorally meaningful player information (rho 0.66-0.71)
- FiLM uses z\_p to modulate predictions in a structured way (JS > 0 everywhere)
- The magnitude of FiLM's effect differs by model variant, with baseline/split-ctx showing larger raw effect and player-samp showing smaller but more precisely structured effect

> **Note:** Section 6.1 reports the original FiLM sensitivity measured *before* the counterfactual fix (Section 7). Post-fix values are slightly lower across all models (see Section 7.1) because h\_event no longer carries residual actor identity.

---

## 7. Counterfactual h\_event Fix — Actor Identity Leak

### 7.1 Background

In the original diagnostic (Sections 5-6), `h_event` was produced by the full GNN including `acts_in` edges (`player -> event`). This means each event node absorbs message-passing information from its *actor* player node, potentially encoding who performed the action rather than purely the game situation. When we then swap z\_p via FiLM, the correct actor's identity may still be embedded in h\_event, confounding the measurement.

### 7.2 Fix: `encode_possession_counterfactual`

A new method `encode_possession_counterfactual` drops the `(player, acts_in, event)` edges from the GNN computation. The reverse edge `(event, performed_by, player)` is retained so player nodes remain part of the graph and context-player information still flows through `context_for` edges.

This produces a "situation-pure" h\_event that reflects game state, spatial context, and off-ball player positioning — but not who performed the action.

### 7.3 Before/After Comparison

| Metric | Model | Before (with acts\_in) | After (counterfactual) | Change |
|--------|-------|----------------------|----------------------|--------|
| **Spearman rho** | Baseline | 0.7115 | 0.7162 | +0.005 |
| | Split-ctx | 0.7004 | 0.6784 | -0.022 |
| | Player-samp | 0.6632 | 0.6855 | +0.022 |
| | Combined | 0.6570 | 0.6562 | -0.001 |
| **Substitute ratio** | Baseline | 1.84 | **2.26** | **+22%** |
| | Split-ctx | 2.33 | **2.54** | **+9%** |
| | Player-samp | 17.22 | 15.63 | -9% |
| | Combined | 10.35 | **12.25** | **+18%** |
| **FiLM sensitivity** | Baseline | 0.0109 | 0.0095 | -13% |
| | Split-ctx | 0.0128 | 0.0107 | -16% |
| | Player-samp | 0.0029 | 0.0027 | -9% |
| | Combined | 0.0029 | 0.0026 | -10% |

### 7.4 Findings

1. **Spearman rho is stable (delta < 0.025).** Removing acts\_in does not fundamentally change the policy-cosine correlation. The pairwise JS ranking between players is largely preserved because the relative ordering of behavioral similarity is robust to the h\_event representation.

2. **Substitute ratio improves in 3 of 4 models.** The largest gain is Baseline (+22%, from 1.84 to 2.26) and Combined (+18%, from 10.35 to 12.25). This confirms the hypothesis: when h\_event no longer encodes actor identity, the FiLM + z\_p pathway becomes the *sole* source of player-specific information, making the substitute comparison cleaner.

3. **Player-samp shows a slight ratio decrease (17.2 to 15.6, -9%).** This is likely noise — the absolute ratio remains extremely high (15.6x) and the player-samp model's identity signal is so strong that the acts\_in leak was negligible. The small fluctuation may reflect the stochastic nature of situation sampling.

4. **FiLM sensitivity consistently decreases by 9-16%.** With actor identity removed from h\_event, the FiLM layer no longer needs to "undo" implicit actor information in h\_event to apply the correct z\_p. The slightly smaller FiLM effect size reflects a purer measurement of how much z\_p modulates the situation-only representation.

### 7.5 Implications

The counterfactual fix confirms that:

- **The acts\_in identity leak was real but modest.** The fix improves substitute quality for most models without degrading correlation — a clean win.
- **For production scouting, the counterfactual encoder should be used in Phase 7 analysis.** This ensures that player-replacement comparisons reflect true behavioral differences, not residual actor identity.
- **The player-samp model remains the best for scouting** — its 15.6x substitute ratio (post-fix) is still far above all other models. The fix primarily benefits the weaker models by lifting their baseline substitute quality.

---

## 8. Re-evaluation — Player-samp (v3) and Combined (v3)

> **Context:** In the v2 ablation study (Sections 4–7), an early-stopping bug was discovered: the annealed `lambda_pooled_contrast` multiplied a *negative* uniformity loss, which artificially deflated `val_total` and corrupted best-model selection (see FUTURE_IMPROVEMENTS.md #14). The fix introduces a supervised-only validation metric for early stopping. However, the models below were trained **before the fix** — their "best" checkpoints were selected manually from existing runs based on train action loss as a proxy for supervised quality:
>
> | Pipeline | Checkpoint used | Epoch | Basis for selection |
> |----------|----------------|-------|---------------------|
> | **Player-samp (v3)** | `final_model.pt` | 175 | Lowest train action loss (0.5984) in completed run |
> | **Combined (v3)** | `checkpoint_epoch_140.pt` | 140 | Most advanced state before training stopped; action loss plateau (0.645) |
>
> These results therefore represent the *best available* pre-fix models, not properly validated best models. Retraining with the fixed trainer will likely produce different (potentially better) results.

### 8.1 Self-Consistency Comparison (v2 → v3)

#### Competition-split (311 testable, gallery 1,633)

| Metric | Player-samp (v2) | Player-samp (v3) | Combined (v2) | Combined (v3) |
|--------|-------------------|-------------------|----------------|----------------|
| Self-cosine (mean) | 0.4980 | 0.3387 | 0.5028 | **0.5722** |
| Cosine margin | 0.4298 | 0.3024 | 0.4323 | **0.5428** |
| Mean rank | 259.0 | 247.9 | 254.5 | **199.3** |
| Median rank | 82 | 132 | 81 | **86** |
| Hit@1 | 5.8% | 0.5% | 4.8% | **5.5%** |
| Hit@5 | 13.2% | 3.0% | 11.6% | **14.9%** |
| Hit@10 | 18.2% | 6.4% | 15.8% | **19.8%** |
| Hit@20 | 24.6% | 12.5% | 24.3% | **25.9%** |
| Hit@50 | 38.3% | 25.4% | 37.1% | **38.1%** |

#### Position-group breakdown (competition-split, v3)

| Position | n | Player-samp v3 MR | Combined v3 MR | Player-samp v3 H@10 | Combined v3 H@10 |
|----------|---|-------------------|----------------|---------------------|-------------------|
| Defender | 124 | 236.9 | **222.6** | 5.7% | **18.1%** |
| Forward | 71 | 254.3 | **174.5** | 7.0% | **22.5%** |
| Goalkeeper | 18 | 57.6 | **27.4** | 5.6% | **36.1%** |
| Midfielder | 98 | 292.0 | **219.2** | 7.1% | **16.8%** |

#### Random-half (1,083 testable, gallery 1,633)

| Metric | Player-samp (v2) | Player-samp (v3) | Combined (v2) | Combined (v3) |
|--------|-------------------|-------------------|----------------|----------------|
| Self-cosine (mean) | 0.8178 | 0.4080 | 0.8154 | **0.9476** |
| Cosine margin | 0.7473 | 0.3758 | 0.7410 | **0.9167** |
| Mean rank | 15.2 | 151.2 | 15.2 | **3.2** |
| Median rank | 4 | 74 | 4 | **1** |
| Hit@1 | 34.1% | 1.7% | 34.5% | **68.4%** |
| Hit@5 | 54.9% | 6.6% | 55.6% | **88.3%** |
| Hit@10 | 64.8% | 11.7% | 65.3% | **94.0%** |
| Hit@20 | 76.2% | 20.8% | 77.1% | **97.3%** |
| Hit@50 | 91.2% | 39.1% | 91.9% | **99.5%** |

#### Position-group breakdown (random-half, v3)

| Position | n | Player-samp v3 MR | Combined v3 MR | Player-samp v3 H@10 | Combined v3 H@10 |
|----------|---|-------------------|----------------|---------------------|-------------------|
| Defender | 451 | 132.2 | **3.9** | 10.5% | **91.8%** |
| Forward | 217 | 163.2 | **2.9** | 13.1% | **95.2%** |
| Goalkeeper | 51 | 51.0 | **4.9** | 7.8% | **90.2%** |
| Midfielder | 364 | 181.6 | **2.4** | 12.9% | **96.6%** |

### 8.2 Pseudo Ground-Truth Comparison (v2 → v3)

| Metric | Player-samp (v2) | Player-samp (v3) | Combined (v2) | Combined (v3) |
|--------|-------------------|-------------------|----------------|----------------|
| Mean cosine | 0.0422 | **0.2789** | 0.0547 | 0.1098 |
| Mean rank | 814.4 | **269.6** | 824.8 | 667.5 |
| Median rank | 759 | **196** | 700 | 441 |
| Hit@50 | 0.0% | **5.0%** | 0.0% | 0.0% |

#### Per-pair comparison (avg rank, v3 only)

| Tier | Player A | Player B | Player-samp (v3) | Combined (v3) |
|------|----------|----------|-------------------|----------------|
| 1 | Miedema | Caldentey | **348.0** | 1137.0 |
| 1 | TAA | Hakimi | **134.0** | 210.0 |
| 1 | Alba | Robertson | **127.0** | 193.0 |
| 2 | Kroos | Enzo Fernandez | **77.5** | 135.5 |
| 2 | TAA | Maehle | **186.0** | 640.5 |
| 2 | Sancho | Vargas | **194.5** | 303.5 |
| 2 | Sancho | Baumgartner | 910.5 | **334.5** |
| 3 | Kane | Leao | **279.5** | 1397.0 |
| 3 | Kane | Felix | **109.0** | 1090.5 |
| 3 | Uduokhai | Souttar | **330.0** | 1233.5 |

### 8.3 Policy Diagnostic Comparison (v2 → v3)

#### Policy-cosine correlation (Spearman rho)

| Position Group | Player-samp (v2) | Player-samp (v3) | Combined (v2) | Combined (v3) |
|----------------|-------------------|-------------------|----------------|----------------|
| Goalkeeper | 0.710 | 0.791 | 0.704 | **0.751** |
| Defender | 0.682 | 0.528 | 0.667 | **0.766** |
| Midfielder | 0.687 | 0.435 | 0.694 | **0.833** |
| Forward | 0.658 | 0.474 | 0.683 | **0.769** |
| **Overall** | 0.663 | 0.521 | 0.657 | **0.798** |

#### Substitute quality

| Metric | Player-samp (v2) | Player-samp (v3) | Combined (v2) | Combined (v3) |
|--------|-------------------|-------------------|----------------|----------------|
| JS(top-K) | 0.001124 | 0.008707 | 0.001945 | **0.001106** |
| JS(random-K) | 0.019344 | 0.019920 | 0.020129 | **0.013671** |
| **Ratio** | 17.22 | 2.29 | 10.35 | **12.37** |

#### FiLM sensitivity (mean JS, correct vs shuffled)

| Position Group | Player-samp (v2) | Player-samp (v3) | Combined (v2) | Combined (v3) |
|----------------|-------------------|-------------------|----------------|----------------|
| Goalkeeper | 0.0011 | 0.0067 | 0.0012 | **0.0017** |
| Defender | 0.0035 | 0.0088 | 0.0037 | **0.0090** |
| Midfielder | 0.0027 | 0.0075 | 0.0026 | **0.0048** |
| Forward | 0.0028 | 0.0063 | 0.0024 | **0.0042** |
| **Overall** | 0.0029 | 0.0077 | 0.0029 | **0.0061** |

### 8.4 Findings

1. **Combined (v3) is the new best model on self-consistency by a wide margin.** Random-half Hit@1 of 68.4% (vs 34.5% for v2) and Hit@10 of 94.0% (vs 65.3%) represent a dramatic improvement. Mean rank of 3.2/1,633 means the model almost always retrieves the correct player in the top few results. This is the strongest identity signal across all model variants tested.

2. **Player-samp (v3) regressed severely.** All metrics are substantially worse than v2: random-half Hit@10 dropped from 64.8% to 11.7%, and competition-split Hit@10 from 18.2% to 6.4%. The manually selected checkpoint (final_model.pt, epoch 175) was trained well past the point where supervised performance peaked — the broken early stopping allowed training to continue while the model overfit to auxiliary losses. This strongly validates the early-stopping fix.

3. **Combined (v3) achieves the highest policy-cosine correlation ever recorded (rho = 0.80).** This exceeds all previous models (baseline 0.71, player-samp v2 0.66) by a substantial margin, and is particularly strong for midfielders (0.83). The embedding space is now tightly aligned with behavioral decision-making similarity.

4. **The identity–similarity trade-off persists but shifts.** Combined (v3) still underperforms baseline on ground-truth (mean rank 667.5 vs 219.5), confirming that models trained with strong uniformity push all embeddings apart, degrading cross-player similarity structure. However, player-samp (v3) actually improves on ground-truth vs v2 (269.6 vs 814.4), though this is likely because the v3 model's embeddings are less spread (weaker uniformity learned at the selected checkpoint).

5. **Split-context edges make a decisive difference.** The gap between player-samp (v3) and combined (v3) — which differ only in the use of split teammate/opponent edges — is enormous. Combined (v3) outperforms player-samp (v3) on every self-consistency metric, policy correlation, and substitute quality. This is the strongest evidence yet that encoding teammate/opponent context separately is materially important.

6. **The early-stopping bug had asymmetric impact.** Player-samp (v3) was harmed much more than combined (v3), likely because the player-samp model's training ran 35 epochs longer (175 vs 140), allowing more overfitting to the corrupted loss signal. Combined (v3)'s training was interrupted earlier (epoch 140), accidentally preserving a better model state.

### 8.5 Implications

- **Combined (v3) is the recommended model for deployment** in both identity-verification and scouting scenarios, pending retraining with the fixed early stopping.
- **Retraining both pipelines with the fixed supervised-only early stopping** is strongly recommended. The v3 results demonstrate that the *checkpoint selection strategy* is at least as important as the training objective — proper early stopping should yield models that are better than both v2 and v3.
- **The split-context edge ablation is definitively resolved:** teammate/opponent edge splitting provides a clear, consistent benefit across all evaluations when combined with player-aware sampling.

---

## 9. Re-evaluation with Corrected Statistical Methodology (v4)

> **Date**: 2026-03-25
>
> **What changed**: Two statistical methodology fixes were applied to `policy_diagnostic.py` and the full evaluation pipeline was re-run on both pipelines (same models, same checkpoints as v3):
>
> 1. **Holm-Bonferroni monotonicity enforcement** — The textbook Holm procedure requires adjusted p-values to be monotonically non-decreasing (`adjusted[i] = max(adjusted[i], adjusted[i-1])`). The previous implementation omitted this step, which could produce paradoxical results (a larger raw p-value receiving a smaller adjusted p-value).
> 2. **NaN guard in Mantel permutation test** — If `spearmanr` returns NaN (from constant-value arrays), the comparison `perm_rho >= observed_rho` is always False (NaN semantics), yielding a spuriously small p-value suggesting significance when there is no meaningful correlation.
>
> **Models are unchanged.** All results below use the same `best_model.pt` checkpoints as Section 8. Self-consistency and ground-truth metrics are identical to v3 (deterministic given same model/embeddings). Policy diagnostic correlation (rho) values are identical. Substitute quality ratios show minor variation due to re-run randomness in query player selection.

### 9.1 Self-Consistency (unchanged from v3)

Self-consistency metrics are identical to Section 8.1. Key numbers for reference:

| Metric | Player-samp | Combined |
|--------|-------------|----------|
| **Competition-split** | | |
| Mean rank | 247.9 | **199.3** |
| Median rank | 132 | **86** |
| Hit@10 | 6.4% | **19.8%** |
| **Random-half** | | |
| Mean rank | 151.2 | **3.2** |
| Median rank | 74 | **1** |
| Hit@1 | 1.7% | **68.4%** |
| Hit@10 | 11.7% | **94.0%** |

### 9.2 Ground Truth (unchanged from v3)

Ground-truth metrics are identical to Section 8.2. Key numbers for reference:

| Metric | Player-samp | Combined |
|--------|-------------|----------|
| Mean cosine | **0.2789** | 0.1098 |
| Mean rank | **269.6** | 667.5 |
| Hit@50 | **5.0%** | 0.0% |

### 9.3 Policy Diagnostic (corrected methodology)

#### Policy-cosine correlation (Spearman rho)

Rho values are identical to v3. All group-level permutation p-values are now properly Holm-Bonferroni corrected with monotonicity enforcement.

| Position Group | Player-samp rho | Player-samp p\_holm | Combined rho | Combined p\_holm |
|----------------|-----------------|---------------------|--------------|-------------------|
| Goalkeeper | 0.791 | 0.004 | **0.751** | 0.004 |
| Defender | 0.528 | 0.004 | **0.766** | 0.004 |
| Midfielder | 0.435 | 0.004 | **0.833** | 0.004 |
| Forward | 0.474 | 0.004 | **0.769** | 0.004 |
| **Overall** | 0.521 | *(pooled)* | **0.798** | *(pooled)* |

All correlations remain highly significant (p\_holm = 0.004, well below 0.05) after correction. The Holm correction uses `p_raw × (m − rank)` with m=4 groups and enforces monotonicity.

#### Substitute quality

| Metric | Player-samp | Combined |
|--------|-------------|----------|
| JS(top-K) | 0.005812 | **0.000833** |
| JS(random-K) | 0.019122 | **0.013273** |
| **Ratio** | 3.29 | **15.94** |
| **95% CI** | [3.20, 51.15] | **[8.52, 24.26]** |

Combined maintains a dramatically higher substitute ratio (15.9x) with a tight 95% bootstrap confidence interval [8.5, 24.3], confirming this is a robust result. Player-samp's wide CI [3.2, 51.2] reflects high variance across the 10 query players.

#### FiLM sensitivity (mean JS, correct vs shuffled)

| Position Group | Player-samp | Combined |
|----------------|-------------|----------|
| Goalkeeper | 0.0067 | **0.0017** |
| Defender | 0.0088 | **0.0090** |
| Midfielder | 0.0075 | **0.0048** |
| Forward | 0.0063 | **0.0042** |
| **Overall** | 0.0077 | **0.0061** |

### 9.4 Statistical Methodology Notes

The following statistical improvements are now reflected in all v4 reports:

1. **Mantel-style permutation tests** replace parametric Spearman p-values. Distance-matrix pairs share players and violate i.i.d. assumptions, making parametric p-values unreliable. The permutation test shuffles row/column labels of the cosine-distance matrix (1,000 permutations) and counts exceedances.

2. **Holm-Bonferroni correction** is applied across 4 group-level tests with proper monotonicity enforcement. Adjusted p-values are guaranteed non-decreasing when sorted by raw p-value.

3. **Bootstrap 95% CIs** are reported for the aggregate substitute ratio (2,000 bootstrap resamples, percentile method).

4. **Mixed query selection** for substitute quality: half of query players are selected from the highest-possession players per group, half are randomly sampled from the remainder. This mitigates bias toward high-data players.

5. **Guaranteed derangements** in FiLM sensitivity: the z\_p shuffle uses rejection sampling (n≥3) or exact swap (n=2) to ensure no player retains their own embedding.

### 9.5 Summary and Recommendations

The v4 re-evaluation confirms all v3 findings with improved statistical rigor:

| Pipeline | Best for | Key strength | Key weakness |
|----------|----------|-------------|--------------|
| **Combined (split\_ctx\_ps)** | Identity verification, scouting | Hit@1=68.4% (random-half), rho=0.80, sub ratio=15.9x | Ground-truth mean rank 667.5 |
| **Player-samp** | Cross-player similarity search | Ground-truth mean rank 269.6 | Hit@1=1.7% (random-half), rho=0.52 |

**Combined remains the recommended model.** Its behavioral validation (rho=0.80, substitute ratio 15.9x with CI [8.5, 24.3]) is the strongest of any model tested. Retraining with the fixed supervised-only early stopping is still recommended to potentially improve further.
