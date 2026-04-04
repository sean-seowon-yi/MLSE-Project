# Evaluation Results

This document presents a critical, metric-by-metric analysis of every model evaluated under `evaluations/`. All numbers are extracted directly from the JSON result files; every claim is verifiable against the raw data.

## 1. Models Evaluated

**GNN models (11 evaluation folders):** 10 named pipelines in `main.py` `PIPELINE_REGISTRY` that currently have runs under `evaluations/`, plus 1 archived run (`pos_ablated_split_ctx_ema_v2`) kept for comparison. The registry also lists `acts_in_dropout_pos` (position-regularized acts-in dropout); there is **no** `evaluations/acts_in_dropout_pos/` folder yet, so that preset is omitted from the tables below. (The registry has 11 entries total.)

| Tag | Position ablation | Split-context edges | Player sampling | Uniformity t | Lambda pooled | EMA alignment | Lambda alignment | Extra |
|-----|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---|
| `baseline` | No | No | No | 2.0 | 0.3 | No | - | — |
| `player_samp` | No | No | Yes | 2.0 | 0.5 | No | - | — |
| `split_ctx` | No | Yes | No | 2.0 | 0.3 | No | - | — |
| `split_ctx_ps` | No | Yes | Yes | 2.0 | 0.5 | No | - | — |
| `pos_ablated` | Yes | No | No | 2.0 | 0.3 | No | - | — |
| `pos_ablated_split_ctx` | Yes | Yes | No | 4.0 | 1.0 | No | - | — |
| `pos_ablated_split_ctx_v2` | Yes | Yes | No | 4.0 | 0.7 | No | - | — |
| `pos_ablated_split_ctx_ema` | Yes | Yes | No | 4.0 | 1.0 | Yes | 0.3 | — |
| `pos_ablated_split_ctx_ema_v2` | Yes | Yes | No | 4.0 | 1.0 | Yes | 0.1 | Archived (not in registry) |
| `acts_in_dropout` | Yes | Yes | No | 4.0 | 1.0 | No | - | `acts_in_dropout=0.3` |
| `acts_in_dropout_pos` | Yes | Yes | No | 4.0 | 1.0 | No | - | p=0.3, `lambda_pos=0.3` (no eval run yet) |
| `acts_in_dropout_pos_gu` | Yes | Yes | No | 4.0 | 1.0 | No | - | p=0.3, `lambda_pos=0.3`, `uniformity_group_weight=3.0` |

`pos_ablated_split_ctx_ema_v2` is included in this report from `evaluations/` but is not in `PIPELINE_REGISTRY`.

**Heuristic baselines (3):** non-learned methods, no self-consistency/policy/test metrics available.

| Tag | Method |
|-----|--------|
| `h_mean_features` | Mean of raw StatsBomb features per player |
| `h_action_profile` | Action-type distribution histogram per player |
| `h_fifa_attributes` | FIFA video-game attribute vectors |

---

## 2. Pseudo-Ground-Truth Pair Retrieval

15 LLM-generated similar-player pairs (5 tier-1, 7 tier-2, 3 tier-3) across men's and women's football, sourced from publicly cited analytics comparisons and compiled by an LLM (see `docs/pseudo_ground_truth.md`). These are **directional sanity checks, not expert-validated ground truth** — the pairs are reasonable (e.g., Modrić–Kroos, VVD–Dias) but carry LLM biases toward fame, media narratives, and positional similarity. Disagreement with these pairs does not necessarily indicate a model flaw. Gallery size: 984 male / 647 female. Lower rank = better. All GNN models use 1,633 embedded players; heuristics use 1,965 (h_mean_features, h_action_profile) or 1,450 (h_fifa_attributes).

### 2.1 Overall Summary

| Model | Mean Rank | Median Rank | Hit@5 | Hit@10 | Hit@20 | Hit@50 |
|-------|----------:|----------:|------:|-------:|-------:|-------:|
| split_ctx | **56.1** | 40 | 0.100 | 0.233 | 0.400 | 0.600 |
| pos_ablated_split_ctx_ema | 62.6 | 37 | 0.133 | 0.233 | 0.400 | **0.700** |
| pos_ablated_split_ctx | 63.1 | 39 | 0.200 | **0.267** | 0.300 | 0.633 |
| pos_ablated_split_ctx_ema_v2 | 64.3 | 42 | 0.167 | **0.267** | 0.333 | 0.633 |
| acts_in_dropout_pos_gu | 66.7 | 41 | 0.200 | 0.233 | 0.367 | 0.567 |
| baseline | 67.6 | 30 | 0.167 | 0.233 | **0.433** | 0.667 |
| acts_in_dropout | 68.2 | 32 | 0.167 | 0.267 | 0.400 | 0.633 |
| pos_ablated | 69.6 | 30 | 0.200 | 0.233 | 0.333 | 0.567 |
| player_samp | 70.2 | 71 | 0.167 | 0.200 | 0.233 | 0.333 |
| split_ctx_ps | 71.4 | 73 | 0.133 | 0.167 | 0.267 | 0.333 |
| h_mean_features | 108.7 | 35 | 0.200 | 0.267 | 0.267 | 0.633 |
| h_fifa_attributes | 130.0 | 73 | 0.200 | 0.267 | 0.300 | 0.400 |
| pos_ablated_split_ctx_v2 | 198.8 | 161 | 0.033 | 0.067 | 0.100 | 0.200 |
| h_action_profile | 287.2 | 61 | 0.100 | 0.133 | 0.300 | 0.467 |

**Key observations:**

- `split_ctx` has the lowest mean rank (56.1), but its confidence interval [35.8, 83.3] overlaps heavily with the `pos_ablated_split_ctx` family ([36.6, 92.6] for ema; [38.0, 91.7] for base). No model is statistically significantly better than another at the top.
- `pos_ablated_split_ctx_ema` has the best hit@50 (0.700) meaning 70% of known pairs appear within the top 50 neighbors.
- `acts_in_dropout_pos_gu` achieves the **third-best** mean rank (66.7) among GNN models—between `pos_ablated_split_ctx_ema_v2` (64.3) and `baseline` (67.6)—but does not beat `pos_ablated_split_ctx_ema` on hit@50 (0.567 vs 0.700).
- Player-sampling models (`player_samp`, `split_ctx_ps`) have conspicuously high median ranks (71, 73) despite moderate mean ranks, indicating retrieval quality is inconsistent, with a few easy pairs performing well but most pairs ranked poorly.
- `pos_ablated_split_ctx_v2` (lambda_pooled=0.7, uniformity_t=4.0) collapses entirely, confirming that even a modest reduction in uniformity weight is catastrophic for the pos_ablated architecture.
- The `baseline` model (mean_rank 67.6, median 30) is surprisingly competitive with more complex models, outperforming `player_samp` and `split_ctx_ps` on median rank.

### 2.2 Tier-1 Pairs (Highest-Confidence Substitutes)

These 5 pairs (Modric-Kroos, VVD-Dias, Alba-Robertson, TAA-Hakimi, Bonmati-Putellas) represent the strongest known substitute relationships.

| Model | Tier-1 Mean Rank | Tier-1 Hit@10 |
|-------|------------------:|--------------:|
| split_ctx | 50.2 | 0.000 |
| baseline | 57.2 | 0.200 |
| pos_ablated_split_ctx | 89.5 | 0.200 |
| pos_ablated | 93.6 | 0.000 |
| pos_ablated_split_ctx_ema | 97.3 | 0.200 |
| pos_ablated_split_ctx_ema_v2 | 97.3 | 0.200 |
| acts_in_dropout_pos_gu | 102.5 | 0.200 |
| acts_in_dropout | 107.3 | 0.200 |
| player_samp | 115.3 | **0.000** |
| split_ctx_ps | 116.9 | **0.000** |

**Critical finding:** Both player-sampling models score 0.0 hit@10 on tier-1 pairs. They cannot place any of the five most obvious substitutes within the top 10. The `baseline` model outperforms them on tier-1 pairs. `split_ctx` achieves the best tier-1 mean rank (50.2) but also has 0.0 hit@10, indicating its advantages come from moderate ranks across the board rather than top-10 placement. The acts-in dropout runs reach 0.2 hit@10 on tier-1 but at **worse** mean ranks (102–107) than the `pos_ablated_split_ctx` family.

### 2.3 Pair-Level Detail for Key Pairs

Selected pair-level ranks (format: rank of B in A's neighbors / rank of A in B's neighbors):

| Pair | pos_ablated_split_ctx | ema | ema_v2 | acts_in_dropout | player_samp |
|------|-----:|-----:|-----:|-----:|-----:|
| VVD - Dias | 6/8 | 8/8 | 10/9 | 7/10 | 57/91 |
| Alba - Robertson | 33/41 | 37/33 | 41/48 | 40/26 | 86/112 |
| TAA - Hakimi | 33/99 | 18/97 | 15/92 | 15/92 | 128/81 |
| Saka - Dembele | 3/4 | 2/3 | 1/2 | 2/4 | 1/2 |
| Kane - Lewandowski | 27/40 | 17/17 | 32/38 | 22/18 | 68/95 |
| Kroos - Enzo | 20/62 | 11/25 | 15/56 | 18/51 | 88/71 |
| Musiala - Foden | 3/2 | 12/4 | 8/4 | 7/3 | 2/1 |
| Bellingham - Griezmann | 69/59 | 57/44 | 44/53 | 65/65 | 15/25 |

The `pos_ablated_split_ctx` family consistently ranks pairs in the 10-50 range, while `player_samp` scatters them broadly (ranks 1-2 for easy cases like Saka-Dembele, but 68-128 for harder pairs like Kane-Lewandowski or TAA-Hakimi). The `ema` variant achieves the best ranks on Kane-Lewandowski (17/17) and Kroos-Enzo (11/25). `acts_in_dropout` is broadly aligned with the split-ctx family on the pairs above (same orders of magnitude as `pos_ablated_split_ctx` / `ema`).

---

## 3. Self-Consistency

Two evaluation protocols: **competition_split** (same player across different competitions -- harder, more realistic) and **random_half** (same player, random half of possessions from same data -- easier). 311 players with multi-competition data; 1,083 players with enough possessions for random splits.

### 3.1 Competition Split (Cross-Context Stability)

| Model | Self-Cosine Mean | Mean Rank | Hit@10 |
|-------|--:|--:|--:|
| pos_ablated_split_ctx | 0.766 | **67.3** | 0.346 |
| pos_ablated_split_ctx_ema_v2 | 0.767 | **67.5** | **0.349** |
| pos_ablated_split_ctx_ema | 0.769 | 67.7 | 0.318 |
| acts_in_dropout | 0.763 | 68.9 | **0.351** |
| acts_in_dropout_pos_gu | 0.746 | 74.0 | 0.328 |
| pos_ablated | 0.974 | 81.3 | 0.275 |
| baseline | 0.885 | 81.6 | 0.238 |
| split_ctx | 0.859 | 83.9 | 0.215 |
| split_ctx_ps | 0.573 | 89.3 | 0.262 |
| player_samp | 0.569 | 93.5 | 0.264 |
| pos_ablated_split_ctx_v2 | 0.325 | 200.8 | 0.076 |

**Key observations:**

- The `pos_ablated_split_ctx` family dominates competition-split mean rank (67.3-67.7) and hit@10 (0.318-0.349), despite having lower self-cosine values than `pos_ablated` (0.974) or `baseline` (0.885). This means their embeddings are more spread out (lower raw cosine) but the *relative* ordering is far more accurate -- the same player's embedding from different competitions is found closer to themselves in terms of rank.
- `acts_in_dropout` reaches the **best hit@10** in this suite (0.351) with mean rank 68.9—slightly behind the split-ctx family on mean rank but competitive. `acts_in_dropout_pos_gu` trades some of that for stronger group-uniformity regularization: mean rank 74.0, hit@10 0.328.
- High self-cosine with poor rank (e.g., `pos_ablated`: cosine 0.974 but mean_rank 81.3) indicates embedding compression. All players look similar to each other, so even a high cosine to yourself doesn't guarantee a low rank.
- `player_samp` has the worst competition-split mean rank (93.5) among non-collapsed models. Cross-context stability is a major weakness.
- `split_ctx_ps` performs similarly poorly (89.3), suggesting player sampling degrades cross-competition generalization regardless of edge architecture.

### 3.2 Random-Half Split (Within-Context Stability)

| Model | Self-Cosine Mean | Mean Rank | Hit@10 |
|-------|--:|--:|--:|
| split_ctx_ps | 0.926 | **2.7** | **0.951** |
| player_samp | 0.931 | 2.9 | 0.940 |
| pos_ablated_split_ctx_ema_v2 | 0.900 | 6.1 | 0.857 |
| pos_ablated_split_ctx | 0.896 | 6.3 | 0.842 |
| acts_in_dropout_pos_gu | 0.880 | 8.2 | 0.796 |
| acts_in_dropout | 0.882 | 8.7 | 0.785 |
| pos_ablated_split_ctx_ema | 0.885 | 9.3 | 0.781 |
| pos_ablated | 0.989 | 13.9 | 0.697 |
| baseline | 0.948 | 26.1 | 0.510 |
| split_ctx | 0.913 | 36.4 | 0.395 |
| pos_ablated_split_ctx_v2 | 0.332 | 177.8 | 0.094 |

**Key observations:**

- Player-sampling models excel here (mean_rank 2.7-2.9, hit@10 0.94-0.95) because the sampling mechanism explicitly trains the model to produce identical embeddings for the same player across different possession subsets. This is a direct consequence of the training objective rather than emergent generalization.
- The `pos_ablated_split_ctx` family achieves strong random-half consistency (mean_rank 6.1-9.3) without player sampling, demonstrating that position ablation + split-context edges + strong uniformity produces naturally stable representations.
- `acts_in_dropout` and `acts_in_dropout_pos_gu` sit between the base `pos_ablated_split_ctx` and the EMA variant on random-half mean rank (8.2-8.7 vs 6.3-9.3), consistent with dropout noise on the action stream slightly loosening within-context tightness.
- `pos_ablated_split_ctx_ema` (mean_rank 9.3) is slightly worse than its non-EMA counterpart (6.3), likely because the alignment loss regularizer trades some within-context tightness for improved behavioral fidelity.

### 3.3 Competition vs Random-Half Gap

The gap between competition-split and random-half performance reveals how well a model generalizes across contexts:

| Model | Comp Mean Rank | Random Mean Rank | Gap |
|-------|--:|--:|--:|
| pos_ablated_split_ctx | 67.3 | 6.3 | 61.0 |
| pos_ablated_split_ctx_ema | 67.7 | 9.3 | 58.4 |
| pos_ablated_split_ctx_ema_v2 | 67.5 | 6.1 | 61.4 |
| acts_in_dropout | 68.9 | 8.7 | 60.2 |
| acts_in_dropout_pos_gu | 74.0 | 8.2 | 65.8 |
| baseline | 81.6 | 26.1 | 55.5 |
| pos_ablated | 81.3 | 13.9 | 67.4 |
| split_ctx | 83.9 | 36.4 | 47.5 |
| split_ctx_ps | 89.3 | 2.7 | **86.6** |
| player_samp | 93.5 | 2.9 | **90.6** |

Player-sampling models have the largest gap (86-91 ranks), meaning their strong within-context consistency does not transfer across competitions. The `pos_ablated_split_ctx` family and `acts_in_dropout` variants have moderate gaps (~58-66) with much better absolute competition-split performance than player-sampling models.

---

## 4. Policy Diagnostic

Behavioral fidelity evaluation: does embedding proximity reflect actual behavioral similarity? Measured via Spearman rho between embedding cosine distance and Jensen-Shannon divergence of action distributions, computed within position-gender groups (8 groups). Also: substitute quality ratio (how much better are top-10 neighbors than random neighbors, in JS divergence terms). Not available for heuristic baselines.

### 4.1 Spearman Rho (Behavioral Ordering Fidelity)

| Model | Overall Rho | GK(f) | GK(m) | Def(f) | Def(m) | Mid(f) | Mid(m) | Fwd(f) | Fwd(m) |
|-------|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| pos_ablated | **0.974** | **0.786** | 0.660 | **0.980** | **0.984** | **0.950** | **0.955** | **0.910** | **0.920** |
| pos_ablated_split_ctx_ema | 0.944 | 0.722 | 0.604 | 0.939 | 0.946 | 0.909 | 0.898 | 0.849 | 0.879 |
| pos_ablated_split_ctx_ema_v2 | 0.934 | 0.612 | 0.655 | 0.940 | 0.951 | 0.901 | 0.879 | 0.834 | 0.839 |
| pos_ablated_split_ctx | 0.929 | 0.550 | 0.547 | 0.926 | 0.942 | 0.898 | 0.873 | 0.839 | 0.843 |
| acts_in_dropout | 0.928 | 0.629 | 0.591 | 0.912 | 0.924 | 0.895 | 0.875 | 0.843 | 0.829 |
| acts_in_dropout_pos_gu | 0.922 | 0.699 | 0.615 | 0.885 | 0.904 | 0.884 | 0.852 | 0.847 | 0.835 |
| player_samp | 0.788 | 0.768 | 0.697 | 0.781 | 0.790 | 0.792 | 0.804 | 0.745 | 0.732 |
| split_ctx_ps | 0.778 | 0.759 | 0.730 | 0.693 | 0.704 | 0.799 | 0.801 | 0.728 | 0.721 |
| baseline | 0.744 | 0.750 | 0.682 | 0.763 | 0.807 | 0.768 | 0.745 | 0.690 | 0.701 |
| split_ctx | 0.710 | 0.773 | 0.643 | 0.741 | 0.745 | 0.747 | 0.742 | 0.609 | 0.627 |
| pos_ablated_split_ctx_v2 | 0.686 | 0.571 | 0.734 | 0.685 | 0.722 | 0.619 | 0.602 | 0.596 | 0.544 |

**Key observations:**

- `pos_ablated` achieves exceptionally high rho (0.974) because its compressed embedding space (very high cosine between players) coincidentally aligns well with behavioral distance ordering. However, this comes at the cost of practical retrieval (GT mean_rank 69.6, competition SC mean_rank 81.3).
- Among models with good retrieval quality, `pos_ablated_split_ctx_ema` has the highest rho (0.944), followed by `ema_v2` (0.934), the base variant (0.929), then `acts_in_dropout` (0.928) and `acts_in_dropout_pos_gu` (0.922). Acts-in dropout sits between the plain split-ctx stack and player-sampling models on overall rho.
- All position groups except goalkeepers show rho > 0.83 for the `pos_ablated_split_ctx` family. Goalkeepers are weaker (0.55-0.72), likely because keeper actions are less diverse and harder to differentiate.
- `player_samp` (0.788) and `split_ctx_ps` (0.778) show substantially lower behavioral fidelity than the `pos_ablated_split_ctx` family, despite having higher substitute quality ratios (see below).

### 4.2 Substitute Quality Ratio

| Model | Top-k JS | Random-k JS | Ratio | 95% CI |
|-------|--:|--:|--:|---|
| split_ctx_ps | **0.001025** | 0.013442 | **13.11** | [6.73, 29.96] |
| player_samp | 0.001122 | 0.013367 | 11.91 | [5.84, 29.60] |
| pos_ablated | 0.001110 | 0.009535 | 8.59 | - |
| pos_ablated_split_ctx_ema | 0.001040 | 0.008265 | 7.95 | [4.23, 16.34] |
| acts_in_dropout | 0.001031 | 0.007990 | 7.75 | [4.29, 13.15] |
| acts_in_dropout_pos_gu | 0.001051 | 0.008015 | 7.62 | [3.84, 17.33] |
| pos_ablated_split_ctx_ema_v2 | 0.001453 | 0.009228 | 6.35 | [3.43, 12.50] |
| pos_ablated_split_ctx | 0.001408 | 0.009332 | 6.63 | [3.83, 10.97] |
| baseline | 0.003298 | 0.012652 | 3.84 | - |
| split_ctx | 0.004611 | 0.014442 | 3.13 | - |
| pos_ablated_split_ctx_v2 | 0.006962 | 0.021340 | 3.07 | - |

**Critical interpretation:**

The ratio is random-k JS / top-k JS. A high ratio can be achieved either by genuinely better top-k neighbors OR by a more spread-out embedding space that inflates the random-k denominator. The confidence intervals are wide and overlapping for all models above 6.0.

**Absolute top-k JS** (the actual behavioral similarity of retrieved neighbors) is more diagnostic:
- `split_ctx_ps` (0.001025) and `pos_ablated_split_ctx_ema` (0.001040) produce virtually identical top-k neighbor quality.
- `acts_in_dropout` (0.001031) matches that band; `acts_in_dropout_pos_gu` (0.001051) is marginally higher.
- `player_samp` (0.001122) is only marginally different.
- The high ratios for `player_samp` and `split_ctx_ps` come from higher random-k JS (0.01337-0.01344 vs 0.00827-0.00933), reflecting wider embedding spread, not genuinely superior retrieval.

Given the overlapping CIs and near-identical absolute top-k JS values, the ratio differences among the top models are not practically meaningful.

### 4.3 Empirical Behavioral Fidelity (Observed Actions)

The metrics above (§4.1, §4.2) use the model's own action-prediction head to compute behavioral similarity. This creates a self-referential loop: the model's predictions are shaped by the same training signal that shaped the embeddings. **Empirical behavioral fidelity** replaces model predictions with **actual observed player actions** from the event data, providing an independent check.

For each player, events are bucketed by coarse game state (pitch third × under-pressure = 6 buckets). Within each bucket, an empirical action-type histogram is built over 5 coarse categories (Pass, Carry, Shot, Dribble, Other). Pairwise JS divergence between players' empirical histograms (averaged over shared valid buckets, minimum 3 shared with ≥15 events each) produces a behavioral distance that is entirely external to the model. Spearman ρ then correlates embedding cosine distance with this empirical behavioral distance. Additionally, a substitute quality ratio compares the observed-action similarity of embedding neighbors vs. random same-group peers.

**Key difference from §4.1–4.2:** These metrics cannot be inflated by the model learning to predict its own training signal. They measure whether embedding geometry genuinely aligns with how players actually behave in matched situations.

**Coverage:** 10 of 11 GNN models (`pos_ablated_split_ctx_ema_v2` has no empirical behavioral evaluation). 1,633 embedded players, 183,580 valid pairs overall.

### 4.4 Empirical Behavioral Spearman Rho

| Model | Overall EB Rho | Def(f) | Def(m) | Mid(f) | Mid(m) | Fwd(f) | Fwd(m) |
|-------|--:|--:|--:|--:|--:|--:|--:|
| split_ctx_ps | **0.2296** | 0.2858 | **0.3078** | **0.1664** | 0.1667 | 0.1694 | **0.2715** |
| player_samp | 0.2230 | **0.2824** | 0.2986 | 0.1535 | 0.1672 | 0.1705 | 0.2678 |
| pos_ablated_split_ctx_ema | 0.2074 | 0.2455 | 0.2957 | **0.2498** | **0.2497** | 0.1835 | 0.2489 |
| pos_ablated_split_ctx | 0.2070 | 0.2291 | 0.2755 | 0.2395 | 0.2572 | 0.1904 | 0.2645 |
| acts_in_dropout | 0.2066 | 0.2366 | 0.2849 | 0.2454 | 0.2433 | 0.1909 | 0.2520 |
| pos_ablated | 0.1961 | 0.2719 | 0.2991 | 0.2363 | 0.2772 | **0.2138** | 0.2679 |
| acts_in_dropout_pos_gu | 0.1946 | 0.2280 | 0.2648 | 0.2492 | 0.2332 | 0.1847 | 0.2401 |
| split_ctx | 0.1891 | 0.2810 | **0.3628** | 0.1448 | 0.1666 | 0.0470 | 0.1228 |
| baseline | 0.1879 | 0.2794 | 0.3621 | 0.1559 | 0.1449 | 0.0583 | 0.1224 |
| pos_ablated_split_ctx_v2 | 0.1038 | 0.1773 | 0.1823 | 0.0549 | 0.0298 | 0.0913 | 0.0755 |

**Key observations:**

- EB Rho values (0.10–0.23) are dramatically lower than model-based Rho (0.69–0.97). This reflects the difference between comparing against smooth model predictions vs. noisy empirical histograms with sparse event counts per bucket. The absolute scale is not directly comparable.
- **The ranking is inverted from model-based Rho.** `split_ctx_ps` and `player_samp` lead on EB Rho (0.23, 0.22) despite having the lowest model-based Rho. `pos_ablated` (model-based Rho champion at 0.974) drops to mid-pack (0.196). This inversion occurs because EB Rho rewards models with wider embedding spread (larger cosine distance variance), which makes rank correlation with behavioral distance easier to detect. Compressed embeddings (like `pos_ablated`) have very little cosine distance variance, suppressing EB Rho even if their relative ordering is reasonable.
- The `pos_ablated_split_ctx` family (0.207) and `acts_in_dropout` (0.207) are tightly clustered in the middle, with `acts_in_dropout_pos_gu` slightly behind (0.195).
- `split_ctx` and `baseline` show a distinctive pattern: strong EB Rho for defenders (0.28–0.36) but very weak for forwards (0.05–0.12), mirroring the model-based pattern. Forwards' action distributions are more variable and harder to predict from coarse game-state buckets.
- All overall EB Rho values exceed 0.10 except the collapsed `pos_ablated_split_ctx_v2`, confirming that even this coarse empirical measure detects real behavioral structure in the embedding space.

### 4.5 Empirical Behavioral Substitute Quality

| Model | EB Top-K JS | EB Random-K JS | EB Ratio | EB 95% CI |
|-------|--:|--:|--:|---|
| acts_in_dropout | **0.024620** | 0.048087 | **1.953** | [1.51, 2.61] |
| acts_in_dropout_pos_gu | 0.025507 | 0.044515 | 1.745 | [1.45, 2.37] |
| pos_ablated_split_ctx_ema | 0.025242 | 0.043373 | 1.718 | [1.40, 2.39] |
| split_ctx_ps | 0.027256 | 0.046284 | 1.698 | [1.35, 2.20] |
| pos_ablated | 0.027385 | 0.045397 | 1.658 | [1.18, 2.43] |
| pos_ablated_split_ctx | 0.025986 | 0.042180 | 1.623 | [1.25, 2.83] |
| player_samp | 0.027478 | 0.042914 | 1.562 | [1.19, 2.16] |
| baseline | 0.032515 | 0.042970 | 1.322 | [1.18, 1.82] |
| split_ctx | 0.035682 | 0.044504 | 1.247 | [1.17, 1.76] |
| pos_ablated_split_ctx_v2 | 0.037828 | 0.045822 | 1.211 | [1.04, 1.56] |

9 query players (3 defenders, 3 midfielders, 3 forwards), K=10.

**Key observations:**

- **`acts_in_dropout` achieves the highest empirical substitute ratio (1.95)**, meaning its embedding neighbors are nearly twice as behaviorally similar as random same-group peers in terms of observed actions. This is the strongest model on the most externally grounded substitute-quality metric.
- The `acts_in_dropout` family and `pos_ablated_split_ctx_ema` form a strong cluster (ratio 1.72–1.95) well separated from the bottom three (`baseline` 1.32, `split_ctx` 1.25, `pos_ablated_split_ctx_v2` 1.21).
- Absolute EB top-K JS values are 10–30× higher than model-based top-K JS (0.025 vs 0.001). This is expected: model-based JS uses the model's own smooth predicted distributions, while empirical JS uses noisy histograms from limited data. The absolute values are not comparable across methods.
- `split_ctx_ps` and `player_samp` (ratio 1.70, 1.56) perform moderately despite leading on EB Rho, suggesting their high EB Rho partially reflects embedding spread rather than genuinely superior behavioral discrimination.
- All 95% CIs exclude 1.0 except `pos_ablated_split_ctx_v2` (lower bound 1.04, barely above 1). Every non-collapsed model produces neighbors that are empirically better than random, validating the embedding approach.

---

## 5. Supervised Task Quality (Test Metrics)

Action prediction on held-out test data. This measures representation learning quality as a proxy task.

| Model | Action F1 | Angle F1 | Length F1 | F1 Avg | Shot AUC | Goal AUC |
|-------|--:|--:|--:|--:|--:|--:|
| split_ctx_ps | 0.735 | 0.639 | **0.691** | 0.688 | 0.966 | 0.972 |
| player_samp | **0.733** | **0.677** | 0.690 | **0.700** | 0.961 | 0.977 |
| baseline | 0.728 | 0.657 | 0.690 | 0.692 | 0.972 | **0.989** |
| split_ctx | 0.727 | 0.639 | 0.687 | 0.684 | 0.973 | 0.989 |
| pos_ablated | 0.725 | 0.654 | 0.681 | 0.687 | 0.973 | 0.986 |
| pos_ablated_split_ctx_ema_v2 | 0.726 | 0.624 | 0.671 | 0.674 | 0.974 | 0.987 |
| pos_ablated_split_ctx | 0.724 | 0.627 | 0.675 | 0.675 | **0.976** | 0.985 |
| acts_in_dropout_pos_gu | 0.713 | 0.609 | 0.657 | 0.660 | 0.975 | 0.988 |
| acts_in_dropout | 0.709 | 0.614 | 0.648 | 0.657 | 0.973 | 0.986 |
| pos_ablated_split_ctx_ema | 0.707 | 0.605 | 0.638 | 0.650 | 0.973 | 0.988 |
| pos_ablated_split_ctx_v2 | 0.696 | 0.593 | 0.630 | 0.639 | 0.974 | 0.987 |

**Key observations:**

- `player_samp` has the best average F1 (0.700), primarily driven by its angle prediction advantage (0.677 vs next-best 0.657).
- The `pos_ablated_split_ctx` family shows lower F1 scores (0.650-0.675) than the baseline (0.692). This is expected: position ablation removes an informative feature, and the strong uniformity loss (lambda=1.0, t=4.0) penalizes the supervised objective to achieve better embedding geometry.
- `acts_in_dropout` and `acts_in_dropout_pos_gu` land between the baseline and the strong-uniformity split-ctx runs on F1 avg (0.657-0.660), reflecting stochastic masking of the action stream during training.
- `pos_ablated_split_ctx_ema` (lambda_alignment=0.3) has the worst F1 among non-collapsed split-ctx runs (0.650), confirming the alignment loss further reduces supervised performance. The v2 variant (lambda_alignment=0.1) recovers to 0.674.
- Outcome prediction (shot/goal AUC) is uniformly excellent across all models (0.961-0.976 for shots, 0.972-0.989 for goals), suggesting the outcome heads are less sensitive to architectural changes.
- The F1 differences across models are small in absolute terms (range 0.639-0.700, spread of 0.061). The supervised task is not the primary objective.

---

## 6. FIFA Comparison

External validation against FIFA video-game ratings. 16 query-neighbor pairs per model, comparing FIFA overall ratings and main-6 attribute differences. Position match % indicates how often the top neighbor shares the same broad position group.

| Model | Avg Cosine Sim | Avg Overall Diff | Avg Main-6 Diff | Pos Match % |
|-------|--:|--:|--:|--:|
| h_fifa_attributes | 0.994 | **5.0** | **5.0** | **75** |
| h_mean_features | 0.994 | 6.3 | 10.5 | 50 |
| h_action_profile | 0.988 | **4.1** | 8.2 | 31 |
| pos_ablated | 0.994 | 5.8 | 8.6 | 50 |
| baseline | 0.978 | 6.1 | 8.6 | 44 |
| acts_in_dropout | 0.917 | 5.6 | 7.9 | 50 |
| acts_in_dropout_pos_gu | 0.913 | 6.0 | 7.7 | 50 |
| split_ctx | 0.958 | 7.3 | 8.5 | 56 |
| pos_ablated_split_ctx_ema | 0.916 | 7.4 | 9.1 | 56 |
| pos_ablated_split_ctx_ema_v2 | 0.915 | 7.6 | 8.8 | 50 |
| pos_ablated_split_ctx | 0.913 | 7.6 | 9.0 | 50 |
| player_samp | 0.904 | 5.8 | 8.9 | 50 |
| split_ctx_ps | 0.903 | 7.2 | 9.2 | 25 |
| pos_ablated_split_ctx_v2 | 0.709 | 5.1 | 9.8 | 38 |

**Key observations:**

- `h_fifa_attributes` trivially achieves the best main-6 diff (5.0) and position match (75%) because it uses FIFA attributes directly. This baseline exists to calibrate expectations.
- Models with compressed embeddings (`pos_ablated`, `baseline`) show high avg cosine similarity (0.978-0.994) but this reflects that all players look similar, not that neighbors are genuinely better.
- `acts_in_dropout` and `acts_in_dropout_pos_gu` achieve **lower** avg overall and main-6 diffs (5.6-6.0 and 7.7-7.9) than the core `pos_ablated_split_ctx` family (7.4-7.6 / 8.8-9.1), i.e. closer FIFA agreement on this small sample—without matching `h_fifa_attributes`.
- The `pos_ablated_split_ctx` family has moderate FIFA alignment (avg overall diff 7.4-7.6, main-6 diff 8.8-9.1), somewhat worse than simpler models. This is expected since these models optimize for behavioral similarity from event data, not for FIFA attribute matching.
- `split_ctx_ps` has the worst position match among GNN models (25%), meaning 75% of its recommended substitutes play a different position group. For practical scouting, this is a concern.
- FIFA comparison has inherent limitations: FIFA ratings reflect subjective assessments and commercial considerations, not pure behavioral similarity.

---

## 7. Qualitative Neighbor Inspection

Nearest-neighbor tables for 15 query players across selected models. These provide face-validity checks.

### 7.1 Cristiano Ronaldo (Center Forward)

| Rank | pos_ablated_split_ctx | ema | ema_v2 | acts_in_dropout | player_samp |
|---:|---|---|---|---|---|
| 1 | Morata (0.944) | Morata (0.954) | Memphis Depay (0.951) | Morata (0.953) | Kane (0.954) |
| 2 | Memphis Depay (0.942) | Memphis Depay (0.947) | Morata (0.934) | Memphis Depay (0.948) | Morata (0.954) |
| 3 | Kane (0.929) | Kane (0.935) | V. Boniface (0.933) | Kane (0.934) | Arnautovic (0.952) |
| 4 | V. Boniface (0.927) | Mitrović (0.934) | Mitrovic (0.929) | V. Boniface (0.932) | Schick (0.951) |
| 5 | Petkovic (0.919) | Yaremchuk (0.933) | Kane (0.925) | Benzema (0.926) | Livaja (0.948) |

All models return plausible center forwards. `player_samp` shows tighter cosine similarity among neighbors (0.948-0.954), reflecting its compressed within-position clustering. The `pos_ablated_split_ctx` family shows wider spread, suggesting more discriminative embeddings. `acts_in_dropout` matches Morata/Memphis/Kane/Boniface/Benzema—nearly the same cast as the split-ctx family with Benzema instead of Petković at rank 5.

### 7.2 Toni Kroos (Left Defensive Midfield)

| Rank | pos_ablated_split_ctx | ema | ema_v2 | player_samp |
|---:|---|---|---|---|
| 1 | Declan Rice (0.937) | Declan Rice (0.936) | Declan Rice (0.943) | Frenkie de Jong (0.954) |
| 2 | Grillitsch (0.916) | Frenkie de Jong (0.915) | Grillitsch (0.903) | Declan Rice (0.944) |
| 3 | Frenkie de Jong (0.914) | Grillitsch (0.905) | Frenkie de Jong (0.901) | Axel Witsel (0.942) |
| 4 | Reijnders (0.905) | Reijnders (0.897) | Ekdal (0.897) | Casimiro (0.934) |
| 5 | M. Arnold (0.899) | Lobotka (0.890) | Casimiro (0.895) | Reijnders (0.928) |

Strong agreement across models: Rice, de Jong, Grillitsch are consistently top-ranked. These are all tempo-controlling deep-lying midfielders, validating the behavioral representation.

### 7.3 Messi (Right Center Forward) -- Stress Test

| Rank | pos_ablated_split_ctx | ema | player_samp |
|---:|---|---|---|
| 1 | Griezmann (0.939) | Griezmann (0.942) | Deniz Undav (0.907) |
| 2 | Arda Guler (0.905) | Tadić (0.912) | Kulusevski (0.881) |
| 3 | McGinn (0.904) | McGinn (0.907) | Almoez Ali (0.875) |
| 4 | Shaqiri (0.903) | Majer (0.905) | Wout Weghorst (0.863) |
| 5 | Tadic (0.891) | Shaqiri (0.902) | Sporar (0.852) |

`pos_ablated_split_ctx` and `ema` retrieve attacking playmakers (Griezmann, Tadić, McGinn, Shaqiri, Majer) that match Messi's creative role. `player_samp` retrieves center forwards by position (Undav, Weghorst, Sporar) rather than play style, suggesting position dominates its embeddings more than behavioral nuance.

---

## 8. Head-Coach Qualitative Scoring

### Setup

Six query players (Neuer, Van Dijk, Alexander-Arnold, De Bruyne, Messi, Mbappé) were evaluated across six models. For each, the model's top-5 nearest neighbours (from `evaluations/{tag}/qualitative_neighbors/`) were scored 1–5 on how reasonable they are as substitutes or stylistic equivalents.

**Prompt used:**

> "You can imagine yourself as a soccer headcoach in a realistic scenario. Only give 5 for really the best matches. Strictly score them with brief reasonings."

**Scoring model:** Claude Opus (Anthropic), acting as the evaluator. No external references were consulted beyond general football knowledge; all candidate lists were verified to match the JSON evaluation artifacts exactly.

**Scale:** 1 = poor (no candidate is a realistic substitute), 2 = weak (one reasonable match at best), 3 = decent (2–3 useful suggestions), 4 = good (3–4 genuinely useful), 5 = excellent (most/all are strong stylistic matches; reserved for truly outstanding lists).

### Results

| Query | M1 baseline | M2 pos_abl_sc | M3 split_ctx | M4 pos_abl_sc_ema | M5 acts_drop | M6 acts_drop_pos_gu |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|
| **Neuer** | 2 | 3 | 3 | **4** | 3 | **4** |
| **Van Dijk** | 3 | 2 | 3 | 2 | 3 | **4** |
| **TAA** | 3 | 2 | 1 | 2 | 2 | 2 |
| **De Bruyne** | 3 | 3 | 2 | 3 | 3 | **4** |
| **Messi** | 1 | 3 | 1 | 3 | 3 | 3 |
| **Mbappé** | 1 | **4** | 1 | **4** | **4** | **4** |
| **Total (/30)** | **13** | **17** | **11** | **18** | **18** | **21** |
| **Average** | **2.2** | **2.8** | **1.8** | **3.0** | **3.0** | **3.5** |

### Per-Query Reasoning

**Neuer (GK)**
- M1 (2): Pickford/Livaković are decent international keepers but Hrádecký, Vanja Milinković Savić, and Bachmann lack Neuer's sweeper-keeper profile. Misses Donnarumma, Lloris, Ederson.
- M2 (3): Donnarumma is a strong pick (elite, sweeper tendencies). Rest are generic.
- M3 (3): Maignan stands out — elite sweeper-keeper with great distribution, arguably the best possible Neuer comparison.
- M4 (4): Pickford, Donnarumma, Unai Simón, Lloris — four international #1 keepers with sweeping/distribution qualities. Strongest GK list.
- M5 (3): Donnarumma is the standout. Noppert and Dimitrievski are limited.
- M6 (4): Lloris, Unai Simón, Donnarumma — three elite keepers with sweeper tendencies. Tied for best GK list.

**Van Dijk (LCB)**
- M1 (3): Thiago Silva (elite ball-playing CB, leadership) is an excellent match. Vertonghen is solid. Others are generic.
- M2 (2): Akanji (Man City, composed, ball-playing) is the best pick but there is no standout VVD-caliber match.
- M3 (3): Thiago Silva + Calafiori (Arsenal, young ball-playing LCB) are strong modern picks.
- M4 (2): Akanji is good but Kashia and Khoukhi are obscure, lower-tier CBs that drag the list down.
- M5 (3): Thiago Silva (outstanding) and Akanji (solid) anchor the list. Arajuuri is a miss.
- M6 (4): **Rúben Dias** is arguably THE best possible VVD comparison in the dataset (PL elite, commanding, ball-playing). Akanji adds depth. Best CB list across models.

**Trent Alexander-Arnold (RDM)**
- M1 (3): Hakimi (elite creative attacking fullback) and Carvajal (elite progressive RB) are individually strong. Collins Fai is a miss.
- M2 (2): Dani Alves (legendary creative RB) is conceptually excellent but data is old. Geertruida is versatile; others miss TAA's creative passing.
- M3 (1): Hysaj, Sabaly, Cash, Tymchyk, Wass — all generic right-backs with none of TAA's defining creative vision. The positional model retrieves position peers, not stylistic matches.
- M4 (2): Dani Alves is the right idea. Militão, Lucas Melo, Johnston don't capture the creative profile.
- M5 (2): Dani Alves and Timber have some appeal but the rest miss.
- M6 (2): Geertruida and De Paul (creative progressive passing parallels) are interesting; Timber is versatile. Still no model truly captures TAA's unique profile.

**De Bruyne (CAM)**
- M1 (3): Foden (excellent — creative, versatile, Man City system, vision) is a standout. Xavi Simons reasonable. Dina Ebimbe is a miss.
- M2 (3): Bruno Fernandes (excellent — creative, throughballs, set-pieces) and Zieliński (technical playmaker) are strong.
- M3 (2): Bruno Fernandes saves the list. Dina Ebimbe and Sliti are misses.
- M4 (3): Zieliński is good; Messi appearing is interesting (both are elite creators). Tadić is reasonable. Robin Lod is a miss.
- M5 (3): Zieliński and Hamšík (creative playmaker, similar passing range) are strong picks.
- M6 (4): Bruno Fernandes (excellent), Zieliński (good), Griezmann (creative versatile attacker) — three genuinely useful suggestions in one list. Best KDB list.

**Messi (RCF)**
- M1 (1): Gakpo, Che Adams, Vlahović, Bergwijn — mostly physical strikers or direct wingers. No one captures dribbling + vision + playmaking. Suárez has some link-up chemistry historically but is a different profile.
- M2 (3): Griezmann (intelligent movement, creative between lines) is the best conceptual match. Shaqiri and Tadić have creative/technical elements.
- M3 (1): Che Adams, Mikautadze, Campbell, Kalajdžić (6'7" target man), Dovbyk — not a single creative/technical forward. Positional model retrieves center forwards by role.
- M4 (3): Griezmann (good), Tadić (creative, vision), Shaqiri (technical). Reasonable cluster of creative attacking players.
- M5 (3): Griezmann, Shaqiri, Tadić, Kramarić (creative Croatian forward) — consistent creative forward cluster.
- M6 (3): Griezmann, Shaqiri, Arda Güler, Kramarić, Mertens — all are creative, technically gifted attacking players. No single list captures Messi's unique combination, but this is a reasonable set.

**Mbappé (LW)**
- M1 (1): Yılmaz, Immobile, Kieffer Moore (aerial target man), Petković — almost entirely center forwards / poachers. None share pace, directness, or wing play.
- M2 (4): Olmo, Diogo Jota, Boufal, João Félix, Musiala — all creative, technically gifted forward/wingers. Strong stylistic cluster.
- M3 (1): Embolo, Dzyuba, Füllkrug, Morata, Petković — strikers and target men. The positional model fails here completely.
- M4 (4): Thuram (French, pacy), Diogo Jota (LW, clinical), Olmo, Gakpo (LW, pace, goals), João Félix — strong modern left-sided attackers.
- M5 (4): Olmo, Diogo Jota, João Félix, Thuram, Insigne (creative Italian LW) — consistently strong set.
- M6 (4): Olmo, Diogo Jota, Gakpo, Thuram, Sterling (pacy LW, dribbling) — all five are modern pace-based left-sided attackers.

### Key Takeaways

1. **Model 6 (`acts_in_dropout_pos_gu`) wins the qualitative evaluation** (21/30, avg 3.5) despite ranking 5th on pseudo-GT mean rank in the automated metrics. Its group-uniformity term produces better within-position discrimination, yielding Rúben Dias for VVD and Bruno Fernandes for KDB — picks no other model surfaces.
2. **Position-ablated models (M2, M4, M5, M6) dramatically outperform positional models (M1, M3) on creative/unique players** (Messi, Mbappé). Models that keep position information retrieve positional peers (strikers for Messi, poachers for Mbappé) rather than stylistic matches.
3. **Model 3 (`split_ctx`) has the best automated pseudo-GT mean rank (56.1) but the worst qualitative score (11/30)**. This highlights a divergence between pair-retrieval metrics and face-validity neighbor inspection for famous players.
4. **TAA is universally difficult** (max score 3, from M1 via Hakimi/Carvajal). His unique creative-RB/inverted-midfielder hybrid profile has no close equivalent in the dataset.
5. **No model scores 5 on any query.** The closest to a 5 is M6's VVD list (anchored by Rúben Dias).

---

## 9. Critical Assessment

### 9.1 What the evaluation suite measures well

- **Pseudo-ground-truth pairs** test the substitute-finding use case with LLM-generated pairs. These are directional sanity checks, not expert-validated ground truth (see §10.1).
- **Policy diagnostic** tests whether embedding proximity reflects actual on-pitch behavioral similarity using action distributions from 200 simulated situations.
- **Empirical behavioral fidelity** (§4.3–4.5) cross-validates the policy diagnostic without self-referential bias: it compares embedding distance against JS divergence of **observed** player actions bucketed by game state. This is the strongest non-circular behavioral signal in the suite.
- **Competition-split self-consistency** tests real-world robustness: can the model produce stable player representations from data collected in different tournaments?

### 9.2 What the evaluation suite does NOT test

- **Temporal stability:** All data comes from a fixed window. We do not test whether a player's embedding changes appropriately as their playing style evolves.
- **Low-data players:** The minimum possession threshold filters out many real scouting targets (young/emerging players with limited data).
- **Same-position discrimination:** The policy diagnostic evaluates within position-gender groups, but the pseudo-ground-truth pairs include cross-positional comparisons (e.g., Bellingham-Griezmann, where one is nominally a midfielder and the other a forward).
- **Practical retrieval at scale:** All galleries have ~1,000 players. In production, galleries may contain 10,000+ players across multiple leagues and seasons.

### 9.3 Why `pos_ablated_split_ctx_v2` collapsed

`pos_ablated_split_ctx_v2` reduced uniformity weight (lambda=0.7) compared to the base variant (lambda=1.0), while keeping the same t=4.0. The result is catastrophic: GT mean_rank 198.8, competition-split mean_rank 200.8, random-half hit@10 0.094. This confirms that position ablation creates a strong dependency on uniformity loss to maintain embedding spread. Reducing lambda from 1.0 to 0.7 — a 30% reduction — causes representational collapse.

### 9.4 The player-sampling paradox

`player_samp` and `split_ctx_ps` achieve the best random-half self-consistency (mean_rank 2.7-2.9) and highest substitute quality ratios (11.9-13.1), yet fail at pseudo-GT retrieval (tier-1 hit@10 = 0.0) and competition-split consistency (mean_rank 89-94).

**Explanation:** Player sampling forces the model to see the same player across different mini-batches, creating tight per-player clusters. This directly optimizes for within-context self-consistency (the random-half test) and produces wider embedding spread (higher random-k JS in the substitute ratio denominator). However, it does not teach the model what makes two *different* players functionally similar. The training signal is "same player = close" rather than "similar behavior = close." The result is a model that excels at player re-identification but underperforms at cross-player similarity retrieval.

### 9.5 The behavioral fidelity-retrieval trade-off

`pos_ablated` has the highest Spearman rho (0.974) but mediocre retrieval (GT mean_rank 69.6). This occurs because its highly compressed embedding space (mean cosine dist 0.085) means the behavioral ordering is preserved but the absolute distances between players are tiny, making retrieval noisy. The `pos_ablated_split_ctx` family sacrifices some rho (0.929-0.944) for much better retrieval (mean_rank 62.6-64.3) by spreading embeddings apart with stronger uniformity loss.

---

## 10. Weighted Model Selection

### 10.1 Why the Pseudo-Ground-Truth Pairs Are Not Gold-Standard

The 15 "ground-truth" pairs (§2) were **generated by an LLM** from publicly cited analytics comparisons, not validated by football domain experts or StatsBomb analysts (see `docs/pseudo_ground_truth.md`, which labels them "directional sanity checks, not strict benchmarks"). This matters:

- The pairs reflect LLM biases toward famous players, media narratives, and positional similarity
- A model that disagrees with the LLM's judgment may be capturing genuine behavioral patterns the LLM missed
- Weighting pseudo-GT retrieval as the primary metric would effectively optimize for "agreement with an LLM" rather than "finding good substitutes"

Consequently, pseudo-GT metrics receive **reduced weight** (0.12 combined) compared to objectively grounded metrics (behavioral + cross-context = 0.55).

### 10.2 All 14 Metrics & Weight Justification

Each metric is min-max normalized across all 11 GNN models (1.0 = best, 0.0 = worst). The weighted score is `Σ(w_i × norm_i) / Σ(w_i for available metrics)`, so models with missing data (e.g., no qualitative or no empirical behavioral) are evaluated on the metrics they have.

| # | Metric | Weight | Category | Reasoning |
|---|--------|-------:|----------|-----------|
| 1 | Spearman Rho (model-based) | **0.14** | Model Behavioral | Behavioral ordering fidelity from the model's action-prediction head. Reduced from 0.18 (12-metric) because EB Rho now provides non-self-referential behavioral correlation. |
| 2 | Absolute Top-k JS (model-based) | **0.10** | Model Behavioral | Do retrieved neighbors play similarly per the model? Reduced from 0.14 because empirical behavioral metrics now cross-validate this signal. |
| 3 | Comp-Split Mean Rank | **0.17** | Cross-Context | Cross-competition robustness — essential for scouting across leagues. Objective, computed from real data. |
| 4 | Comp-Split Hit@10 | **0.06** | Cross-Context | Precision supplement: can the model find the same player in the top 10 across competitions? |
| 5 | Random-Half Mean Rank | **0.05** | Within-Context | Within-context embedding stability. Easier test, catches degenerate models. |
| 6 | Pseudo-GT Mean Rank | **0.08** | Pseudo-GT | LLM-generated pair retrieval. Informative but biased. Reduced weight. |
| 7 | Pseudo-GT Hit@50 | **0.04** | Pseudo-GT | Practical shortlist from LLM pairs. |
| 8 | Sub Quality Ratio (model-based) | **0.04** | Model Behavioral | Relative improvement of top-k neighbors over random. Reduced from 0.05. |
| 9 | FIFA Main-6 Diff | **0.05** | External | Independent skill-profile validation against FIFA ratings. Small sample (16 non-GK pairs) but provides external check. |
| 10 | FIFA Position Match % | **0.05** | External | Practical relevance: substitutes typically must share a position group. |
| 11 | Test F1 Avg | **0.03** | Proxy | Supervised proxy task. Guards against degenerate representations but is not the end goal. |
| 12 | Qualitative Head-Coach Avg | **0.10** | Face Validity | Scoring by an evaluator (Claude) on 6 famous-player queries (§8). Available for 6 of 11 models; weight redistributed for the other 5. |
| 13 | EB Spearman Rho | **0.07** | Empirical Behavioral | Non-self-referential correlation between embedding distance and JS divergence of **observed** action distributions (§4.4). Immune to self-referential inflation. Available for 10 of 11 models. |
| 14 | EB Substitute Ratio | **0.02** | Empirical Behavioral | Ratio of random-K to top-K observed-action JS (§4.5). Directly tests whether embedding neighbors are better behavioral matches than random same-group peers, using actual decisions. |
| | **Total** | **1.00** | | |

**Category totals:**

| Category | Metrics | Combined Weight | Reasoning |
|----------|---------|----------------:|-----------|
| Model-Based Behavioral | Rho + Top-k JS + Sub Ratio | **0.28** | Model's action predictions correlate with embedding geometry. Self-referential but smooth and high-resolution. Reduced from 0.37 (12-metric) to accommodate empirical behavioral. |
| Empirical Behavioral | EB Rho + EB Sub Ratio | **0.09** | Non-self-referential behavioral validation from observed actions. Noisier but externally grounded. |
| Cross-Context Robustness | Comp MR + Comp H@10 | **0.23** | Production-critical: scouting across leagues requires stable embeddings. |
| Pseudo-GT Retrieval | GT MR + GT H@50 | **0.12** | Useful sanity check but LLM-generated pairs, not expert-validated. |
| External Validation | FIFA M6 + FIFA Pos + Qualitative | **0.20** | Independent checks from outside the training paradigm. |
| Supplementary | Random MR + F1 | **0.08** | Useful but less diagnostic for substitute finding specifically. |

**Excluded metrics and why:**
- **Shot/Goal AUC**: Not discriminative (range 0.961–0.976 / 0.972–0.989 across all models)
- **Self-cosine values**: Misleading — high cosine can reflect embedding compression, not quality (e.g., `pos_ablated` cosine 0.974 but poor retrieval)
- **Individual F1 components**: Summarized by F1 Avg
- **GT Hit@5/10/20, Median Rank**: Noisier or redundant with GT Mean Rank + Hit@50
- **Group-wise Rho**: Summarized by Overall Rho
- **EB Top-K JS (absolute)**: Correlated with EB Sub Ratio; the ratio is more diagnostic for substitute quality

### 10.3 Comprehensive Final Scores — All Models

**Raw metric values** for every model on all 14 evaluation dimensions:

| Model | Rho | Top-k JS | Comp MR | Comp H@10 | Rand MR | GT MR | GT H@50 | Sub Ratio | FIFA M6 | FIFA Pos% | F1 Avg | Qual | EB Rho | EB Ratio |
|-------|----:|--------:|---------:|----------:|--------:|------:|--------:|----------:|--------:|----------:|-------:|-----:|-------:|---------:|
| baseline | 0.744 | 0.003298 | 81.6 | 0.238 | 26.1 | 67.6 | 0.667 | 3.84 | 8.6 | 44 | 0.692 | 2.2 | 0.1879 | 1.322 |
| player_samp | 0.788 | 0.001122 | 93.5 | 0.264 | 2.9 | 70.2 | 0.333 | 11.91 | 8.9 | 50 | 0.700 | — | 0.2230 | 1.562 |
| split_ctx | 0.710 | 0.004611 | 83.9 | 0.215 | 36.4 | 56.1 | 0.600 | 3.13 | 8.5 | 56 | 0.684 | 1.8 | 0.1891 | 1.247 |
| split_ctx_ps | 0.778 | 0.001025 | 89.3 | 0.262 | 2.7 | 71.4 | 0.333 | 13.11 | 9.2 | 25 | 0.688 | — | 0.2296 | 1.698 |
| pos_ablated | 0.974 | 0.001110 | 81.3 | 0.275 | 13.9 | 69.6 | 0.567 | 8.59 | 8.6 | 50 | 0.687 | — | 0.1961 | 1.658 |
| pos_ablated_split_ctx | 0.929 | 0.001408 | 67.3 | 0.346 | 6.3 | 63.1 | 0.633 | 6.63 | 9.0 | 50 | 0.675 | 2.8 | 0.2070 | 1.623 |
| pos_ablated_split_ctx_v2 | 0.686 | 0.006962 | 200.8 | 0.076 | 177.8 | 198.8 | 0.200 | 3.07 | 9.8 | 38 | 0.639 | — | 0.1038 | 1.211 |
| pos_ablated_split_ctx_ema | 0.944 | 0.001040 | 67.7 | 0.318 | 9.3 | 62.6 | 0.700 | 7.95 | 9.1 | 56 | 0.650 | 3.0 | 0.2074 | 1.718 |
| pos_ablated_split_ctx_ema_v2 | 0.934 | 0.001453 | 67.5 | 0.349 | 6.1 | 64.3 | 0.633 | 6.35 | 8.8 | 50 | 0.674 | — | — | — |
| acts_in_dropout | 0.928 | 0.001031 | 68.9 | 0.351 | 8.7 | 68.2 | 0.633 | 7.75 | 7.9 | 50 | 0.657 | 3.0 | 0.2066 | 1.953 |
| acts_in_dropout_pos_gu | 0.922 | 0.001051 | 74.0 | 0.328 | 8.2 | 66.7 | 0.567 | 7.62 | 7.7 | 50 | 0.660 | 3.5 | 0.1946 | 1.745 |

Direction key: Rho ↑, Top-k JS ↓, Comp MR ↓, Comp H@10 ↑, Rand MR ↓, GT MR ↓, GT H@50 ↑, Sub Ratio ↑, FIFA M6 ↓, FIFA Pos% ↑, F1 Avg ↑, Qual ↑, EB Rho ↑, EB Ratio ↑. "—" = data not available for this model.

**Min-max normalized scores** (1.0 = best, 0.0 = worst across all 11 GNN models):

| Model | Rho | JS | Comp MR | Comp H10 | Rand MR | GT MR | GT H50 | Sub Ratio | FIFA M6 | FIFA Pos | F1 | Qual | EB Rho | EB Sub | **Weighted Score** |
|-------|----:|---:|--------:|---------:|--------:|------:|-------:|----------:|--------:|---------:|---:|-----:|-------:|-------:|-------------------:|
| acts_in_dropout_pos_gu | 0.82 | 1.00 | 0.95 | 0.92 | 0.97 | 0.93 | 0.73 | 0.45 | 1.00 | 0.81 | 0.34 | 1.00 | 0.72 | 0.72 | **0.867** |
| acts_in_dropout | 0.84 | 1.00 | 0.99 | 1.00 | 0.97 | 0.92 | 0.87 | 0.47 | 0.90 | 0.81 | 0.29 | 0.71 | 0.82 | 1.00 | **0.864** |
| pos_ablated_split_ctx_ema_v2 | 0.86 | 0.93 | 1.00 | 0.99 | 0.98 | 0.94 | 0.87 | 0.33 | 0.48 | 0.81 | 0.56 | — | — | — | **0.860** |
| pos_ablated | 1.00 | 0.99 | 0.90 | 0.73 | 0.94 | 0.91 | 0.73 | 0.55 | 0.57 | 0.81 | 0.78 | — | 0.73 | 0.60 | **0.847** |
| pos_ablated_split_ctx_ema | 0.90 | 1.00 | 1.00 | 0.88 | 0.96 | 0.95 | 1.00 | 0.49 | 0.33 | 1.00 | 0.17 | 0.71 | 0.82 | 0.68 | **0.846** |
| pos_ablated_split_ctx | 0.85 | 0.94 | 1.00 | 0.98 | 0.98 | 0.95 | 0.87 | 0.35 | 0.38 | 0.81 | 0.59 | 0.59 | 0.82 | 0.56 | **0.820** |
| player_samp | 0.36 | 0.98 | 0.80 | 0.68 | 1.00 | 0.90 | 0.27 | 0.88 | 0.43 | 0.81 | 1.00 | — | 0.95 | 0.47 | **0.735** |
| split_ctx_ps | 0.32 | 1.00 | 0.84 | 0.68 | 1.00 | 0.89 | 0.27 | 1.00 | 0.29 | 0.00 | 0.81 | — | 1.00 | 0.66 | **0.692** |
| baseline | 0.20 | 0.62 | 0.89 | 0.59 | 0.87 | 0.92 | 0.93 | 0.08 | 0.57 | 0.61 | 0.86 | 0.24 | 0.67 | 0.15 | **0.593** |
| split_ctx | 0.08 | 0.40 | 0.88 | 0.51 | 0.81 | 1.00 | 0.80 | 0.01 | 0.62 | 1.00 | 0.74 | 0.00 | 0.68 | 0.05 | **0.536** |
| pos_ablated_split_ctx_v2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.42 | 0.00 | — | 0.00 | 0.00 | **0.023** |

Models with "—" have their available metric weights renormalized so scores remain comparable. `pos_ablated_split_ctx_ema_v2` is missing both Qual and EB metrics (÷ 0.81); `pos_ablated`, `player_samp`, `split_ctx_ps`, `pos_ablated_split_ctx_v2` are missing Qual only (÷ 0.90).

### 10.4 Sensitivity Analysis

| Weighting variant | #1 | #2 | #3 |
|-------------------|-----|-----|-----|
| **Balanced** (14-metric, as above) | **acts_in_dropout_pos_gu** (0.867) | acts_in_dropout (0.864) | pos_ablated_split_ctx_ema_v2 (0.860) |
| **Behavioral-Heavy** (Rho=.17, JS=.12, EBR=.09, EBS=.03) | acts_in_dropout (0.862) | **acts_in_dropout_pos_gu** (0.857) | pos_ablated (0.850) |
| **Retrieval-Heavy** (Comp=.20, CH10=.08, GT=.10) | acts_in_dropout (0.877) | **acts_in_dropout_pos_gu** (0.874) | pos_ablated_split_ctx_ema (0.855) |
| **External-Heavy** (FIFA=.08+.08, Qual=.14) | **acts_in_dropout_pos_gu** (0.878) | acts_in_dropout (0.859) | pos_ablated (0.834) |

The two `acts_in_dropout` variants are always top-2 across all weight variants. `acts_in_dropout_pos_gu` wins under balanced and external-heavy weights (where its best-in-class qualitative and FIFA scores dominate); `acts_in_dropout` wins under behavioral-heavy and retrieval-heavy weights (where its best-in-class EB Sub Ratio and Comp-Split Hit@10 dominate).

Among models with **full 14-metric coverage** (no missing qualitative or EB):

`acts_in_dropout_pos_gu` (0.867) > `acts_in_dropout` (0.864) > `pos_ablated_split_ctx_ema` (0.846) > `pos_ablated_split_ctx` (0.820)

This ordering is robust across all weight variants.

### 10.5 Caveat on `pos_ablated` (#4)

`pos_ablated` ranks 4th (0.847) but has only 13/14 metric coverage (no qualitative data). Its top ranking on model-based Spearman rho (1.0 normalized, from 0.974) contributes 0.14 to its score. However, this high rho is a consequence of embedding compression (self-cosine 0.974, mean cosine distance 0.085) — all players look nearly identical, so behavioral ordering is trivially preserved. Its practical retrieval metrics (GT MR 69.6, comp MR 81.3) are substantially worse than the `pos_ablated_split_ctx` family. Its EB Rho (0.1961) is mid-pack, confirming that compression inflates model-based Rho but not the externally grounded empirical metric.

If qualitative data were available, `pos_ablated` would likely score poorly (its compressed embedding space produces generic, undifferentiated neighbor lists). Its true ranking is probably 5th–6th.

### 10.6 Caveat on `pos_ablated_split_ctx_ema_v2` (#3)

`pos_ablated_split_ctx_ema_v2` ranks 3rd (0.860) but is missing both qualitative (0.10) and empirical behavioral (0.09) metrics — its score is based on only 11 of 14 dimensions (÷ 0.81). If EB metrics were available (most likely mid-pack based on its position between `pos_ablated_split_ctx_ema` and `pos_ablated_split_ctx` architecturally), its true ranking would likely be 4th–5th.

---

## 11. Conclusion

### Best model for substitute-player retrieval: `acts_in_dropout_pos_gu`

**Weighted score = 0.867** (§10.3), highest among all 11 models under the 14-metric substitute-optimized analysis. This result is robust: top-2 across all four weight variants tested (§10.4).

| Criterion | Value | Rank / 11 | Evidence |
|-----------|------:|:---------:|----------|
| Spearman Rho (model) | 0.922 | 6 | Strong behavioral fidelity without compression artifacts |
| Absolute Top-k JS | 0.001051 | 4 | Retrieved neighbors genuinely play alike |
| Comp-Split Mean Rank | 74.0 | 5 | Moderate cross-context robustness |
| Comp-Split Hit@10 | 0.328 | 4 | Good cross-context precision |
| FIFA Main-6 Diff | **7.7** | **1** | **Best** FIFA skill-profile match among all GNN models |
| Qualitative Head-Coach | **3.5/5** | **1** | **Best** face-validity neighbors (Rúben Dias for VVD, Bruno Fernandes for KDB) |
| EB Substitute Ratio | 1.745 | 2 | Embedding neighbors 1.75× more behaviorally similar than random (observed actions) |
| Pseudo-GT Mean Rank | 66.7 | 5 | Moderate on LLM-generated pairs |

**Why this model wins:** Group-uniformity regularization (`uniformity_group_weight=3.0`) sharpens within-position discrimination, producing neighbors that are not only behaviorally similar (top-k JS 0.001051) but also externally validated — closest FIFA skill profiles (7.7-point main-6 diff) and highest qualitative approval (3.5/5). The combination of strong internal behavioral metrics with the best external validation is what sets it apart. Its EB Substitute Ratio (1.745, 2nd-best) confirms that this advantage extends to real observed player behavior, not just model predictions.

**Limitations:**
- Competition-split mean rank (74.0) is 6.7 points behind the family best (`pos_ablated_split_ctx` at 67.3). The group-uniformity term trades some cross-context stability for sharper within-position boundaries.
- EB Rho (0.1946) is mid-pack (7th of 10), lower than `acts_in_dropout` (0.2066). However, this partially reflects lower embedding spread reducing rank-correlation power, not poorer behavioral alignment.
- Pseudo-GT mean rank (66.7) is mid-pack. If the LLM-generated pairs were genuine expert ground truth, this would be more concerning.
- Qualitative evaluation covers only 6 famous players. Performance on obscure/emerging players is untested.

### Runner-up: `acts_in_dropout` (score = 0.864)

| Metric | Value | Note |
|--------|------:|------|
| EB Substitute Ratio | **1.953** | **Best** — neighbors nearly 2× better than random on observed actions |
| Comp-Split Hit@10 | **0.351** | **Best** cross-context precision |
| Top-k JS (model) | **0.001031** | **2nd-best** absolute neighbor quality |
| EB Rho | 0.2066 | 5th — solid empirical behavioral correlation |
| FIFA Main-6 Diff | 7.9 | 2nd-best FIFA skill match |
| Qualitative | 3.0/5 | Good face validity |

Simpler than `pos_gu` (no group-uniformity term). **Wins under behavioral-heavy and retrieval-heavy weight variants** (§10.4). The margin behind `pos_gu` is now just 0.003 (down from 0.016 in the 12-metric analysis), with `acts_in_dropout`'s best-in-class EB Substitute Ratio (1.953) nearly closing the gap. Best choice if maximizing cross-competition hit@10 or empirical behavioral validation matters most.

### Also competitive: `pos_ablated_split_ctx_ema` (score = 0.846)

Still strong:
- **Best pseudo-GT hit@50 (0.700)** and **2nd-best GT mean rank (62.6)** — excels at retrieving LLM-curated pairs
- Best model-based Spearman rho among fully-evaluated models (0.944)
- EB Substitute Ratio 1.718 (3rd-best) and EB Rho 0.2074 (3rd-best) — consistent empirical behavioral performance
- Best competition-split mean rank in its family (67.7)

Its weakness: FIFA main-6 diff of 9.1 (2nd-worst among non-collapsed models) and lowest supervised F1 (0.650). When pseudo-GT is treated as gold standard, this model wins; when external validation is included and GT is properly discounted, it drops to 5th. This reflects its optimization for internal consistency over externally validated neighbor quality.

### Not recommended: `player_samp`, `split_ctx_ps`, `split_ctx`

- **`player_samp` / `split_ctx_ps`** (scores 0.735 / 0.692): Tier-1 hit@10 = 0.0, competition-split mean rank > 89. Despite leading on EB Rho (0.22–0.23), their EB Substitute Ratios are mid-pack (1.56–1.70), confirming that their high EB Rho partially reflects embedding spread rather than superior behavioral discrimination. Strengths (random-half consistency, model-based sub ratio) are artifacts of the player sampling mechanism (§9.4).
- **`split_ctx`** (score 0.536): Best pseudo-GT mean rank (56.1) but worst model-based Spearman rho (0.710), worst top-k JS (0.004611), worst qualitative score (1.8/5), and near-worst EB Substitute Ratio (1.247). Retrieves positional peers rather than behavioral matches.
- **`pos_ablated_split_ctx_v2`** (score 0.023): Collapsed due to insufficient uniformity weight.
