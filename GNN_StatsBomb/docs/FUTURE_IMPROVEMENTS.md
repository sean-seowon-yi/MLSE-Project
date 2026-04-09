# Future Improvements & Considerations

This document captures assessment of the current system and concrete directions for improvement, for future reference. It is not a commitment to implement; it is a living list of options.

---

## Critical vulnerabilities & blind spots

The following points were identified as theoretical and practical vulnerabilities in the current design. They are recorded here so that future work can address them explicitly.

### 1. ~~Graph edge granularity: teammate vs opponent~~ (IMPLEMENTED — audit #13)

**Issue:** Phase 3 used a single edge type `(player, context_for, event)` for 360 spatial context. A teammate 3 m ahead is a passing option; an opponent 3 m ahead is a block. With one edge type, the GNN had to rely only on the player node’s `is_possession_team` feature to tell them apart.

**Improvement:** Split into two edge types: `(player, context_for_tm, event)` and `(player, context_for_opp, event)`. HeteroConv now learns separate projection matrices and attention weights for offensive vs defensive spatial constraints.

**Status:** Implemented. See SYSTEM_DESIGN.md audit #13. `GraphConfig.split_context_edges` (default `False` for backward compatibility; enable via `--split_context_edges` CLI flag). Requires re-running Phases 3–5 with the flag.

### 2. Temporal edge scaling

**Issue:** Time deltas are normalised as \min(\Delta t / 30, 1.0). Thirty seconds is a very large window; the tactical difference between a 0.5 s one-touch and a 3 s delayed pass is large, while 20 s vs 25 s is negligible. Linear scaling compresses the important 0–5 s range into a narrow band (e.g. 0.0–0.16), reducing the model’s ability to distinguish tempo.

**Limitation — delta time is noisy:** The underlying \Delta t is a noisy proxy for “tempo” or “urgency.” It is derived from StatsBomb event timestamps (or minute/second), which mix annotation delay, ball-contact-to-contact time, and occasional stoppages within a possession. The same nominal 2 s gap can correspond to different tactical situations (e.g. quick one-touch vs deliberate pause). If timestamps are only to the second, the 0–2 s range is coarse (few distinct values). Outliers (very short double-taps or long pauses) can further distort the signal. So temporal edge attributes should be treated as a weak, possibly optional feature rather than a reliable measure of true tempo; consider ablations with/without them, or down-weighting them. Better ball-contact-level timestamps would make any scaling (linear or non-linear) more informative.

**Improvement:** Use a non-linear scaling, e.g. log scale or exponential decay \exp(-\lambda \Delta t), to give higher resolution to short intervals and asymptote for long delays.

### 3. “Ghosting” counterfactual fallacy (Phase 7)

**Issue:** Phase 7 is framed as “if placed in the exact same game situation, both players would choose similar actions.” In reality, substituting a different player (e.g. an aggressive dribbler for a traditional winger) would have changed opponent positioning (gravity) before the event. The model is evaluating how player B reacts to the constraints created by player A’s presence, not a strict 1:1 counterfactual.

**Improvement:** Reframe the analysis as **“action preference given identical constraints”** and state this limitation clearly in documentation and reports.

### 4. Discretization of continuous actions

**Issue:** Phase 4 bins angles into 9 classes and lengths into 5. Angle and length are continuous; binning makes near-boundary passes (e.g. 15 m vs 16 m) appear as different classes and can over-penalise the model during training.

**Improvement:** Consider a mixture density network (MDN) head or direct regression of continuous (dx, dy) (e.g. Huber or MSE) for passes and carries, instead of (or in addition to) classification bins.

### 5. Minimum possessions and confidence

**Issue:** `min_samples_per_player = 50` may still yield noisy z_p given high variance in situations. There is no confidence or uncertainty attached to embeddings.

**Improvement:** Add a confidence score (e.g. based on number of possessions and/or embedding stability) and use it to filter or down-weight low-sample players in similarity search (Phase 6).

### 6. ~~Evaluation metrics do not validate similarity~~ (PARTIALLY IMPLEMENTED)

**Issue:** Phase 5 evaluation is based on action/outcome prediction accuracy. There is no ground truth for “similarity,” and no check that nearest neighbours correspond to domain intuition.

**Improvement:**  

- Build a small, manually annotated set of “known similar” players (e.g. from established statistical clusters or expert labels).  
- Report **rank correlation** or **Hit@K** between model rankings and this reference.  
- This ensures the latent space aligns with expert notion of similarity.

**Status:** Partially addressed. `--mode ground_truth` (`src/phase6_inference/ground_truth.py`) evaluates pseudo ground-truth player pairs via rank checks and Hit@K. `--mode self_consistency` (`src/phase6_inference/self_consistency.py`) tests embedding stability across competition splits and random halves. Together these provide intrinsic validation that nearest neighbours are meaningful. A fully expert-annotated ground truth set remains future work.

### 7. 360 context and “one node per player”

**Status:** SYSTEM_DESIGN, PHASE3, README, and this plan now describe the implementation: **actors** are one node per `player_id` per possession; **off-ball 360** nodes are **one per (event, slot)** in `graph_builder.py` (`("tm", local_ev_idx, k)` / `("opp", local_ev_idx, k)`).

**Improvement (optional):** Use one player node per distinct identity per possession and put **edge attributes on `context_for`**: e.g. `edge_attr = (dx, dy, dist, angle, teammate/opponent flag, optionally is_goalkeeper)` when freeze frames expose stable IDs. That would reduce node count while preserving per-event geometry.

### 8. ~~Situation encoding h_{\text{event}} may not be player-agnostic~~ (IMPLEMENTED)

**Issue:** h_{\text{event}} is used as “the situation” and then different players’ z_p are swapped in Phase 7. That is only valid if h_{\text{event}} does not encode actor identity. Currently, message passing includes `(player, acts_in, event)` and `(event, performed_by, player)`. Even without an explicit player_id on events, player nodes carry position embedding, possession-team flag, and (in the current spec) actor (dx, dy). So h_{\text{event}} can absorb “when the actor is a CB” or similar role/style leakage. Position is masked from event *features*, but it can still enter event nodes via the player→event edges. Phase 7 then risks swapping z_p into a situation embedding that already “knows” who acted.

**Improvement:**  

- **Counterfactual-safe encoding:** When extracting h_{\text{event}} for Phase 7, use an encoder variant that removes or masks `acts_in` / `performed_by` so h_{\text{event}} depends only on context players and temporal history.  
- Or at least: stop-gradient or zero the actor’s player features on the `acts_in` message path so that h_{\text{event}} is not conditioned on actor identity.

**Status:** Implemented via encode_possession_counterfactual in src/phase4_model/model.py. This method drops (player, acts_in, event) edges while retaining (event, performed_by, player) and context_for edges. Used in both the policy diagnostic (src/phase6_inference/policy_diagnostic.py) and Phase 7 situation comparison (src/phase7_analysis/situation_comparison.py). Before/after comparison shows the fix improves substitute quality in 3/4 models (up to +22%) without degrading correlation. See EVALUATION_RESULTS.md Section 7.

### 9. ~~Learning “usage / tactical environment” vs “decision policy”~~ (IMPLEMENTED)

**Issue:** Even with correct masking, z_p is learned by aggregating over the distribution of situations a player experiences. That distribution is confounded by team tactics, league, quality of teammates/opponents, and role. Two players can look similar because they receive the ball in similar zones under similar pressure, not because their *conditional* action choices are similar. Phase 7 (same h_{\text{event}}, swap z_p) partly addresses this, but retrieval is still \cos(z_p, z_q).

**Improvement:** Define a diagnostic over a canonical set of situations S^ *(shared across players) and **policy distance***  
*D(p,q) = \mathbb{E}_{s \sim S^*}[ \mathrm{JS}(\pi_p(\cdot|s), \pi_q(\cdot|s)) ].  
Check correlation between this and cosine distance in z_p. If correlation is weak, the embedding space is not aligned with “decision policy similarity.”

**Status:** Implemented in \src/phase6_inference/policy_diagnostic.py\ (\--mode policy_diagnostic\). Spearman rho = 0.686-0.974 across evaluated GNN variants (see [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) §4), confirming that the embedding space is well-aligned with decision-policy similarity.

### 10. ~~FiLM does not mathematically force use of z_p~~ (PARTIALLY IMPLEMENTED)

**Issue:** The argument “concatenation can be ignored; FiLM cannot” is not strict: the model can learn \gamma(\cdot) \approx 0 and \beta(\cdot) \approx 0, effectively removing dependence on z_p if situation features already predict well. Contrastive and uniformity losses help but uniformity only spreads z_p — it does not guarantee that neighbours reflect policy similarity.

**Improvement:**  

- **Sensitivity:** For fixed h_{\text{event}}, measure how much predicted action distributions change when swapping z_p across players. If the change is tiny, z_p is not load-bearing.  
- **Regulariser:** e.g. predict a baseline policy from h_{\text{event}} only, then a player-specific delta; penalise small deltas only when they hurt likelihood, so that player conditioning must explain residual.
**Status:** Sensitivity diagnostic implemented in `src/phase6_inference/policy_diagnostic.py`. FiLM is confirmed load-bearing (JS > 0 for all groups) in evaluated pipelines. Effect is larger for baseline/split-ctx (0.011-0.013) than player-samp (0.003), reflecting tighter within-group clustering in the latter. See [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) §4 (policy diagnostic). The regulariser suggestion remains unimplemented.

### 11. ~~Contrastive positives may be too scarce~~ (IMPLEMENTED)

**Issue:** InfoNCE works best when each anchor has multiple positives. With “96 possessions per batch,” whether the same player appears multiple times depends on dataset composition; many batches may have only one possession per player ⇒ contrastive signal is weak or noisy.

**Improvement:** Sample batches by player (e.g. sample K players, then M possessions each) so each player has guaranteed positives within the batch; or use a memory queue (MoCo-style) so positives/negatives persist across batches.

**Status:** Implemented via `PlayerAwareBatchSampler` in `src/phase5_training/sampler.py`. Default K=16 players, M=6 possessions each (=96 batch size). Enabled via `--player_sampling` CLI flag. Backward compatible (defaults to `False`, existing baselines unaffected). Produces ~240 contrastive positive pairs per batch vs ~0 with random shuffling.

### 15. ~~Gender-aware evaluation~~ (IMPLEMENTED)

**Issue:** The embedding space mixes male and female players from different competitions (e.g. Euro 2024 and Women's World Cup 2023). Without gender filtering, a female query player could receive male players as top candidates (or vice versa), producing meaningless comparisons.

**Fix:** All similarity-based modes (search, ground truth, self-consistency, policy diagnostic, Phase 7 analysis, FIFA comparison) now filter candidates by gender. Gender is derived from competition metadata (`competitions.json` → `competition_gender`) via `build_gender_map()` in `similarity_search.py`. The filter is applied post-embedding — training and embedding generation are unchanged; only candidate ranking is affected.

**Status:** Implemented across `similarity_search.py`, `ground_truth.py`, `self_consistency.py`, `policy_diagnostic.py`, `report_builder.py`, and `test_fifa_comparison.py`.

### 16. ~~FIFA stat comparison for external validation~~ (IMPLEMENTED)

**Issue:** All evaluation metrics were internal to the GNN system. There was no external validation against independently measured player attributes.

**Fix:** `test_fifa_comparison.py` matches StatsBomb players to FIFA/EA Sports FC data and compares the FIFA stats of query players with their GNN-recommended substitutes. Generates radar charts, scatter plots (cosine similarity vs stat difference with Spearman ρ), per-stat breakdowns, and summary dashboards.

**Status:** Implemented. `match_fifa_players.py` handles the matching; `test_fifa_comparison.py` runs the comparison with visualizations. Integrated into the unified evaluation pipeline via `--mode fifa_comparison`.

### 12. Missing or coarse context (minute, score)

**Issue:** Period is included but not minute; score differential was rejected. Behaviour varies within a half and with scoreline. Omitting them can make the model attribute those effects to “player trait,” inflating z_p with context (e.g. a player who often appears in late-game chasing scenarios may look “direct/vertical” due to context, not style).

**Improvement:**  

- Include **minute** (e.g. normalised within period) in situation features.  
- Include **score differential** only in the situation encoder (e.g. for action prediction), and **exclude it from the path that produces z_p** (or regress/adversarially remove it from z_p) so that similarity is not confounded by score state.

### 13. ~~Substitute-specific evaluation~~ (IMPLEMENTED)

**Issue:** Action accuracy and macro F1 measure “can the model imitate,” not “are nearest neighbours good substitutes.”

**Improvement:** Add evaluation that targets substitution quality:  

- **Counterfactual agreement:** For held-out events, compare KL/JS between predicted action distributions of query vs neighbour in the same h_{\text{event}}. Report mean/median JS for top-k vs random-k.  
- **Top-k retrieval task:** Given held-out events from player p, retrieve a candidate set and score which player’s z best predicts p’s actions on those events.  
- **Stability:** Neighbours should be stable under resampling of possessions; if they change wildly, z_p is not robust.

**Status:** Implemented in `src/phase6_inference/policy_diagnostic.py` (`--mode policy_diagnostic`). Top-K cosine neighbours are 3.07-13.11x more behaviourally similar than random same-group players (ratio metric). See [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) §4.2.

### 14. ~~Early stopping corrupted by auxiliary loss annealing~~ (FIXED)

**Issue:** The validation metric used for early stopping, best-model checkpointing, and LR scheduling was `val_total` — the full combined loss including contrastive and pooled uniformity terms. When `anneal_pooled_weight` is enabled, `lambda_pooled_contrast` ramps up over training (e.g. 0.1 → 0.5). Because pooled uniformity is a *negative* loss (it rewards spread), increasing its weight artificially deflates `val_total` even when supervised losses (action, outcome) stagnate or worsen. This caused the trainer to:

1. Save "best" models based on a misleadingly low total loss rather than actual prediction quality.
2. Delay early stopping because the annealing schedule kept pushing total loss down.
3. Under-trigger LR reduction (`ReduceLROnPlateau`) since the scheduler saw improving loss.

**Fix:** `trainer.py` was refactored so that:

- `_validate()` returns a dictionary of all individual validation losses.
- A **supervised-only** metric `val_supervised = val_action + λ_outcome · val_outcome` (using fixed weights only) is computed and used for early stopping, best-model selection, and `ReduceLROnPlateau.step()`.
- All individual validation losses (action, outcome, contrastive, pooled_uniform) are logged to history and printed per epoch for full visibility.
- The auxiliary losses (contrastive, uniformity) still contribute to the training gradient via `val_total`, but do not influence model selection.

**Status:** Fixed in `src/phase5_training/trainer.py`. Existing checkpoints trained before this fix should be retrained for properly validated model selection.

---

### Assessment against implementation and data

The following notes reflect checks against the actual codebase and StatsBomb data. Use them to prioritise or qualify the items above.


| #   | Verdict                   | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| --- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | **Implemented** (audit #13) | Split into context_for_tm / context_for_opp. GraphConfig.split_context_edges flag enables ablation. See SYSTEM_DESIGN.md audit #13. |
| 2   | **Keep**                  | Time deltas are linear `min(Δt/30, 1)` in `graph_builder.py` and `possession_builder.py`; timestamps come from StatsBomb `timestamp` or `minute`/`second`. Non-linear scaling is a real improvement. Delta time is a noisy proxy for tempo (see limitation in §2).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 3   | **Keep**                  | Phase 7 wording in SYSTEM_DESIGN is indeed strong; reframing as “action preference given identical constraints” is accurate and does not depend on data.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 4   | **Keep**                  | `action_targets.py` uses 9 angle bins and 5 length bins; boundaries can over-penalise. MDN/regression is a valid extension; no data constraint.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 5   | **Keep**                  | `min_samples_per_player = 50` is in config; no confidence or stability score exists. Improvement is consistent with the goal.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 6   | **Partially Implemented** | Pseudo ground-truth evaluation (`ground_truth.py`) and self-consistency tests (`self_consistency.py`) now validate that nearest neighbours are meaningful. Fully expert-annotated ground truth remains future work. |
| 7   | **Done (docs)**           | Specs now match code: context = one node per (event, slot). Optional: fewer nodes via `context_for` edge attributes if deduplicating by ID. |
| 8   | **Implemented**                  | Phase 7 uses `encode_possession(g_dev)` then `h_event[local_ev_idx]`; the full graph (including `acts_in` / `performed_by`) is used, so actor identity can leak into h_{\text{event}}. Counterfactual-safe encoder or stop-gradient is valid.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 9   | **Implemented**           | Policy distance diagnostic implemented in `src/phase6_inference/policy_diagnostic.py`. Overall Spearman rho 0.686-0.974 across evaluated GNN variants (pooled overall; see EVALUATION_RESULTS.md §4).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 10  | **Partially Implemented**                  | FiLM can still learn near-identity; sensitivity and residual regulariser are useful checks. No data dependency.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| 11  | **Implemented**           | `PlayerAwareBatchSampler` (K=16, M=6) guarantees contrastive positive pairs. Enabled via `--player_sampling` CLI flag; backward compatible.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 12  | **Keep (with data note)** | **Minute:** Available in metadata (`possession_builder` and `feature_encoder` use `minute`, `second`, `timestamp`); not currently in the 126-D event vector (only period one-hot is). Can be added. **Score:** Match-level `home_score`/`away_score` exist in data prep; **running score at event time** is not in StatsBomb open data and would need to be computed from goal events if used.                                                                                                                                                                                                                                                                                                                                                                     |
| 13  | **Implemented**           | Substitute-quality diagnostic implemented in `src/phase6_inference/policy_diagnostic.py`. Top-K cosine neighbours are 3.07-13.11x more behaviorally similar than random same-group players. See EVALUATION_RESULTS.md §5.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| 14  | **Fixed**                 | Early stopping / best-model selection was corrupted by annealed auxiliary loss weights deflating `val_total`. Fixed: trainer now uses supervised-only metric (`action + λ_outcome · outcome`) for early stopping and LR scheduling. See `trainer.py`. |
| 15  | **Implemented**           | Gender-aware evaluation: all similarity modes filter candidates by gender derived from competition metadata. Implemented in `similarity_search.py`, `ground_truth.py`, `self_consistency.py`, `policy_diagnostic.py`, `report_builder.py`, `test_fifa_comparison.py`. |
| 16  | **Implemented**           | FIFA stat comparison: external validation via `test_fifa_comparison.py` comparing GNN substitute recommendations against FIFA/EA Sports FC player attributes. Generates radar charts, scatter plots, per-stat breakdowns, and summary dashboards. |


**Summary:** No item is unnecessary or out of scope. The only substantive correction is **#7**: the graph is not “silently wrong” — context geometry is per-event via many context nodes. The improvement is an optional, cleaner design (edge_attr instead of many nodes), and the written spec should be aligned with the implementation. **#14** was a critical training bug discovered post-hoc and has been fixed. Position ablation, EMA alignment, and acts-in dropout variants are available as architectural options; see `PIPELINE_REGISTRY` in `main.py` (currently **11** named pipelines).

---

## State-of-the-art assessment

**Summary:** The system is **state-of-the-art caliber** in design and execution for the specific problem (player similarity defined as “would act similarly in the same situations”), but claiming *the* state of the art would require benchmarks and head-to-head comparisons.

- **Design:** Heterogeneous GNN on possession graphs with 360 context, FiLM conditioning, dual contrastive + uniformity losses, and situation-level counterfactual evaluation is research-grade and comparable to recent work in the space.
- **Limitation:** There is no agreed benchmark for “player similarity from events/360,” and no direct comparison to other published systems on the same data and metric. True SOTA status is established by the community via benchmarks and literature.

---

## Room for improvement

### 1. Model & representation

- **Larger / deeper GNN:** Current 2-layer, 4-head, 64-D setup could be extended (deeper or wider) to capture longer-range dependencies and finer style differences. Trade-off: overfitting and compute.
- **Multi-scale pooling:** Pooling is over full possessions. Adding sub-sequence embeddings (e.g. “first 5 events,” “last 3 before shot”) could yield situation-type embeddings (e.g. “build-up style” vs “final-third style”) alongside the global `z_p`.
- **Explicit outcome/value:** The outcome head exists but does not feed into the embedding. Conditioning `z_p` on “tends to end possessions in high-xG” (e.g. via an auxiliary loss or small value head) could better separate risk-taking vs safe players.

### 2. Training objectives

- ~~**Contrastive temperature:** τ = 0.05 is fixed. A small schedule or learnable temperature could sharpen separation later in training.~~ **Implemented:** Cosine-annealed temperature schedule (e.g. 0.15 → 0.02) via `anneal_temperature` config flag.
- ~~**Pooled weight scheduling:** Fixed λ_pooled weight.~~ **Implemented:** Cosine-annealed `lambda_pooled_contrast` (e.g. 0.1 → 0.5) via `anneal_pooled_weight` config flag. **Note:** Annealing these auxiliary weights exposed a training bug (see "Critical vulnerabilities" #14 below) that has since been fixed.
- ~~**Pooled alignment:** Uniformity spreads everyone apart. A mild **alignment** term for same-player pooled `z_p` across different batch samples (where such pairs exist) could stabilize identity.~~ **Implemented:** EMA cross-batch alignment loss in `losses.py`. An `EMAPlayerMemoryBank` maintains smoothed historical `z_p` per player; `AlignmentLoss = 1 - cos(z_p_current, z_p_ema)` penalizes drift. Enabled via `--ema_alignment` (weight `lambda_alignment=0.3`, momentum 0.999). The `pos_ablated_split_ctx_ema` pipeline uses this. See EVALUATION_RESULTS.md.
- **Action loss weighting:** Beyond Focal + class weights, weighting by “surprise” (e.g. inverse model confidence) could focus the model on harder, more discriminative events.

### 3. Data & labels

- **Finer roles:** Coarse position groups (e.g. “Center Defensive Midfield”) could be refined with role labels (e.g. “single pivot,” “double pivot”) so hard-negative mining pushes apart more similar-looking players.
- **Context conditioning:** Filtering or weighting by game state (e.g. score, minute) could make embeddings more comparable (e.g. “when chasing the game”) and reduce confounds.
- ~~**Multi-match consistency:** A regularizer or loss encouraging the same player’s `z_p` to be stable across matches would improve reliability of similarity over time.~~ **Partially addressed:** The EMA alignment loss regularizes `z_p` stability across batches. Competition-split self-consistency tests confirm improved stability in EMA-enabled models. Full match-level regularization remains future work.

### 4. Evaluation & analysis

- **Human evaluation:** A small study (“Given this situation, who is more similar to the query, A or B?”) compared to model rankings would ground “similar” in human judgment.
- **Downstream tasks:** Use embeddings in a downstream task (e.g. transfer success, team fit, or availability) and compare to heuristics to test real-world utility.
- **Calibration:** Check whether similarity scores or action probabilities are well calibrated and add calibration if needed.

### 5. Operational & product

- **Uncertainty:** Model uncertainty or confidence per embedding (e.g. via ensembles or dropout at inference) would help flag low-confidence similarities.
- **Explainability:** “Player A is similar because they pass more in the final third” (e.g. via attention or feature attribution) would make the system more interpretable and trustworthy.
- **Efficiency:** Smaller distilled model or pruning for deployment; faster similarity search (e.g. ANN indices) if the player set grows.

---

## Document info

- **Created:** For future considerations; reflects post–uniformity-loss and dual-channel position run. Updated post position-ablation and EMA alignment implementation.
- **Last updated:** Post position-ablation, EMA alignment, acts-in dropout presets, and full `full_eval_all` runs. All evaluation pipelines filter by gender. Unified evaluation pipeline (`full_eval`, `full_eval_all`) covers all registered pipelines. See [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) for cross-model analysis.
- **Critical vulnerabilities & blind spots:** Added from external critical analysis; items agreed and transcribed for future work (graph geometry, temporal scaling, counterfactual framing, discretization, evaluation, 360/node semantics, situation encoder leakage, usage vs policy, FiLM sensitivity, contrastive batching, context features, substitute-specific metrics, early-stopping bug #14, gender filtering #15, FIFA comparison #16).
- **See also:** [SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md), [README.md](../README.md), phase docs in this folder.

