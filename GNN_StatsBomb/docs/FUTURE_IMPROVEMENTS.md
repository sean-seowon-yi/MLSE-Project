# Future Improvements & Considerations

This document captures assessment of the current system and concrete directions for improvement, for future reference. It is not a commitment to implement; it is a living list of options.

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

- **Contrastive temperature:** τ = 0.05 is fixed. A small schedule or learnable temperature could sharpen separation later in training.
- **Pooled loss:** Uniformity spreads everyone apart. A mild **alignment** term for same-player pooled `z_p` across different batch samples (where such pairs exist) could stabilize identity.
- **Action loss weighting:** Beyond Focal + class weights, weighting by “surprise” (e.g. inverse model confidence) could focus the model on harder, more discriminative events.

### 3. Data & labels

- **Finer roles:** Coarse position groups (e.g. “Center Defensive Midfield”) could be refined with role labels (e.g. “single pivot,” “double pivot”) so hard-negative mining pushes apart more similar-looking players.
- **Context conditioning:** Filtering or weighting by game state (e.g. score, minute) could make embeddings more comparable (e.g. “when chasing the game”) and reduce confounds.
- **Multi-match consistency:** A regularizer or loss encouraging the same player’s `z_p` to be stable across matches would improve reliability of similarity over time.

### 4. Evaluation & analysis

- **Human evaluation:** A small study (“Given this situation, who is more similar to the query, A or B?”) compared to model rankings would ground “similar” in human judgment.
- **Downstream tasks:** Use embeddings in a downstream task (e.g. transfer success, team fit, or availability) and compare to baselines to test real-world utility.
- **Calibration:** Check whether similarity scores or action probabilities are well calibrated and add calibration if needed.

### 5. Operational & product

- **Uncertainty:** Model uncertainty or confidence per embedding (e.g. via ensembles or dropout at inference) would help flag low-confidence similarities.
- **Explainability:** “Player A is similar because they pass more in the final third” (e.g. via attention or feature attribution) would make the system more interpretable and trustworthy.
- **Efficiency:** Smaller distilled model or pruning for deployment; faster similarity search (e.g. ANN indices) if the player set grows.

---

## Document info

- **Created:** For future considerations; reflects post–uniformity-loss and dual-channel position run.
- **See also:** [SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md), [README.md](../README.md), phase docs in this folder.
