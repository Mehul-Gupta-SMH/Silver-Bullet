# SilverBullet Feature Ablation Framework

**Version:** 1.0  
**Last updated:** 2026-04-01  
**Script:** `backend/ablation_cluster.py`  
**Experiment logs:** `ablation_reports/experiments/{timestamp}_{tag}/`

---

## Purpose

This framework defines a reproducible, quantitative process for deciding whether each feature map in the SilverBullet pipeline is contributing genuine signal or is noise. Every run produces a timestamped experiment directory with a full JSON report and human-readable Markdown summary.

The goal is not to blindly drop features — it is to understand *why* each feature is or is not discriminative, and to record that reasoning so it can be revisited when the training data or feature pipeline changes.

---

## The Six Measures

Each feature is evaluated on six independent axes. No single measure is sufficient; the tier assignment (below) requires multiple conditions to hold simultaneously.

### Measure A — Pearson r (feature vs label)

Linear monotonic association between the spatial mean of the feature map and the binary label (0/1). Bonferroni correction applied at `alpha=0.05 / n_features`.

- **Strength:** captures linear signal directly
- **Weakness:** sensitive to outliers; misses non-linear structure
- **Code:** `scipy.stats.pearsonr`

### Measure B — Spearman r (feature vs label)

Rank-based monotonic association. Robust to skewed distributions (entity maps are heavily zero-inflated — most pairs have no entities of a given type, producing a spike at 1.0 from the "both zero = agree" rule).

- **Strength:** handles non-normal, zero-inflated distributions
- **Weakness:** still misses non-monotonic relationships
- **Code:** `scipy.stats.spearmanr`

### Measure C — Cohen's d (label=0 vs label=1 groups)

Standardised mean difference between the label=0 and label=1 distributions of each feature. Effect size in standard-deviation units, independent of sample size (unlike p-values, which grow with N).

- **Formula:** `(mean_label1 - mean_label0) / pooled_std`
- **Pooled std:** `sqrt((std0^2 + std1^2) / 2)` (Welch formula)
- **Zero-variance guard:** if `pooled_std < 1e-12`, set `d = 0.0` and note `cohens_d_note: zero_variance`
- **Strength:** interpretable regardless of N; reveals practical significance
- **Code:** computed from numpy arrays

### Measure D — Mutual Information (feature vs label)

Non-parametric, non-linear association using a k-nearest-neighbour entropy estimator. Captures any statistical dependence, including non-monotonic relationships (e.g., a feature that peaks at both extremes).

- **Units:** bits (divide raw nats by ln(2))
- **Theoretical max:** 1 bit for a perfectly predictive binary label
- **Practical range:** 0.0 (no information) to ~0.3 bits for strong features
- **Code:** `sklearn.feature_selection.mutual_info_classif(discrete_features=False, n_neighbors=3)`

### Measure E — Pairwise Feature Redundancy (cross-correlation)

For each feature, the maximum absolute Pearson r it has with any *other* feature. Identifies features that are near-proxies of stronger features — keeping both adds no information to the model but does add noise channels.

- **Redundancy threshold:** `max_cross_r >= 0.85`
- **Output:** `max_cross_r` scalar + `top3_correlated_peers` list
- **Code:** `np.corrcoef(X.T)` over the Nx34 feature matrix

### Measure F — Mode Consistency Score

For each of the 3 evaluation modes (context-vs-generated, reference-vs-generated, model-vs-model), features are ranked by `|Pearson r|` within that mode's data subset. Mode consistency = fraction of modes where the feature ranks in the top half (rank <= 17 of 34).

- **Values:** 0.0 (no mode top-half), 0.33, 0.67, 1.0 (all modes top-half)
- **Purpose:** a feature with high global correlation but only in one mode may be overfitting that mode's distribution
- **Fallback:** if only one mode is present (e.g., `--mode` flag used), mode consistency is set to `null` and not used in tier assignment

---

## Tier Definitions

Tiers are assigned by checking conditions from STRONG downward. A feature takes the highest tier whose full condition set is satisfied. All conditions within a tier must hold simultaneously (AND logic), except MARGINAL which uses OR.

### STRONG

| Measure | Threshold |
|---------|-----------|
| abs(Pearson r) | >= 0.25 |
| Pearson p | < Bonferroni threshold (p < 0.05/n_features) |
| abs(Spearman r) | >= 0.25 |
| abs(Cohen's d) | >= 0.50 (medium effect) |
| Mutual Info | >= 0.020 bits |
| Mode consistency | >= 0.67 (top-half in >= 2 of 3 modes) |

Expected features: `entailment`, `contradiction` (Pearson ~0.49, Cohen's d ~1.0+).

### MODERATE

| Measure | Threshold |
|---------|-----------|
| abs(Pearson r) | >= 0.10 |
| Pearson p | < Bonferroni threshold |
| abs(Spearman r) | >= 0.10 |
| abs(Cohen's d) | >= 0.25 (small-to-medium) |
| Mutual Info | >= 0.008 bits |
| Mode consistency | >= 0.33 (top-half in >= 1 mode) |

Expected features: all semantic PREC/REC/cosine maps (mxbai + Qwen3), lexical features (dice/jaccard/rouge/lcs), `neutral`.

### WEAK

| Measure | Threshold |
|---------|-----------|
| abs(Pearson r) | >= 0.05 |
| Pearson p | < 0.05 (uncorrected — not Bonferroni) |
| abs(Cohen's d) | >= 0.10 |
| Mutual Info | >= 0.002 bits |

Note: Bonferroni correction is NOT required here. With N~6000, the bar to reach p<0.05 for r=0.05 is already cleared; this tier captures genuine but small signal that doesn't survive the strict multiple-comparison correction.

Expected features: `entity_percentage`, `entity_product`, `entity_law`, `entity_time`, `entity_location`, `entity_duration`.

### MARGINAL

Any ONE of:
- abs(Pearson r) >= 0.02
- abs(Cohen's d) >= 0.05
- Mutual Info >= 0.001 bits

Features here have a detectable numeric association that doesn't meet the statistical or practical threshold for WEAK. They are not confirmed noise but are not validated signal either.

Expected features: `entity_language`, `entity_quantity`, `entity_number`.

### NOISE

Everything that does not meet MARGINAL conditions. With N=6293, even r=0.02 yields p~0.12, so features failing to reach even r=0.02 are statistically indistinguishable from random noise.

Expected features: `entity_date`, `entity_money`, `entity_person`, `entity_event`, `entity_organization`, `SOFT_ROW_*`, `SOFT_COL_*`.

---

## Verdict Rules

Evaluated in order. Verdicts are: **KEEP**, **REVIEW**, **DROP**.

```
IF tier == NOISE:
    verdict = DROP

ELSE IF tier == MARGINAL:
    verdict = REVIEW
    (retain only if domain knowledge justifies inclusion)

ELSE IF tier == WEAK AND redundant:
    verdict = REVIEW
    (near-duplicate of a higher-tier feature; ablation test before dropping)

ELSE IF tier == WEAK AND NOT redundant:
    verdict = KEEP
    (small but validated independent signal)

ELSE IF tier == MODERATE AND redundant:
    verdict = REVIEW
    (moderate signal but highly correlated with a peer; ablation test)

ELSE IF tier == MODERATE AND NOT redundant:
    verdict = KEEP

ELSE IF tier == STRONG:
    verdict = KEEP
    (regardless of redundancy — strong features are always kept)
```

The `reason` field in the report records the specific numeric values that determined the tier and verdict, naming any redundant peer by name.

---

## Redundancy Criterion

A feature is **redundant** if its `max_cross_r >= 0.85` with any other feature in the set. This threshold means the two features share >72% of their variance (`r^2 = 0.72`). At this level of collinearity, a CNN cannot reliably distinguish the two channels and one can be removed without loss.

Redundancy is a separate axis from the tier — it modifies the verdict but does not lower the tier label. Both the tier (signal quality) and redundancy (information uniqueness) are preserved independently in the report.

---

## Interpreting Results in Context

### On p-values and sample size

With N=6293, even r=0.025 yields p~0.045 (uncorrected). The Bonferroni threshold (~0.00147 for 34 features) is the appropriate significance gate. Features that survive Bonferroni have genuine associations. Features that fail it may still have real signal — but the claim cannot be made confidently from this dataset alone.

### On the sampling argument

Entity features may appear weak on the current dataset because most external training pairs (STS-B, MNLI, QQP, QNLI) contain few named entities. The `_type_agreement()` function returns 1.0 when both texts have zero entities of a type — so near-uniform entity maps are expected on entity-sparse data. To test the sampling hypothesis: filter to pairs where at least one entity of the relevant type appears in either text, then recheck the correlation. If it rises substantially, the feature is conditionally useful and worth retaining for domain-specific deployment.

### On SOFT_ROW/SOFT_COL

These features have p=0.47-0.80 on N=6293 — far too high to attribute to sampling. A genuine signal of r=0.03 would be detectable (p~0.017) at this N. The soft alignment maps are structurally capturing variance in the data, but that variance is orthogonal to the label. They are structural noise, not sampling noise.

### On ARI

K-means ARI of ~0.06 means the 34-feature space barely separates label classes unsupervised. This is expected — the CNN learns a non-linear decision boundary across all 34 channels jointly. ARI measures only whether simple Euclidean clustering in scaled feature-mean space aligns with labels, which is a much weaker task than what the CNN does. A low ARI does not mean the features are useless; it means linear clustering on spatial means is insufficient.

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-04-01 | Initial framework — 6 measures, 5 tiers, 3 verdicts; calibrated against 34-feature v3.0 pipeline on 6293 pairs |
