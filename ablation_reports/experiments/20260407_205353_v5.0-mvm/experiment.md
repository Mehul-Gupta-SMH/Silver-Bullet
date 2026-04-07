# Ablation Experiment Report
Generated: 2026-04-07T20:53:53.880227  |  Tag: v5.0-mvm  |  Pairs: 2231  |  Missing: 0

## Run Parameters
- Mode filter: model-vs-model
- Split filter: all
- Framework version: 1.0
- Bonferroni threshold: p < 3.5714e-03
- Redundancy threshold: max_cross_r >= 0.85

## Global Clustering Metrics
- K-means ARI (k=2): 0.0712
- PCA PC01 explained variance: 0.585  (cumulative PC05: 0.938)

## Verdict Summary
| Verdict | Count | Features |
|---------|-------|---------|
| KEEP | 9 | rouge3, mixedbread-ai/mxbai-embed-large-v1, PREC_mixedbread-ai/mxbai-embed-large-v1, REC_mixedbread-ai/mxbai-embed-large-v1, PREC_Qwen/Qwen3-Embedding-0.6B, entailment, neutral, contradiction, entity_percentage |
| REVIEW | 5 | dice, rouge, jaccard, lcs_token, lcs_char |
| DROP | 0 | — |

## Tier Summary
| Tier | Count | Features |
|------|-------|---------|
| STRONG | 6 | mixedbread-ai/mxbai-embed-large-v1, PREC_mixedbread-ai/mxbai-embed-large-v1, REC_mixedbread-ai/mxbai-embed-large-v1, PREC_Qwen/Qwen3-Embedding-0.6B, entailment, contradiction |
| MODERATE | 7 | dice, rouge3, rouge, jaccard, neutral, lcs_token, lcs_char |
| WEAK | 1 | entity_percentage |
| MARGINAL | 0 | — |
| NOISE | 0 | — |

## Per-Feature Detail
(sorted by |Pearson r| descending)

### entailment -- STRONG -- KEEP
- Pearson r: +0.5648  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.5674
- Cohen's d: 1.3682
- Mutual Info: 0.2901 bits
- Max cross-r: 0.6101 with 'neutral' (redundant: no)
- Mode consistency: n/a
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.5648 (p=3.22e-188, Bonferroni:OK), Spearman r=+0.5674, Cohen's d=1.3682, MI=0.2901 bits

### contradiction -- STRONG -- KEEP
- Pearson r: -0.5238  (p=< 1e-10, Bonferroni: OK)
- Spearman r: -0.5247
- Cohen's d: -1.2296
- Mutual Info: 0.2600 bits
- Max cross-r: 0.5409 with 'entailment' (redundant: no)
- Mode consistency: n/a
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=-0.5238 (p=1.78e-157, Bonferroni:OK), Spearman r=-0.5247, Cohen's d=-1.2296, MI=0.2600 bits

### PREC_mixedbread-ai/mxbai-embed-large-v1 -- STRONG -- KEEP
- Pearson r: +0.4137  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.3884
- Cohen's d: 0.9085
- Mutual Info: 0.1564 bits
- Max cross-r: 0.9500 with 'mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.4137 (p=5.75e-93, Bonferroni:OK), Spearman r=+0.3884, Cohen's d=0.9085, MI=0.1564 bits. REDUNDANT: max_cross_r=0.9500 with 'mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9500)

### REC_mixedbread-ai/mxbai-embed-large-v1 -- STRONG -- KEEP
- Pearson r: +0.4039  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.3798
- Cohen's d: 0.8828
- Mutual Info: 0.1486 bits
- Max cross-r: 0.9527 with 'mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.4039 (p=2.59e-88, Bonferroni:OK), Spearman r=+0.3798, Cohen's d=0.8828, MI=0.1486 bits. REDUNDANT: max_cross_r=0.9527 with 'mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9527)

### mixedbread-ai/mxbai-embed-large-v1 -- STRONG -- KEEP
- Pearson r: +0.3946  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.3723
- Cohen's d: 0.8585
- Mutual Info: 0.1454 bits
- Max cross-r: 0.9527 with 'REC_mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.3946 (p=5.28e-84, Bonferroni:OK), Spearman r=+0.3723, Cohen's d=0.8585, MI=0.1454 bits. REDUNDANT: max_cross_r=0.9527 with 'REC_mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9527)

### PREC_Qwen/Qwen3-Embedding-0.6B -- STRONG -- KEEP
- Pearson r: +0.3908  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.3694
- Cohen's d: 0.8488
- Mutual Info: 0.1258 bits
- Max cross-r: 0.9099 with 'PREC_mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.3908 (p=2.65e-82, Bonferroni:OK), Spearman r=+0.3694, Cohen's d=0.8488, MI=0.1258 bits. REDUNDANT: max_cross_r=0.9099 with 'PREC_mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9099)

### dice -- MODERATE -- REVIEW
- Pearson r: +0.2290  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.2222
- Cohen's d: 0.4702
- Mutual Info: 0.0858 bits
- Max cross-r: 0.9821 with 'jaccard' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'jaccard'. Ablation test recommended before dropping. Pearson r=+0.2290 (p=6.32e-28, Bonferroni:OK), Spearman r=+0.2222, Cohen's d=0.4702, MI=0.0858 bits. REDUNDANT: max_cross_r=0.9821 with 'jaccard' (cross_r=+0.9821)

### lcs_token -- MODERATE -- REVIEW
- Pearson r: +0.2145  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.2150
- Cohen's d: 0.4390
- Mutual Info: 0.0696 bits
- Max cross-r: 0.9164 with 'lcs_char' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'lcs_char'. Ablation test recommended before dropping. Pearson r=+0.2145 (p=1.25e-24, Bonferroni:OK), Spearman r=+0.2150, Cohen's d=0.4390, MI=0.0696 bits. REDUNDANT: max_cross_r=0.9164 with 'lcs_char' (cross_r=+0.9164)

### lcs_char -- MODERATE -- REVIEW
- Pearson r: +0.2018  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1987
- Cohen's d: 0.4119
- Mutual Info: 0.0303 bits
- Max cross-r: 0.9164 with 'lcs_token' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'lcs_token'. Ablation test recommended before dropping. Pearson r=+0.2018 (p=6.17e-22, Bonferroni:OK), Spearman r=+0.1987, Cohen's d=0.4119, MI=0.0303 bits. REDUNDANT: max_cross_r=0.9164 with 'lcs_token' (cross_r=+0.9164)

### rouge -- MODERATE -- REVIEW
- Pearson r: +0.1953  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1884
- Cohen's d: 0.3980
- Mutual Info: 0.0598 bits
- Max cross-r: 0.9392 with 'dice' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'dice'. Ablation test recommended before dropping. Pearson r=+0.1953 (p=1.31e-20, Bonferroni:OK), Spearman r=+0.1884, Cohen's d=0.3980, MI=0.0598 bits. REDUNDANT: max_cross_r=0.9392 with 'dice' (cross_r=+0.9392)

### jaccard -- MODERATE -- REVIEW
- Pearson r: +0.1942  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.2271
- Cohen's d: 0.3959
- Mutual Info: 0.0821 bits
- Max cross-r: 0.9821 with 'dice' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'dice'. Ablation test recommended before dropping. Pearson r=+0.1942 (p=2.08e-20, Bonferroni:OK), Spearman r=+0.2271, Cohen's d=0.3959, MI=0.0821 bits. REDUNDANT: max_cross_r=0.9821 with 'dice' (cross_r=+0.9821)

### rouge3 -- MODERATE -- KEEP
- Pearson r: +0.1457  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1616
- Cohen's d: 0.2943
- Mutual Info: 0.0112 bits
- Max cross-r: 0.8355 with 'jaccard' (redundant: no)
- Mode consistency: n/a
- **Reason:** Moderate validated signal with independent information. Pearson r=+0.1457 (p=4.75e-12, Bonferroni:OK), Spearman r=+0.1616, Cohen's d=0.2943, MI=0.0112 bits

### neutral -- MODERATE -- KEEP
- Pearson r: -0.1389  (p=< 1e-10, Bonferroni: OK)
- Spearman r: -0.1177
- Cohen's d: -0.2805
- Mutual Info: 0.0892 bits
- Max cross-r: 0.6101 with 'entailment' (redundant: no)
- Mode consistency: n/a
- **Reason:** Moderate validated signal with independent information. Pearson r=-0.1389 (p=4.37e-11, Bonferroni:OK), Spearman r=-0.1177, Cohen's d=-0.2805, MI=0.0892 bits

### entity_percentage -- WEAK -- KEEP
- Pearson r: +0.1055  (p=5.86e-07, Bonferroni: OK)
- Spearman r: +0.1164
- Cohen's d: 0.2122
- Mutual Info: 0.0168 bits
- Max cross-r: 0.0997 with 'mixedbread-ai/mxbai-embed-large-v1' (redundant: no)
- Mode consistency: n/a
- **Reason:** Small but validated independent signal. Pearson r=+0.1055 (p=5.86e-07, Bonferroni:OK), Spearman r=+0.1164, Cohen's d=0.2122, MI=0.0168 bits

## PCA Top Loadings (PC01-PC05)
| PC | Feature 1 | Loading | Feature 2 | Loading | Feature 3 | Loading |
|----|-----------|---------|-----------|---------|-----------|---------|
| PC01 | dice | +0.3300 | jaccard | +0.3248 | lcs_token | +0.3151 |
| PC02 | contradiction | +0.5441 | entailment | -0.4512 | rouge3 | +0.2818 |
| PC03 | neutral | +0.8074 | entailment | -0.4222 | contradiction | -0.3553 |
| PC04 | entity_percentage | +0.9943 | neutral | +0.0768 | entailment | -0.0416 |
| PC05 | contradiction | +0.5314 | entailment | -0.4237 | PREC_mixedbread-ai/mxbai-embed-large-v1 | +0.3335 |

## Top-10 K-means Discriminative Features
| Feature | Centroid Distance |
|---------|-----------------|
| dice | 1.6381 |
| jaccard | 1.6112 |
| lcs_token | 1.5832 |
| rouge | 1.5704 |
| lcs_char | 1.5106 |
| rouge3 | 1.4243 |
| mixedbread-ai/mxbai-embed-large-v1 | 1.3608 |
| PREC_mixedbread-ai/mxbai-embed-large-v1 | 1.2890 |
| REC_mixedbread-ai/mxbai-embed-large-v1 | 1.2877 |
| PREC_Qwen/Qwen3-Embedding-0.6B | 1.2503 |

## Per-Mode Pearson r
| Feature | model-vs-model |
|---------|---------|
| entailment | +0.5648 |
| contradiction | -0.5238 |
| PREC_mixedbread-ai/mxbai-embed-large-v1 | +0.4137 |
| REC_mixedbread-ai/mxbai-embed-large-v1 | +0.4039 |
| mixedbread-ai/mxbai-embed-large-v1 | +0.3946 |
| PREC_Qwen/Qwen3-Embedding-0.6B | +0.3908 |
| dice | +0.2290 |
| lcs_token | +0.2145 |
| lcs_char | +0.2018 |
| rouge | +0.1953 |
| jaccard | +0.1942 |
| rouge3 | +0.1457 |
| neutral | -0.1389 |
| entity_percentage | +0.1055 |