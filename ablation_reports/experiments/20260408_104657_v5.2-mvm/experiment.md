# Ablation Experiment Report
Generated: 2026-04-08T10:46:57.032958  |  Tag: v5.2-mvm  |  Pairs: 2231  |  Missing: 0

## Run Parameters
- Mode filter: model-vs-model
- Split filter: all
- Framework version: 1.0
- Bonferroni threshold: p < 2.1739e-03
- Redundancy threshold: max_cross_r >= 0.85

## Global Clustering Metrics
- K-means ARI (k=2): 0.0771
- PCA PC01 explained variance: 0.381  (cumulative PC05: 0.734)

## Verdict Summary
| Verdict | Count | Features |
|---------|-------|---------|
| KEEP | 14 | rouge3, mixedbread-ai/mxbai-embed-large-v1, PREC_mixedbread-ai/mxbai-embed-large-v1, REC_mixedbread-ai/mxbai-embed-large-v1, PREC_Qwen/Qwen3-Embedding-0.6B, entailment, neutral, contradiction, entity_percentage, entity_value_prec, entity_value_rec, numeric_jaccard, entity_percentage_value_prec, entity_percentage_value_rec |
| REVIEW | 7 | dice, rouge, jaccard, lcs_token, lcs_char, entity_location_value_prec, entity_location_value_rec |
| DROP | 2 | entity_time_value_prec, entity_time_value_rec |

## Tier Summary
| Tier | Count | Features |
|------|-------|---------|
| STRONG | 6 | mixedbread-ai/mxbai-embed-large-v1, PREC_mixedbread-ai/mxbai-embed-large-v1, REC_mixedbread-ai/mxbai-embed-large-v1, PREC_Qwen/Qwen3-Embedding-0.6B, entailment, contradiction |
| MODERATE | 8 | dice, rouge3, rouge, jaccard, neutral, entity_value_rec, lcs_token, lcs_char |
| WEAK | 5 | entity_percentage, entity_value_prec, numeric_jaccard, entity_percentage_value_prec, entity_percentage_value_rec |
| MARGINAL | 2 | entity_location_value_prec, entity_location_value_rec |
| NOISE | 2 | entity_time_value_prec, entity_time_value_rec |

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
- Mutual Info: 0.1565 bits
- Max cross-r: 0.9500 with 'mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.4137 (p=5.75e-93, Bonferroni:OK), Spearman r=+0.3884, Cohen's d=0.9085, MI=0.1565 bits. REDUNDANT: max_cross_r=0.9500 with 'mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9500)

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
- Mutual Info: 0.1259 bits
- Max cross-r: 0.9099 with 'PREC_mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.3908 (p=2.65e-82, Bonferroni:OK), Spearman r=+0.3694, Cohen's d=0.8488, MI=0.1259 bits. REDUNDANT: max_cross_r=0.9099 with 'PREC_mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9099)

### dice -- MODERATE -- REVIEW
- Pearson r: +0.2290  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.2222
- Cohen's d: 0.4702
- Mutual Info: 0.0742 bits
- Max cross-r: 0.9821 with 'jaccard' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'jaccard'. Ablation test recommended before dropping. Pearson r=+0.2290 (p=6.32e-28, Bonferroni:OK), Spearman r=+0.2222, Cohen's d=0.4702, MI=0.0742 bits. REDUNDANT: max_cross_r=0.9821 with 'jaccard' (cross_r=+0.9821)

### lcs_token -- MODERATE -- REVIEW
- Pearson r: +0.2145  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.2150
- Cohen's d: 0.4390
- Mutual Info: 0.0432 bits
- Max cross-r: 0.9164 with 'lcs_char' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'lcs_char'. Ablation test recommended before dropping. Pearson r=+0.2145 (p=1.25e-24, Bonferroni:OK), Spearman r=+0.2150, Cohen's d=0.4390, MI=0.0432 bits. REDUNDANT: max_cross_r=0.9164 with 'lcs_char' (cross_r=+0.9164)

### lcs_char -- MODERATE -- REVIEW
- Pearson r: +0.2018  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1987
- Cohen's d: 0.4119
- Mutual Info: 0.0334 bits
- Max cross-r: 0.9164 with 'lcs_token' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'lcs_token'. Ablation test recommended before dropping. Pearson r=+0.2018 (p=6.17e-22, Bonferroni:OK), Spearman r=+0.1987, Cohen's d=0.4119, MI=0.0334 bits. REDUNDANT: max_cross_r=0.9164 with 'lcs_token' (cross_r=+0.9164)

### rouge -- MODERATE -- REVIEW
- Pearson r: +0.1953  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1884
- Cohen's d: 0.3980
- Mutual Info: 0.0250 bits
- Max cross-r: 0.9392 with 'dice' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'dice'. Ablation test recommended before dropping. Pearson r=+0.1953 (p=1.31e-20, Bonferroni:OK), Spearman r=+0.1884, Cohen's d=0.3980, MI=0.0250 bits. REDUNDANT: max_cross_r=0.9392 with 'dice' (cross_r=+0.9392)

### jaccard -- MODERATE -- REVIEW
- Pearson r: +0.1942  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.2271
- Cohen's d: 0.3959
- Mutual Info: 0.0872 bits
- Max cross-r: 0.9821 with 'dice' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'dice'. Ablation test recommended before dropping. Pearson r=+0.1942 (p=2.08e-20, Bonferroni:OK), Spearman r=+0.2271, Cohen's d=0.3959, MI=0.0872 bits. REDUNDANT: max_cross_r=0.9821 with 'dice' (cross_r=+0.9821)

### entity_value_rec -- MODERATE -- KEEP
- Pearson r: +0.1766  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1742
- Cohen's d: 0.3586
- Mutual Info: 0.0161 bits
- Max cross-r: 0.7061 with 'entity_value_prec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Moderate validated signal with independent information. Pearson r=+0.1766 (p=4.42e-17, Bonferroni:OK), Spearman r=+0.1742, Cohen's d=0.3586, MI=0.0161 bits

### entity_value_prec -- WEAK -- KEEP
- Pearson r: +0.1551  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1524
- Cohen's d: 0.3140
- Mutual Info: 0.0026 bits
- Max cross-r: 0.7061 with 'entity_value_rec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Small but validated independent signal. Pearson r=+0.1551 (p=1.72e-13, Bonferroni:OK), Spearman r=+0.1524, Cohen's d=0.3140, MI=0.0026 bits

### rouge3 -- MODERATE -- KEEP
- Pearson r: +0.1457  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1616
- Cohen's d: 0.2943
- Mutual Info: 0.0316 bits
- Max cross-r: 0.8355 with 'jaccard' (redundant: no)
- Mode consistency: n/a
- **Reason:** Moderate validated signal with independent information. Pearson r=+0.1457 (p=4.75e-12, Bonferroni:OK), Spearman r=+0.1616, Cohen's d=0.2943, MI=0.0316 bits

### neutral -- MODERATE -- KEEP
- Pearson r: -0.1389  (p=< 1e-10, Bonferroni: OK)
- Spearman r: -0.1177
- Cohen's d: -0.2805
- Mutual Info: 0.0892 bits
- Max cross-r: 0.6101 with 'entailment' (redundant: no)
- Mode consistency: n/a
- **Reason:** Moderate validated signal with independent information. Pearson r=-0.1389 (p=4.37e-11, Bonferroni:OK), Spearman r=-0.1177, Cohen's d=-0.2805, MI=0.0892 bits

### numeric_jaccard -- WEAK -- KEEP
- Pearson r: +0.1354  (p=1.34e-10, Bonferroni: OK)
- Spearman r: +0.1304
- Cohen's d: 0.2733
- Mutual Info: 0.0028 bits
- Max cross-r: 0.3836 with 'entity_percentage_value_rec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Small but validated independent signal. Pearson r=+0.1354 (p=1.34e-10, Bonferroni:OK), Spearman r=+0.1304, Cohen's d=0.2733, MI=0.0028 bits

### entity_percentage_value_rec -- WEAK -- KEEP
- Pearson r: +0.1202  (p=1.23e-08, Bonferroni: OK)
- Spearman r: +0.1201
- Cohen's d: 0.2421
- Mutual Info: 0.0236 bits
- Max cross-r: 0.8210 with 'entity_percentage' (redundant: no)
- Mode consistency: n/a
- **Reason:** Small but validated independent signal. Pearson r=+0.1202 (p=1.23e-08, Bonferroni:OK), Spearman r=+0.1201, Cohen's d=0.2421, MI=0.0236 bits

### entity_percentage_value_prec -- WEAK -- KEEP
- Pearson r: +0.1068  (p=4.25e-07, Bonferroni: OK)
- Spearman r: +0.1131
- Cohen's d: 0.2148
- Mutual Info: 0.0218 bits
- Max cross-r: 0.8101 with 'entity_percentage_value_rec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Small but validated independent signal. Pearson r=+0.1068 (p=4.25e-07, Bonferroni:OK), Spearman r=+0.1131, Cohen's d=0.2148, MI=0.0218 bits

### entity_percentage -- WEAK -- KEEP
- Pearson r: +0.1055  (p=5.86e-07, Bonferroni: OK)
- Spearman r: +0.1164
- Cohen's d: 0.2122
- Mutual Info: 0.0205 bits
- Max cross-r: 0.8210 with 'entity_percentage_value_rec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Small but validated independent signal. Pearson r=+0.1055 (p=5.86e-07, Bonferroni:OK), Spearman r=+0.1164, Cohen's d=0.2122, MI=0.0205 bits

### entity_location_value_rec -- MARGINAL -- REVIEW
- Pearson r: +0.0527  (p=1.28e-02, Bonferroni: FAIL)
- Spearman r: +0.0501
- Cohen's d: 0.1055
- Mutual Info: 0.0000 bits
- Max cross-r: 0.7610 with 'entity_location_value_prec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0527 (p=1.28e-02, Bonferroni:FAIL), Spearman r=+0.0501, Cohen's d=0.1055, MI=0.0000 bits

### entity_location_value_prec -- MARGINAL -- REVIEW
- Pearson r: +0.0446  (p=3.51e-02, Bonferroni: FAIL)
- Spearman r: +0.0344
- Cohen's d: 0.0893
- Mutual Info: 0.0327 bits
- Max cross-r: 0.7610 with 'entity_location_value_rec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0446 (p=3.51e-02, Bonferroni:FAIL), Spearman r=+0.0344, Cohen's d=0.0893, MI=0.0327 bits

### entity_time_value_prec -- NOISE -- DROP
- Pearson r: -0.0020  (p=9.27e-01, Bonferroni: FAIL)
- Spearman r: +0.0068
- Cohen's d: -0.0039
- Mutual Info: 0.0000 bits
- Max cross-r: 0.7103 with 'entity_time_value_rec' (redundant: no)
- Mode consistency: n/a
- **Reason:** No statistically significant or practically meaningful signal on any measure. Pearson r=-0.0020 (p=9.27e-01, Bonferroni:FAIL), Spearman r=+0.0068, Cohen's d=-0.0039, MI=0.0000 bits

### entity_time_value_rec -- NOISE -- DROP
- Pearson r: -0.0012  (p=9.57e-01, Bonferroni: FAIL)
- Spearman r: +0.0072
- Cohen's d: -0.0023
- Mutual Info: 0.0000 bits
- Max cross-r: 0.7103 with 'entity_time_value_prec' (redundant: no)
- Mode consistency: n/a
- **Reason:** No statistically significant or practically meaningful signal on any measure. Pearson r=-0.0012 (p=9.57e-01, Bonferroni:FAIL), Spearman r=+0.0072, Cohen's d=-0.0023, MI=0.0000 bits

## PCA Top Loadings (PC01-PC05)
| PC | Feature 1 | Loading | Feature 2 | Loading | Feature 3 | Loading |
|----|-----------|---------|-----------|---------|-----------|---------|
| PC01 | dice | +0.3137 | jaccard | +0.3089 | mixedbread-ai/mxbai-embed-large-v1 | +0.3022 |
| PC02 | entity_percentage_value_rec | +0.5490 | entity_percentage_value_prec | +0.4736 | entity_percentage | +0.4598 |
| PC03 | entity_location_value_prec | +0.4932 | entity_location_value_rec | +0.4748 | entity_value_prec | +0.3526 |
| PC04 | entity_time_value_prec | +0.5661 | entity_time_value_rec | +0.5637 | contradiction | +0.3509 |
| PC05 | contradiction | +0.4419 | entailment | -0.3571 | entity_location_value_rec | +0.3557 |

## Top-10 K-means Discriminative Features
| Feature | Centroid Distance |
|---------|-----------------|
| dice | 1.6104 |
| jaccard | 1.5875 |
| rouge | 1.5372 |
| lcs_token | 1.5359 |
| lcs_char | 1.4712 |
| rouge3 | 1.3918 |
| mixedbread-ai/mxbai-embed-large-v1 | 1.3690 |
| PREC_mixedbread-ai/mxbai-embed-large-v1 | 1.2911 |
| REC_mixedbread-ai/mxbai-embed-large-v1 | 1.2877 |
| PREC_Qwen/Qwen3-Embedding-0.6B | 1.2538 |

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
| entity_value_rec | +0.1766 |
| entity_value_prec | +0.1551 |
| rouge3 | +0.1457 |
| neutral | -0.1389 |
| numeric_jaccard | +0.1354 |
| entity_percentage_value_rec | +0.1202 |
| entity_percentage_value_prec | +0.1068 |
| entity_percentage | +0.1055 |
| entity_location_value_rec | +0.0527 |
| entity_location_value_prec | +0.0446 |
| entity_time_value_prec | -0.0020 |
| entity_time_value_rec | -0.0012 |