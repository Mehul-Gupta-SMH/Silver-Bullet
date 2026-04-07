# Ablation Experiment Report
Generated: 2026-04-07T20:53:31.169752  |  Tag: v5.0-rvg  |  Pairs: 1831  |  Missing: 0

## Run Parameters
- Mode filter: reference-vs-generated
- Split filter: all
- Framework version: 1.0
- Bonferroni threshold: p < 3.3333e-03
- Redundancy threshold: max_cross_r >= 0.85

## Global Clustering Metrics
- K-means ARI (k=2): 0.0392
- PCA PC01 explained variance: 0.587  (cumulative PC05: 0.895)

## Verdict Summary
| Verdict | Count | Features |
|---------|-------|---------|
| KEEP | 7 | mixedbread-ai/mxbai-embed-large-v1, PREC_mixedbread-ai/mxbai-embed-large-v1, REC_mixedbread-ai/mxbai-embed-large-v1, PREC_Qwen/Qwen3-Embedding-0.6B, entailment, neutral, contradiction |
| REVIEW | 8 | dice, rouge3, rouge, jaccard, entity_product, entity_percentage, lcs_token, lcs_char |
| DROP | 0 | — |

## Tier Summary
| Tier | Count | Features |
|------|-------|---------|
| STRONG | 6 | mixedbread-ai/mxbai-embed-large-v1, PREC_mixedbread-ai/mxbai-embed-large-v1, REC_mixedbread-ai/mxbai-embed-large-v1, PREC_Qwen/Qwen3-Embedding-0.6B, entailment, contradiction |
| MODERATE | 5 | dice, rouge3, rouge, jaccard, lcs_token |
| WEAK | 1 | neutral |
| MARGINAL | 3 | entity_product, entity_percentage, lcs_char |
| NOISE | 0 | — |

## Per-Feature Detail
(sorted by |Pearson r| descending)

### contradiction -- STRONG -- KEEP
- Pearson r: -0.5174  (p=< 1e-10, Bonferroni: OK)
- Spearman r: -0.5250
- Cohen's d: -1.2089
- Mutual Info: 0.2546 bits
- Max cross-r: 0.4634 with 'entailment' (redundant: no)
- Mode consistency: n/a
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=-0.5174 (p=6.43e-126, Bonferroni:OK), Spearman r=-0.5250, Cohen's d=-1.2089, MI=0.2546 bits

### entailment -- STRONG -- KEEP
- Pearson r: +0.4921  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.4572
- Cohen's d: 1.1298
- Mutual Info: 0.2209 bits
- Max cross-r: 0.6440 with 'neutral' (redundant: no)
- Mode consistency: n/a
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.4921 (p=2.88e-112, Bonferroni:OK), Spearman r=+0.4572, Cohen's d=1.1298, MI=0.2209 bits

### REC_mixedbread-ai/mxbai-embed-large-v1 -- STRONG -- KEEP
- Pearson r: +0.3335  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.3002
- Cohen's d: 0.7072
- Mutual Info: 0.0994 bits
- Max cross-r: 0.9761 with 'mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.3335 (p=8.28e-49, Bonferroni:OK), Spearman r=+0.3002, Cohen's d=0.7072, MI=0.0994 bits. REDUNDANT: max_cross_r=0.9761 with 'mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9761)

### PREC_mixedbread-ai/mxbai-embed-large-v1 -- STRONG -- KEEP
- Pearson r: +0.3300  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.2989
- Cohen's d: 0.6989
- Mutual Info: 0.0942 bits
- Max cross-r: 0.9767 with 'mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.3300 (p=9.09e-48, Bonferroni:OK), Spearman r=+0.2989, Cohen's d=0.6989, MI=0.0942 bits. REDUNDANT: max_cross_r=0.9767 with 'mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9767)

### mixedbread-ai/mxbai-embed-large-v1 -- STRONG -- KEEP
- Pearson r: +0.3265  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.2958
- Cohen's d: 0.6905
- Mutual Info: 0.0938 bits
- Max cross-r: 0.9767 with 'PREC_mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.3265 (p=9.88e-47, Bonferroni:OK), Spearman r=+0.2958, Cohen's d=0.6905, MI=0.0938 bits. REDUNDANT: max_cross_r=0.9767 with 'PREC_mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9767)

### PREC_Qwen/Qwen3-Embedding-0.6B -- STRONG -- KEEP
- Pearson r: +0.2946  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.2640
- Cohen's d: 0.6162
- Mutual Info: 0.0652 bits
- Max cross-r: 0.9036 with 'PREC_mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.2946 (p=5.68e-38, Bonferroni:OK), Spearman r=+0.2640, Cohen's d=0.6162, MI=0.0652 bits. REDUNDANT: max_cross_r=0.9036 with 'PREC_mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9036)

### rouge -- MODERATE -- REVIEW
- Pearson r: +0.1768  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1865
- Cohen's d: 0.3592
- Mutual Info: 0.0477 bits
- Max cross-r: 0.9214 with 'dice' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'dice'. Ablation test recommended before dropping. Pearson r=+0.1768 (p=2.50e-14, Bonferroni:OK), Spearman r=+0.1865, Cohen's d=0.3592, MI=0.0477 bits. REDUNDANT: max_cross_r=0.9214 with 'dice' (cross_r=+0.9214)

### dice -- MODERATE -- REVIEW
- Pearson r: +0.1641  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1754
- Cohen's d: 0.3324
- Mutual Info: 0.0779 bits
- Max cross-r: 0.9846 with 'jaccard' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'jaccard'. Ablation test recommended before dropping. Pearson r=+0.1641 (p=1.63e-12, Bonferroni:OK), Spearman r=+0.1754, Cohen's d=0.3324, MI=0.0779 bits. REDUNDANT: max_cross_r=0.9846 with 'jaccard' (cross_r=+0.9846)

### rouge3 -- MODERATE -- REVIEW
- Pearson r: +0.1404  (p=1.61e-09, Bonferroni: OK)
- Spearman r: +0.1645
- Cohen's d: 0.2834
- Mutual Info: 0.0439 bits
- Max cross-r: 0.8532 with 'jaccard' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'jaccard'. Ablation test recommended before dropping. Pearson r=+0.1404 (p=1.61e-09, Bonferroni:OK), Spearman r=+0.1645, Cohen's d=0.2834, MI=0.0439 bits. REDUNDANT: max_cross_r=0.8532 with 'jaccard' (cross_r=+0.8532)

### lcs_token -- MODERATE -- REVIEW
- Pearson r: +0.1380  (p=3.02e-09, Bonferroni: OK)
- Spearman r: +0.1507
- Cohen's d: 0.2786
- Mutual Info: 0.0509 bits
- Max cross-r: 0.9364 with 'lcs_char' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'lcs_char'. Ablation test recommended before dropping. Pearson r=+0.1380 (p=3.02e-09, Bonferroni:OK), Spearman r=+0.1507, Cohen's d=0.2786, MI=0.0509 bits. REDUNDANT: max_cross_r=0.9364 with 'lcs_char' (cross_r=+0.9364)

### jaccard -- MODERATE -- REVIEW
- Pearson r: +0.1285  (p=3.44e-08, Bonferroni: OK)
- Spearman r: +0.1756
- Cohen's d: 0.2590
- Mutual Info: 0.0844 bits
- Max cross-r: 0.9846 with 'dice' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'dice'. Ablation test recommended before dropping. Pearson r=+0.1285 (p=3.44e-08, Bonferroni:OK), Spearman r=+0.1756, Cohen's d=0.2590, MI=0.0844 bits. REDUNDANT: max_cross_r=0.9846 with 'dice' (cross_r=+0.9846)

### lcs_char -- MARGINAL -- REVIEW
- Pearson r: +0.1081  (p=3.56e-06, Bonferroni: OK)
- Spearman r: +0.1041
- Cohen's d: 0.2173
- Mutual Info: 0.0000 bits
- Max cross-r: 0.9364 with 'lcs_token' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.1081 (p=3.56e-06, Bonferroni:OK), Spearman r=+0.1041, Cohen's d=0.2173, MI=0.0000 bits. REDUNDANT: max_cross_r=0.9364 with 'lcs_token' (cross_r=+0.9364)

### neutral -- WEAK -- KEEP
- Pearson r: -0.0670  (p=4.10e-03, Bonferroni: FAIL)
- Spearman r: -0.0349
- Cohen's d: -0.1343
- Mutual Info: 0.0977 bits
- Max cross-r: 0.6440 with 'entailment' (redundant: no)
- Mode consistency: n/a
- **Reason:** Small but validated independent signal. Pearson r=-0.0670 (p=4.10e-03, Bonferroni:FAIL), Spearman r=-0.0349, Cohen's d=-0.1343, MI=0.0977 bits

### entity_product -- MARGINAL -- REVIEW
- Pearson r: +0.0574  (p=1.40e-02, Bonferroni: FAIL)
- Spearman r: +0.0604
- Cohen's d: 0.1149
- Mutual Info: 0.0000 bits
- Max cross-r: 0.1559 with 'PREC_Qwen/Qwen3-Embedding-0.6B' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0574 (p=1.40e-02, Bonferroni:FAIL), Spearman r=+0.0604, Cohen's d=0.1149, MI=0.0000 bits

### entity_percentage -- MARGINAL -- REVIEW
- Pearson r: +0.0467  (p=4.56e-02, Bonferroni: FAIL)
- Spearman r: +0.0580
- Cohen's d: 0.0935
- Mutual Info: 0.0139 bits
- Max cross-r: 0.0827 with 'neutral' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0467 (p=4.56e-02, Bonferroni:FAIL), Spearman r=+0.0580, Cohen's d=0.0935, MI=0.0139 bits

## PCA Top Loadings (PC01-PC05)
| PC | Feature 1 | Loading | Feature 2 | Loading | Feature 3 | Loading |
|----|-----------|---------|-----------|---------|-----------|---------|
| PC01 | dice | +0.3237 | jaccard | +0.3186 | lcs_token | +0.3119 |
| PC02 | contradiction | +0.7348 | neutral | -0.3138 | entailment | -0.3071 |
| PC03 | neutral | +0.6507 | entailment | -0.5329 | entity_percentage | -0.3077 |
| PC04 | entity_percentage | +0.7427 | entity_product | +0.6360 | entailment | -0.1548 |
| PC05 | entity_product | +0.7512 | entity_percentage | -0.5927 | neutral | -0.1996 |

## Top-10 K-means Discriminative Features
| Feature | Centroid Distance |
|---------|-----------------|
| dice | 1.6958 |
| jaccard | 1.6582 |
| lcs_token | 1.6406 |
| lcs_char | 1.5703 |
| rouge | 1.5568 |
| mixedbread-ai/mxbai-embed-large-v1 | 1.4591 |
| PREC_mixedbread-ai/mxbai-embed-large-v1 | 1.4339 |
| REC_mixedbread-ai/mxbai-embed-large-v1 | 1.4321 |
| rouge3 | 1.4304 |
| PREC_Qwen/Qwen3-Embedding-0.6B | 1.4243 |

## Per-Mode Pearson r
| Feature | reference-vs-generated |
|---------|---------|
| contradiction | -0.5174 |
| entailment | +0.4921 |
| REC_mixedbread-ai/mxbai-embed-large-v1 | +0.3335 |
| PREC_mixedbread-ai/mxbai-embed-large-v1 | +0.3300 |
| mixedbread-ai/mxbai-embed-large-v1 | +0.3265 |
| PREC_Qwen/Qwen3-Embedding-0.6B | +0.2946 |
| rouge | +0.1768 |
| dice | +0.1641 |
| rouge3 | +0.1404 |
| lcs_token | +0.1380 |
| jaccard | +0.1285 |
| lcs_char | +0.1081 |
| neutral | -0.0670 |
| entity_product | +0.0574 |
| entity_percentage | +0.0467 |