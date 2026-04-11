# Ablation Experiment Report
Generated: 2026-04-07T17:44:17.989511  |  Tag: v4.4-all-modes  |  Pairs: 6393  |  Missing: 0

## Run Parameters
- Mode filter: all
- Split filter: all
- Framework version: 1.0
- Bonferroni threshold: p < 2.6316e-03
- Redundancy threshold: max_cross_r >= 0.85

## Global Clustering Metrics
- K-means ARI (k=2): 0.0314
- PCA PC01 explained variance: 0.447  (cumulative PC05: 0.720)

## Verdict Summary
| Verdict | Count | Features |
|---------|-------|---------|
| KEEP | 4 | entailment, neutral, contradiction, entity_percentage |
| REVIEW | 14 | dice, rouge3, rouge, jaccard, mixedbread-ai/mxbai-embed-large-v1, PREC_mixedbread-ai/mxbai-embed-large-v1, REC_mixedbread-ai/mxbai-embed-large-v1, PREC_Qwen/Qwen3-Embedding-0.6B, entity_location, entity_product, entity_law, entity_duration, lcs_token, lcs_char |
| DROP | 1 | entity_time |

## Tier Summary
| Tier | Count | Features |
|------|-------|---------|
| STRONG | 2 | entailment, contradiction |
| MODERATE | 4 | mixedbread-ai/mxbai-embed-large-v1, PREC_mixedbread-ai/mxbai-embed-large-v1, REC_mixedbread-ai/mxbai-embed-large-v1, PREC_Qwen/Qwen3-Embedding-0.6B |
| WEAK | 8 | dice, rouge3, rouge, jaccard, neutral, entity_percentage, lcs_token, lcs_char |
| MARGINAL | 4 | entity_location, entity_product, entity_law, entity_duration |
| NOISE | 1 | entity_time |

## Per-Feature Detail
(sorted by |Pearson r| descending)

### entailment -- STRONG -- KEEP
- Pearson r: +0.4799  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.4750
- Cohen's d: 1.0936
- Mutual Info: 0.2993 bits
- Max cross-r: 0.6389 with 'neutral' (redundant: no)
- Mode consistency: 1.00
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.4799 (p=0.00e+00, Bonferroni:OK), Spearman r=+0.4750, Cohen's d=1.0936, MI=0.2993 bits, mode_consistency=1.00

### contradiction -- STRONG -- KEEP
- Pearson r: -0.4643  (p=< 1e-10, Bonferroni: OK)
- Spearman r: -0.4558
- Cohen's d: -1.0486
- Mutual Info: 0.2720 bits
- Max cross-r: 0.4629 with 'entailment' (redundant: no)
- Mode consistency: 1.00
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=-0.4643 (p=0.00e+00, Bonferroni:OK), Spearman r=-0.4558, Cohen's d=-1.0486, MI=0.2720 bits, mode_consistency=1.00

### PREC_mixedbread-ai/mxbai-embed-large-v1 -- MODERATE -- REVIEW
- Pearson r: +0.2001  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.2046
- Cohen's d: 0.4084
- Mutual Info: 0.1658 bits
- Max cross-r: 0.9557 with 'mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: 0.67
- **Reason:** Moderate signal but highly correlated with 'mixedbread-ai/mxbai-embed-large-v1'. Ablation test recommended before dropping. Pearson r=+0.2001 (p=9.87e-59, Bonferroni:OK), Spearman r=+0.2046, Cohen's d=0.4084, MI=0.1658 bits, mode_consistency=0.67. REDUNDANT: max_cross_r=0.9557 with 'mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9557)

### REC_mixedbread-ai/mxbai-embed-large-v1 -- MODERATE -- REVIEW
- Pearson r: +0.1899  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1943
- Cohen's d: 0.3867
- Mutual Info: 0.1653 bits
- Max cross-r: 0.9585 with 'mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: 1.00
- **Reason:** Moderate signal but highly correlated with 'mixedbread-ai/mxbai-embed-large-v1'. Ablation test recommended before dropping. Pearson r=+0.1899 (p=5.93e-53, Bonferroni:OK), Spearman r=+0.1943, Cohen's d=0.3867, MI=0.1653 bits, mode_consistency=1.00. REDUNDANT: max_cross_r=0.9585 with 'mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9585)

### mixedbread-ai/mxbai-embed-large-v1 -- MODERATE -- REVIEW
- Pearson r: +0.1893  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1905
- Cohen's d: 0.3855
- Mutual Info: 0.1655 bits
- Max cross-r: 0.9585 with 'REC_mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: 0.67
- **Reason:** Moderate signal but highly correlated with 'REC_mixedbread-ai/mxbai-embed-large-v1'. Ablation test recommended before dropping. Pearson r=+0.1893 (p=1.18e-52, Bonferroni:OK), Spearman r=+0.1905, Cohen's d=0.3855, MI=0.1655 bits, mode_consistency=0.67. REDUNDANT: max_cross_r=0.9585 with 'REC_mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9585)

### PREC_Qwen/Qwen3-Embedding-0.6B -- MODERATE -- REVIEW
- Pearson r: +0.1784  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1853
- Cohen's d: 0.3625
- Mutual Info: 0.1608 bits
- Max cross-r: 0.9082 with 'PREC_mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: 0.67
- **Reason:** Moderate signal but highly correlated with 'PREC_mixedbread-ai/mxbai-embed-large-v1'. Ablation test recommended before dropping. Pearson r=+0.1784 (p=7.55e-47, Bonferroni:OK), Spearman r=+0.1853, Cohen's d=0.3625, MI=0.1608 bits, mode_consistency=0.67. REDUNDANT: max_cross_r=0.9082 with 'PREC_mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9082)

### dice -- WEAK -- REVIEW
- Pearson r: +0.1016  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1032
- Cohen's d: 0.2042
- Mutual Info: 0.0867 bits
- Max cross-r: 0.9833 with 'jaccard' (redundant: yes)
- Mode consistency: 1.00
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'jaccard'. Ablation test before dropping. Pearson r=+0.1016 (p=3.88e-16, Bonferroni:OK), Spearman r=+0.1032, Cohen's d=0.2042, MI=0.0867 bits, mode_consistency=1.00. REDUNDANT: max_cross_r=0.9833 with 'jaccard' (cross_r=+0.9833)

### lcs_token -- WEAK -- REVIEW
- Pearson r: +0.0972  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.0847
- Cohen's d: 0.1953
- Mutual Info: 0.0705 bits
- Max cross-r: 0.9377 with 'jaccard' (redundant: yes)
- Mode consistency: 0.67
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'jaccard'. Ablation test before dropping. Pearson r=+0.0972 (p=6.84e-15, Bonferroni:OK), Spearman r=+0.0847, Cohen's d=0.1953, MI=0.0705 bits, mode_consistency=0.67. REDUNDANT: max_cross_r=0.9377 with 'jaccard' (cross_r=+0.9377)

### neutral -- WEAK -- KEEP
- Pearson r: -0.0964  (p=< 1e-10, Bonferroni: OK)
- Spearman r: -0.0781
- Cohen's d: -0.1937
- Mutual Info: 0.2014 bits
- Max cross-r: 0.6389 with 'entailment' (redundant: no)
- Mode consistency: 0.33
- **Reason:** Small but validated independent signal. Pearson r=-0.0964 (p=1.12e-14, Bonferroni:OK), Spearman r=-0.0781, Cohen's d=-0.1937, MI=0.2014 bits, mode_consistency=0.33

### jaccard -- WEAK -- REVIEW
- Pearson r: +0.0919  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1054
- Cohen's d: 0.1845
- Mutual Info: 0.0574 bits
- Max cross-r: 0.9833 with 'dice' (redundant: yes)
- Mode consistency: 0.33
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'dice'. Ablation test before dropping. Pearson r=+0.0919 (p=1.84e-13, Bonferroni:OK), Spearman r=+0.1054, Cohen's d=0.1845, MI=0.0574 bits, mode_consistency=0.33. REDUNDANT: max_cross_r=0.9833 with 'dice' (cross_r=+0.9833)

### rouge3 -- WEAK -- REVIEW
- Pearson r: +0.0886  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1198
- Cohen's d: 0.1778
- Mutual Info: 0.0548 bits
- Max cross-r: 0.8652 with 'jaccard' (redundant: yes)
- Mode consistency: 0.33
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'jaccard'. Ablation test before dropping. Pearson r=+0.0886 (p=1.31e-12, Bonferroni:OK), Spearman r=+0.1198, Cohen's d=0.1778, MI=0.0548 bits, mode_consistency=0.33. REDUNDANT: max_cross_r=0.8652 with 'jaccard' (cross_r=+0.8652)

### rouge -- WEAK -- REVIEW
- Pearson r: +0.0860  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.0774
- Cohen's d: 0.1725
- Mutual Info: 0.0816 bits
- Max cross-r: 0.9363 with 'dice' (redundant: yes)
- Mode consistency: 0.67
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'dice'. Ablation test before dropping. Pearson r=+0.0860 (p=5.83e-12, Bonferroni:OK), Spearman r=+0.0774, Cohen's d=0.1725, MI=0.0816 bits, mode_consistency=0.67. REDUNDANT: max_cross_r=0.9363 with 'dice' (cross_r=+0.9363)

### entity_percentage -- WEAK -- KEEP
- Pearson r: +0.0660  (p=1.27e-07, Bonferroni: OK)
- Spearman r: +0.0725
- Cohen's d: 0.1323
- Mutual Info: 0.0229 bits
- Max cross-r: 0.0675 with 'entity_duration' (redundant: no)
- Mode consistency: 0.00
- **Reason:** Small but validated independent signal. Pearson r=+0.0660 (p=1.27e-07, Bonferroni:OK), Spearman r=+0.0725, Cohen's d=0.1323, MI=0.0229 bits, mode_consistency=0.00

### lcs_char -- WEAK -- REVIEW
- Pearson r: +0.0588  (p=2.51e-06, Bonferroni: OK)
- Spearman r: +0.0458
- Cohen's d: 0.1178
- Mutual Info: 0.1466 bits
- Max cross-r: 0.9268 with 'lcs_token' (redundant: yes)
- Mode consistency: 0.67
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'lcs_token'. Ablation test before dropping. Pearson r=+0.0588 (p=2.51e-06, Bonferroni:OK), Spearman r=+0.0458, Cohen's d=0.1178, MI=0.1466 bits, mode_consistency=0.67. REDUNDANT: max_cross_r=0.9268 with 'lcs_token' (cross_r=+0.9268)

### entity_product -- MARGINAL -- REVIEW
- Pearson r: +0.0502  (p=5.97e-05, Bonferroni: OK)
- Spearman r: +0.0489
- Cohen's d: 0.1005
- Mutual Info: 0.0000 bits
- Max cross-r: 0.1405 with 'PREC_Qwen/Qwen3-Embedding-0.6B' (redundant: no)
- Mode consistency: 0.00
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0502 (p=5.97e-05, Bonferroni:OK), Spearman r=+0.0489, Cohen's d=0.1005, MI=0.0000 bits, mode_consistency=0.00

### entity_duration -- MARGINAL -- REVIEW
- Pearson r: +0.0195  (p=1.19e-01, Bonferroni: FAIL)
- Spearman r: +0.0201
- Cohen's d: 0.0390
- Mutual Info: 0.0104 bits
- Max cross-r: 0.1391 with 'lcs_char' (redundant: no)
- Mode consistency: 0.00
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0195 (p=1.19e-01, Bonferroni:FAIL), Spearman r=+0.0201, Cohen's d=0.0390, MI=0.0104 bits, mode_consistency=0.00

### entity_law -- MARGINAL -- REVIEW
- Pearson r: +0.0108  (p=3.89e-01, Bonferroni: FAIL)
- Spearman r: +0.0029
- Cohen's d: 0.0215
- Mutual Info: 0.0220 bits
- Max cross-r: 0.0759 with 'lcs_char' (redundant: no)
- Mode consistency: 0.00
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0108 (p=3.89e-01, Bonferroni:FAIL), Spearman r=+0.0029, Cohen's d=0.0215, MI=0.0220 bits, mode_consistency=0.00

### entity_location -- MARGINAL -- REVIEW
- Pearson r: -0.0072  (p=5.66e-01, Bonferroni: FAIL)
- Spearman r: -0.0064
- Cohen's d: -0.0144
- Mutual Info: 0.0049 bits
- Max cross-r: 0.3367 with 'lcs_char' (redundant: no)
- Mode consistency: 0.00
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=-0.0072 (p=5.66e-01, Bonferroni:FAIL), Spearman r=-0.0064, Cohen's d=-0.0144, MI=0.0049 bits, mode_consistency=0.00

### entity_time -- NOISE -- DROP
- Pearson r: +0.0027  (p=8.29e-01, Bonferroni: FAIL)
- Spearman r: +0.0019
- Cohen's d: 0.0054
- Mutual Info: 0.0009 bits
- Max cross-r: 0.2502 with 'lcs_char' (redundant: no)
- Mode consistency: 0.00
- **Reason:** No statistically significant or practically meaningful signal on any measure. Pearson r=+0.0027 (p=8.29e-01, Bonferroni:FAIL), Spearman r=+0.0019, Cohen's d=0.0054, MI=0.0009 bits, mode_consistency=0.00

## PCA Top Loadings (PC01-PC05)
| PC | Feature 1 | Loading | Feature 2 | Loading | Feature 3 | Loading |
|----|-----------|---------|-----------|---------|-----------|---------|
| PC01 | dice | +0.3302 | jaccard | +0.3256 | lcs_token | +0.3174 |
| PC02 | contradiction | +0.5772 | entity_duration | +0.2755 | REC_mixedbread-ai/mxbai-embed-large-v1 | -0.2662 |
| PC03 | neutral | +0.7127 | entailment | -0.6172 | rouge | +0.1393 |
| PC04 | entity_duration | +0.4907 | entity_time | +0.4432 | contradiction | -0.4363 |
| PC05 | entity_percentage | +0.6648 | entity_product | +0.4344 | entity_law | -0.4046 |

## Top-10 K-means Discriminative Features
| Feature | Centroid Distance |
|---------|-----------------|
| dice | 1.7236 |
| lcs_token | 1.6880 |
| jaccard | 1.6838 |
| rouge | 1.6735 |
| lcs_char | 1.6213 |
| rouge3 | 1.4635 |
| mixedbread-ai/mxbai-embed-large-v1 | 1.4500 |
| PREC_Qwen/Qwen3-Embedding-0.6B | 1.4003 |
| PREC_mixedbread-ai/mxbai-embed-large-v1 | 1.3958 |
| REC_mixedbread-ai/mxbai-embed-large-v1 | 1.3861 |

## Per-Mode Pearson r
| Feature | context-vs-generated | model-vs-model | reference-vs-generated |
|---------|---------|---------|---------|
| entailment | +0.4019 | +0.5648 | +0.4921 |
| contradiction | -0.3500 | -0.5238 | -0.5174 |
| PREC_mixedbread-ai/mxbai-embed-large-v1 | -0.0819 | +0.4137 | +0.3300 |
| REC_mixedbread-ai/mxbai-embed-large-v1 | -0.1036 | +0.4039 | +0.3335 |
| mixedbread-ai/mxbai-embed-large-v1 | -0.0829 | +0.3946 | +0.3265 |
| PREC_Qwen/Qwen3-Embedding-0.6B | -0.0645 | +0.3908 | +0.2946 |
| dice | -0.1024 | +0.2290 | +0.1641 |
| lcs_token | -0.1008 | +0.2145 | +0.1380 |
| neutral | -0.0857 | -0.1389 | -0.0670 |
| jaccard | -0.0950 | +0.1942 | +0.1285 |
| rouge3 | -0.0699 | +0.1457 | +0.1404 |
| rouge | -0.1171 | +0.1953 | +0.1768 |
| entity_percentage | +0.0380 | +0.1055 | +0.0467 |
| lcs_char | -0.1288 | +0.2018 | +0.1081 |
| entity_product | +0.0282 | +0.0674 | +0.0574 |
| entity_duration | +0.0057 | +0.0306 | +0.0325 |
| entity_law | -0.0030 | +0.0142 | +0.0239 |
| entity_location | -0.0149 | +0.0115 | -0.0165 |
| entity_time | +0.0106 | +0.0084 | -0.0157 |