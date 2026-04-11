# Ablation Experiment Report
Generated: 2026-04-02T19:15:07.137737  |  Tag: v4.0a-all-modes  |  Pairs: 6293  |  Missing: 0

## Run Parameters
- Mode filter: all
- Split filter: all
- Framework version: 1.0
- Bonferroni threshold: p < 1.6667e-03
- Redundancy threshold: max_cross_r >= 0.85

## Global Clustering Metrics
- K-means ARI (k=2): 0.0625
- PCA PC01 explained variance: 0.372  (cumulative PC05: 0.575)

## Verdict Summary
| Verdict | Count | Features |
|---------|-------|---------|
| KEEP | 9 | rouge3, mixedbread-ai/mxbai-embed-large-v1, PREC_mixedbread-ai/mxbai-embed-large-v1, REC_mixedbread-ai/mxbai-embed-large-v1, PREC_Qwen/Qwen3-Embedding-0.6B, entailment, neutral, contradiction, entity_percentage |
| REVIEW | 18 | jaccard, dice, cosine, rouge, Qwen/Qwen3-Embedding-0.6B, REC_Qwen/Qwen3-Embedding-0.6B, entity_person, entity_location, entity_product, entity_law, entity_language, entity_date, entity_time, entity_duration, entity_number, entity_quantity, lcs_token, lcs_char |
| DROP | 3 | entity_organization, entity_event, entity_money |

## Tier Summary
| Tier | Count | Features |
|------|-------|---------|
| STRONG | 6 | mixedbread-ai/mxbai-embed-large-v1, PREC_mixedbread-ai/mxbai-embed-large-v1, REC_mixedbread-ai/mxbai-embed-large-v1, PREC_Qwen/Qwen3-Embedding-0.6B, entailment, contradiction |
| MODERATE | 7 | jaccard, dice, cosine, rouge, Qwen/Qwen3-Embedding-0.6B, REC_Qwen/Qwen3-Embedding-0.6B, lcs_token |
| WEAK | 4 | rouge3, neutral, entity_percentage, lcs_char |
| MARGINAL | 10 | entity_person, entity_location, entity_product, entity_law, entity_language, entity_date, entity_time, entity_duration, entity_number, entity_quantity |
| NOISE | 3 | entity_organization, entity_event, entity_money |

## Per-Feature Detail
(sorted by |Pearson r| descending)

### entailment -- STRONG -- KEEP
- Pearson r: +0.4904  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.4878
- Cohen's d: 1.1251
- Mutual Info: 0.3753 bits
- Max cross-r: 0.6380 with 'neutral' (redundant: no)
- Mode consistency: 1.00
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.4904 (p=0.00e+00, Bonferroni:OK), Spearman r=+0.4878, Cohen's d=1.1251, MI=0.3753 bits, mode_consistency=1.00

### contradiction -- STRONG -- KEEP
- Pearson r: -0.4826  (p=< 1e-10, Bonferroni: OK)
- Spearman r: -0.4670
- Cohen's d: -1.1021
- Mutual Info: 0.3814 bits
- Max cross-r: 0.4801 with 'entailment' (redundant: no)
- Mode consistency: 1.00
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=-0.4826 (p=0.00e+00, Bonferroni:OK), Spearman r=-0.4670, Cohen's d=-1.1021, MI=0.3814 bits, mode_consistency=1.00

### PREC_mixedbread-ai/mxbai-embed-large-v1 -- STRONG -- KEEP
- Pearson r: +0.2928  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.2993
- Cohen's d: 0.6123
- Mutual Info: 0.2754 bits
- Max cross-r: 0.9661 with 'mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: 1.00
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.2928 (p=1.34e-124, Bonferroni:OK), Spearman r=+0.2993, Cohen's d=0.6123, MI=0.2754 bits, mode_consistency=1.00. REDUNDANT: max_cross_r=0.9661 with 'mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9661)

### REC_mixedbread-ai/mxbai-embed-large-v1 -- STRONG -- KEEP
- Pearson r: +0.2813  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.2873
- Cohen's d: 0.5863
- Mutual Info: 0.2605 bits
- Max cross-r: 0.9670 with 'mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: 1.00
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.2813 (p=8.12e-115, Bonferroni:OK), Spearman r=+0.2873, Cohen's d=0.5863, MI=0.2605 bits, mode_consistency=1.00. REDUNDANT: max_cross_r=0.9670 with 'mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9670)

### mixedbread-ai/mxbai-embed-large-v1 -- STRONG -- KEEP
- Pearson r: +0.2729  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.2778
- Cohen's d: 0.5674
- Mutual Info: 0.2685 bits
- Max cross-r: 0.9670 with 'REC_mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: 1.00
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.2729 (p=6.32e-108, Bonferroni:OK), Spearman r=+0.2778, Cohen's d=0.5674, MI=0.2685 bits, mode_consistency=1.00. REDUNDANT: max_cross_r=0.9670 with 'REC_mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9670)

### PREC_Qwen/Qwen3-Embedding-0.6B -- STRONG -- KEEP
- Pearson r: +0.2537  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.2529
- Cohen's d: 0.5244
- Mutual Info: 0.2665 bits
- Max cross-r: 0.9634 with 'Qwen/Qwen3-Embedding-0.6B' (redundant: yes)
- Mode consistency: 1.00
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.2537 (p=5.63e-93, Bonferroni:OK), Spearman r=+0.2529, Cohen's d=0.5244, MI=0.2665 bits, mode_consistency=1.00. REDUNDANT: max_cross_r=0.9634 with 'Qwen/Qwen3-Embedding-0.6B' (cross_r=+0.9634)

### REC_Qwen/Qwen3-Embedding-0.6B -- MODERATE -- REVIEW
- Pearson r: +0.2440  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.2407
- Cohen's d: 0.5032
- Mutual Info: 0.2513 bits
- Max cross-r: 0.9664 with 'Qwen/Qwen3-Embedding-0.6B' (redundant: yes)
- Mode consistency: 1.00
- **Reason:** Moderate signal but highly correlated with 'Qwen/Qwen3-Embedding-0.6B'. Ablation test recommended before dropping. Pearson r=+0.2440 (p=5.56e-86, Bonferroni:OK), Spearman r=+0.2407, Cohen's d=0.5032, MI=0.2513 bits, mode_consistency=1.00. REDUNDANT: max_cross_r=0.9664 with 'Qwen/Qwen3-Embedding-0.6B' (cross_r=+0.9664)

### Qwen/Qwen3-Embedding-0.6B -- MODERATE -- REVIEW
- Pearson r: +0.2356  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.2340
- Cohen's d: 0.4847
- Mutual Info: 0.2592 bits
- Max cross-r: 0.9664 with 'REC_Qwen/Qwen3-Embedding-0.6B' (redundant: yes)
- Mode consistency: 1.00
- **Reason:** Moderate signal but highly correlated with 'REC_Qwen/Qwen3-Embedding-0.6B'. Ablation test recommended before dropping. Pearson r=+0.2356 (p=4.42e-80, Bonferroni:OK), Spearman r=+0.2340, Cohen's d=0.4847, MI=0.2592 bits, mode_consistency=1.00. REDUNDANT: max_cross_r=0.9664 with 'REC_Qwen/Qwen3-Embedding-0.6B' (cross_r=+0.9664)

### dice -- MODERATE -- REVIEW
- Pearson r: +0.1460  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1529
- Cohen's d: 0.2951
- Mutual Info: 0.1242 bits
- Max cross-r: 0.9828 with 'jaccard' (redundant: yes)
- Mode consistency: 1.00
- **Reason:** Moderate signal but highly correlated with 'jaccard'. Ablation test recommended before dropping. Pearson r=+0.1460 (p=2.50e-31, Bonferroni:OK), Spearman r=+0.1529, Cohen's d=0.2951, MI=0.1242 bits, mode_consistency=1.00. REDUNDANT: max_cross_r=0.9828 with 'jaccard' (cross_r=+0.9828)

### cosine -- MODERATE -- REVIEW
- Pearson r: +0.1388  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1409
- Cohen's d: 0.2802
- Mutual Info: 0.2293 bits
- Max cross-r: 0.9380 with 'dice' (redundant: yes)
- Mode consistency: 1.00
- **Reason:** Moderate signal but highly correlated with 'dice'. Ablation test recommended before dropping. Pearson r=+0.1388 (p=1.96e-28, Bonferroni:OK), Spearman r=+0.1409, Cohen's d=0.2802, MI=0.2293 bits, mode_consistency=1.00. REDUNDANT: max_cross_r=0.9380 with 'dice' (cross_r=+0.9380)

### jaccard -- MODERATE -- REVIEW
- Pearson r: +0.1309  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1557
- Cohen's d: 0.2641
- Mutual Info: 0.1195 bits
- Max cross-r: 0.9828 with 'dice' (redundant: yes)
- Mode consistency: 1.00
- **Reason:** Moderate signal but highly correlated with 'dice'. Ablation test recommended before dropping. Pearson r=+0.1309 (p=1.83e-25, Bonferroni:OK), Spearman r=+0.1557, Cohen's d=0.2641, MI=0.1195 bits, mode_consistency=1.00. REDUNDANT: max_cross_r=0.9828 with 'dice' (cross_r=+0.9828)

### rouge -- MODERATE -- REVIEW
- Pearson r: +0.1304  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1320
- Cohen's d: 0.2630
- Mutual Info: 0.1225 bits
- Max cross-r: 0.9342 with 'dice' (redundant: yes)
- Mode consistency: 1.00
- **Reason:** Moderate signal but highly correlated with 'dice'. Ablation test recommended before dropping. Pearson r=+0.1304 (p=2.88e-25, Bonferroni:OK), Spearman r=+0.1320, Cohen's d=0.2630, MI=0.1225 bits, mode_consistency=1.00. REDUNDANT: max_cross_r=0.9342 with 'dice' (cross_r=+0.9342)

### lcs_token -- MODERATE -- REVIEW
- Pearson r: +0.1281  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1244
- Cohen's d: 0.2584
- Mutual Info: 0.0905 bits
- Max cross-r: 0.9263 with 'dice' (redundant: yes)
- Mode consistency: 1.00
- **Reason:** Moderate signal but highly correlated with 'dice'. Ablation test recommended before dropping. Pearson r=+0.1281 (p=1.87e-24, Bonferroni:OK), Spearman r=+0.1244, Cohen's d=0.2584, MI=0.0905 bits, mode_consistency=1.00. REDUNDANT: max_cross_r=0.9263 with 'dice' (cross_r=+0.9263)

### rouge3 -- WEAK -- KEEP
- Pearson r: +0.1175  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1477
- Cohen's d: 0.2366
- Mutual Info: 0.0849 bits
- Max cross-r: 0.8361 with 'jaccard' (redundant: no)
- Mode consistency: 0.67
- **Reason:** Small but validated independent signal. Pearson r=+0.1175 (p=8.58e-21, Bonferroni:OK), Spearman r=+0.1477, Cohen's d=0.2366, MI=0.0849 bits, mode_consistency=0.67

### neutral -- WEAK -- KEEP
- Pearson r: -0.0959  (p=< 1e-10, Bonferroni: OK)
- Spearman r: -0.0814
- Cohen's d: -0.1926
- Mutual Info: 0.2907 bits
- Max cross-r: 0.6380 with 'entailment' (redundant: no)
- Mode consistency: 0.33
- **Reason:** Small but validated independent signal. Pearson r=-0.0959 (p=2.52e-14, Bonferroni:OK), Spearman r=-0.0814, Cohen's d=-0.1926, MI=0.2907 bits, mode_consistency=0.33

### lcs_char -- WEAK -- REVIEW
- Pearson r: +0.0918  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.0905
- Cohen's d: 0.1844
- Mutual Info: 0.2465 bits
- Max cross-r: 0.9213 with 'lcs_token' (redundant: yes)
- Mode consistency: 0.67
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'lcs_token'. Ablation test before dropping. Pearson r=+0.0918 (p=2.90e-13, Bonferroni:OK), Spearman r=+0.0905, Cohen's d=0.1844, MI=0.2465 bits, mode_consistency=0.67. REDUNDANT: max_cross_r=0.9213 with 'lcs_token' (cross_r=+0.9213)

### entity_percentage -- WEAK -- KEEP
- Pearson r: +0.0658  (p=1.73e-07, Bonferroni: OK)
- Spearman r: +0.0793
- Cohen's d: 0.1319
- Mutual Info: 0.0047 bits
- Max cross-r: 0.0957 with 'entity_quantity' (redundant: no)
- Mode consistency: 0.33
- **Reason:** Small but validated independent signal. Pearson r=+0.0658 (p=1.73e-07, Bonferroni:OK), Spearman r=+0.0793, Cohen's d=0.1319, MI=0.0047 bits, mode_consistency=0.33

### entity_product -- MARGINAL -- REVIEW
- Pearson r: +0.0419  (p=8.86e-04, Bonferroni: OK)
- Spearman r: +0.0398
- Cohen's d: 0.0839
- Mutual Info: 0.0014 bits
- Max cross-r: 0.1444 with 'PREC_Qwen/Qwen3-Embedding-0.6B' (redundant: no)
- Mode consistency: 0.00
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0419 (p=8.86e-04, Bonferroni:OK), Spearman r=+0.0398, Cohen's d=0.0839, MI=0.0014 bits, mode_consistency=0.00

### entity_law -- MARGINAL -- REVIEW
- Pearson r: +0.0340  (p=6.97e-03, Bonferroni: FAIL)
- Spearman r: +0.0262
- Cohen's d: 0.0681
- Mutual Info: 0.0026 bits
- Max cross-r: 0.0665 with 'lcs_token' (redundant: no)
- Mode consistency: 0.00
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0340 (p=6.97e-03, Bonferroni:FAIL), Spearman r=+0.0262, Cohen's d=0.0681, MI=0.0026 bits, mode_consistency=0.00

### entity_time -- MARGINAL -- REVIEW
- Pearson r: +0.0296  (p=1.89e-02, Bonferroni: FAIL)
- Spearman r: +0.0239
- Cohen's d: 0.0592
- Mutual Info: 0.0000 bits
- Max cross-r: 0.2905 with 'lcs_char' (redundant: no)
- Mode consistency: 0.00
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0296 (p=1.89e-02, Bonferroni:FAIL), Spearman r=+0.0239, Cohen's d=0.0592, MI=0.0000 bits, mode_consistency=0.00

### entity_location -- MARGINAL -- REVIEW
- Pearson r: +0.0263  (p=3.71e-02, Bonferroni: FAIL)
- Spearman r: +0.0292
- Cohen's d: 0.0526
- Mutual Info: 0.0000 bits
- Max cross-r: 0.2940 with 'lcs_char' (redundant: no)
- Mode consistency: 0.00
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0263 (p=3.71e-02, Bonferroni:FAIL), Spearman r=+0.0292, Cohen's d=0.0526, MI=0.0000 bits, mode_consistency=0.00

### entity_duration -- MARGINAL -- REVIEW
- Pearson r: +0.0250  (p=4.75e-02, Bonferroni: FAIL)
- Spearman r: +0.0227
- Cohen's d: 0.0500
- Mutual Info: 0.0013 bits
- Max cross-r: 0.1115 with 'lcs_char' (redundant: no)
- Mode consistency: 0.00
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0250 (p=4.75e-02, Bonferroni:FAIL), Spearman r=+0.0227, Cohen's d=0.0500, MI=0.0013 bits, mode_consistency=0.00

### entity_language -- MARGINAL -- REVIEW
- Pearson r: +0.0212  (p=9.31e-02, Bonferroni: FAIL)
- Spearman r: +0.0177
- Cohen's d: 0.0424
- Mutual Info: 0.0047 bits
- Max cross-r: 0.1928 with 'entity_date' (redundant: no)
- Mode consistency: 0.00
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0212 (p=9.31e-02, Bonferroni:FAIL), Spearman r=+0.0177, Cohen's d=0.0424, MI=0.0047 bits, mode_consistency=0.00

### entity_quantity -- MARGINAL -- REVIEW
- Pearson r: +0.0181  (p=1.51e-01, Bonferroni: FAIL)
- Spearman r: +0.0271
- Cohen's d: 0.0362
- Mutual Info: 0.0120 bits
- Max cross-r: 0.1342 with 'lcs_char' (redundant: no)
- Mode consistency: 0.00
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0181 (p=1.51e-01, Bonferroni:FAIL), Spearman r=+0.0271, Cohen's d=0.0362, MI=0.0120 bits, mode_consistency=0.00

### entity_number -- MARGINAL -- REVIEW
- Pearson r: +0.0173  (p=1.70e-01, Bonferroni: FAIL)
- Spearman r: +0.0124
- Cohen's d: 0.0346
- Mutual Info: 0.0119 bits
- Max cross-r: 0.1316 with 'lcs_char' (redundant: no)
- Mode consistency: 0.00
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0173 (p=1.70e-01, Bonferroni:FAIL), Spearman r=+0.0124, Cohen's d=0.0346, MI=0.0119 bits, mode_consistency=0.00

### entity_date -- MARGINAL -- REVIEW
- Pearson r: -0.0137  (p=2.78e-01, Bonferroni: FAIL)
- Spearman r: -0.0179
- Cohen's d: -0.0274
- Mutual Info: 0.0038 bits
- Max cross-r: 0.2623 with 'lcs_char' (redundant: no)
- Mode consistency: 0.00
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=-0.0137 (p=2.78e-01, Bonferroni:FAIL), Spearman r=-0.0179, Cohen's d=-0.0274, MI=0.0038 bits, mode_consistency=0.00

### entity_money -- NOISE -- DROP
- Pearson r: +0.0115  (p=3.60e-01, Bonferroni: FAIL)
- Spearman r: +0.0152
- Cohen's d: 0.0231
- Mutual Info: 0.0000 bits
- Max cross-r: 0.0947 with 'lcs_char' (redundant: no)
- Mode consistency: 0.00
- **Reason:** No statistically significant or practically meaningful signal on any measure. Pearson r=+0.0115 (p=3.60e-01, Bonferroni:FAIL), Spearman r=+0.0152, Cohen's d=0.0231, MI=0.0000 bits, mode_consistency=0.00

### entity_person -- MARGINAL -- REVIEW
- Pearson r: +0.0086  (p=4.94e-01, Bonferroni: FAIL)
- Spearman r: +0.0118
- Cohen's d: 0.0172
- Mutual Info: 0.0038 bits
- Max cross-r: 0.3961 with 'Qwen/Qwen3-Embedding-0.6B' (redundant: no)
- Mode consistency: 0.00
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0086 (p=4.94e-01, Bonferroni:FAIL), Spearman r=+0.0118, Cohen's d=0.0172, MI=0.0038 bits, mode_consistency=0.00

### entity_event -- NOISE -- DROP
- Pearson r: +0.0081  (p=5.21e-01, Bonferroni: FAIL)
- Spearman r: +0.0029
- Cohen's d: 0.0162
- Mutual Info: 0.0000 bits
- Max cross-r: 0.1782 with 'lcs_char' (redundant: no)
- Mode consistency: 0.00
- **Reason:** No statistically significant or practically meaningful signal on any measure. Pearson r=+0.0081 (p=5.21e-01, Bonferroni:FAIL), Spearman r=+0.0029, Cohen's d=0.0162, MI=0.0000 bits, mode_consistency=0.00

### entity_organization -- NOISE -- DROP
- Pearson r: -0.0038  (p=7.63e-01, Bonferroni: FAIL)
- Spearman r: -0.0020
- Cohen's d: -0.0076
- Mutual Info: 0.0000 bits
- Max cross-r: 0.2737 with 'lcs_char' (redundant: no)
- Mode consistency: 0.00
- **Reason:** No statistically significant or practically meaningful signal on any measure. Pearson r=-0.0038 (p=7.63e-01, Bonferroni:FAIL), Spearman r=-0.0020, Cohen's d=-0.0076, MI=0.0000 bits, mode_consistency=0.00

## PCA Top Loadings (PC01-PC05)
| PC | Feature 1 | Loading | Feature 2 | Loading | Feature 3 | Loading |
|----|-----------|---------|-----------|---------|-----------|---------|
| PC01 | dice | +0.2833 | jaccard | +0.2772 | Qwen/Qwen3-Embedding-0.6B | +0.2725 |
| PC02 | contradiction | +0.4178 | entity_date | +0.2764 | lcs_char | +0.2458 |
| PC03 | entity_date | +0.4047 | entity_language | +0.3099 | entity_location | +0.3032 |
| PC04 | neutral | +0.5983 | entailment | -0.5561 | entity_percentage | -0.2796 |
| PC05 | contradiction | +0.5072 | entity_quantity | -0.3830 | neutral | -0.3757 |

## Top-10 K-means Discriminative Features
| Feature | Centroid Distance |
|---------|-----------------|
| dice | 1.6266 |
| rouge | 1.5587 |
| jaccard | 1.5440 |
| lcs_token | 1.5347 |
| lcs_char | 1.5186 |
| Qwen/Qwen3-Embedding-0.6B | 1.5179 |
| cosine | 1.4852 |
| mixedbread-ai/mxbai-embed-large-v1 | 1.4776 |
| REC_Qwen/Qwen3-Embedding-0.6B | 1.4662 |
| PREC_Qwen/Qwen3-Embedding-0.6B | 1.4616 |

## Per-Mode Pearson r
| Feature | context-vs-generated | model-vs-model | reference-vs-generated |
|---------|---------|---------|---------|
| entailment | +0.5183 | +0.5182 | +0.4415 |
| contradiction | -0.4424 | -0.5108 | -0.4961 |
| PREC_mixedbread-ai/mxbai-embed-large-v1 | +0.1541 | +0.4204 | +0.3478 |
| REC_mixedbread-ai/mxbai-embed-large-v1 | +0.1427 | +0.4040 | +0.3435 |
| mixedbread-ai/mxbai-embed-large-v1 | +0.1421 | +0.3937 | +0.3337 |
| PREC_Qwen/Qwen3-Embedding-0.6B | +0.1309 | +0.3815 | +0.2936 |
| REC_Qwen/Qwen3-Embedding-0.6B | +0.1204 | +0.3685 | +0.2905 |
| Qwen/Qwen3-Embedding-0.6B | +0.1205 | +0.3569 | +0.2812 |
| dice | +0.0915 | +0.2171 | +0.1620 |
| cosine | +0.0714 | +0.2047 | +0.1587 |
| jaccard | +0.1076 | +0.1865 | +0.1265 |
| rouge | +0.0546 | +0.1891 | +0.1895 |
| lcs_token | +0.0679 | +0.2092 | +0.1296 |
| rouge3 | +0.0531 | +0.1536 | +0.1485 |
| neutral | -0.1344 | -0.1071 | -0.0442 |
| lcs_char | +0.0139 | +0.1955 | +0.1012 |
| entity_percentage | +0.0540 | +0.0915 | +0.0483 |
| entity_product | +0.0340 | +0.0716 | +0.0176 |
| entity_law | +0.0366 | +0.0324 | +0.0333 |
| entity_time | +0.0462 | +0.0207 | +0.0168 |
| entity_location | +0.0114 | +0.0642 | +0.0083 |
| entity_duration | +0.0115 | +0.0383 | +0.0308 |
| entity_language | +0.0032 | +0.0434 | +0.0329 |
| entity_quantity | +0.0082 | +0.0525 | -0.0059 |
| entity_number | +0.0005 | +0.0379 | +0.0192 |
| entity_date | -0.0214 | +0.0100 | -0.0315 |
| entity_money | +0.0112 | +0.0126 | +0.0112 |
| entity_person | -0.0350 | +0.0617 | +0.0186 |
| entity_event | -0.0090 | +0.0131 | +0.0267 |
| entity_organization | -0.0351 | +0.0122 | +0.0203 |