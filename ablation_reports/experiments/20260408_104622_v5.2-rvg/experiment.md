# Ablation Experiment Report
Generated: 2026-04-08T10:46:22.221455  |  Tag: v5.2-rvg  |  Pairs: 1831  |  Missing: 0

## Run Parameters
- Mode filter: reference-vs-generated
- Split filter: all
- Framework version: 1.0
- Bonferroni threshold: p < 1.6667e-03
- Redundancy threshold: max_cross_r >= 0.85

## Global Clustering Metrics
- K-means ARI (k=2): 0.0479
- PCA PC01 explained variance: 0.329  (cumulative PC05: 0.646)

## Verdict Summary
| Verdict | Count | Features |
|---------|-------|---------|
| KEEP | 10 | mixedbread-ai/mxbai-embed-large-v1, PREC_mixedbread-ai/mxbai-embed-large-v1, REC_mixedbread-ai/mxbai-embed-large-v1, PREC_Qwen/Qwen3-Embedding-0.6B, entailment, neutral, contradiction, entity_value_prec, entity_value_rec, numeric_jaccard |
| REVIEW | 20 | dice, rouge3, rouge, jaccard, entity_product, entity_percentage, lcs_token, lcs_char, entity_location_value_prec, entity_location_value_rec, entity_product_value_prec, entity_product_value_rec, entity_date_value_prec, entity_date_value_rec, entity_time_value_prec, entity_time_value_rec, entity_duration_value_prec, entity_duration_value_rec, entity_percentage_value_prec, entity_percentage_value_rec |
| DROP | 0 | — |

## Tier Summary
| Tier | Count | Features |
|------|-------|---------|
| STRONG | 6 | mixedbread-ai/mxbai-embed-large-v1, PREC_mixedbread-ai/mxbai-embed-large-v1, REC_mixedbread-ai/mxbai-embed-large-v1, PREC_Qwen/Qwen3-Embedding-0.6B, entailment, contradiction |
| MODERATE | 7 | dice, rouge3, rouge, jaccard, entity_value_prec, entity_value_rec, lcs_token |
| WEAK | 5 | neutral, entity_product, lcs_char, numeric_jaccard, entity_product_value_rec |
| MARGINAL | 12 | entity_percentage, entity_location_value_prec, entity_location_value_rec, entity_product_value_prec, entity_date_value_prec, entity_date_value_rec, entity_time_value_prec, entity_time_value_rec, entity_duration_value_prec, entity_duration_value_rec, entity_percentage_value_prec, entity_percentage_value_rec |
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
- Mutual Info: 0.0988 bits
- Max cross-r: 0.9761 with 'mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.3335 (p=8.28e-49, Bonferroni:OK), Spearman r=+0.3002, Cohen's d=0.7072, MI=0.0988 bits. REDUNDANT: max_cross_r=0.9761 with 'mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9761)

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
- Mutual Info: 0.0935 bits
- Max cross-r: 0.9767 with 'PREC_mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.3265 (p=9.88e-47, Bonferroni:OK), Spearman r=+0.2958, Cohen's d=0.6905, MI=0.0935 bits. REDUNDANT: max_cross_r=0.9767 with 'PREC_mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9767)

### PREC_Qwen/Qwen3-Embedding-0.6B -- STRONG -- KEEP
- Pearson r: +0.2946  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.2640
- Cohen's d: 0.6162
- Mutual Info: 0.0648 bits
- Max cross-r: 0.9036 with 'PREC_mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.2946 (p=5.68e-38, Bonferroni:OK), Spearman r=+0.2640, Cohen's d=0.6162, MI=0.0648 bits. REDUNDANT: max_cross_r=0.9036 with 'PREC_mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9036)

### rouge -- MODERATE -- REVIEW
- Pearson r: +0.1768  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1865
- Cohen's d: 0.3592
- Mutual Info: 0.0481 bits
- Max cross-r: 0.9214 with 'dice' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'dice'. Ablation test recommended before dropping. Pearson r=+0.1768 (p=2.50e-14, Bonferroni:OK), Spearman r=+0.1865, Cohen's d=0.3592, MI=0.0481 bits. REDUNDANT: max_cross_r=0.9214 with 'dice' (cross_r=+0.9214)

### entity_value_rec -- MODERATE -- KEEP
- Pearson r: +0.1713  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1679
- Cohen's d: 0.3476
- Mutual Info: 0.0381 bits
- Max cross-r: 0.7092 with 'entity_value_prec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Moderate validated signal with independent information. Pearson r=+0.1713 (p=1.56e-13, Bonferroni:OK), Spearman r=+0.1679, Cohen's d=0.3476, MI=0.0381 bits

### dice -- MODERATE -- REVIEW
- Pearson r: +0.1641  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1754
- Cohen's d: 0.3324
- Mutual Info: 0.0597 bits
- Max cross-r: 0.9846 with 'jaccard' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'jaccard'. Ablation test recommended before dropping. Pearson r=+0.1641 (p=1.63e-12, Bonferroni:OK), Spearman r=+0.1754, Cohen's d=0.3324, MI=0.0597 bits. REDUNDANT: max_cross_r=0.9846 with 'jaccard' (cross_r=+0.9846)

### rouge3 -- MODERATE -- REVIEW
- Pearson r: +0.1404  (p=1.61e-09, Bonferroni: OK)
- Spearman r: +0.1645
- Cohen's d: 0.2834
- Mutual Info: 0.0372 bits
- Max cross-r: 0.8532 with 'jaccard' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'jaccard'. Ablation test recommended before dropping. Pearson r=+0.1404 (p=1.61e-09, Bonferroni:OK), Spearman r=+0.1645, Cohen's d=0.2834, MI=0.0372 bits. REDUNDANT: max_cross_r=0.8532 with 'jaccard' (cross_r=+0.8532)

### lcs_token -- MODERATE -- REVIEW
- Pearson r: +0.1380  (p=3.02e-09, Bonferroni: OK)
- Spearman r: +0.1507
- Cohen's d: 0.2786
- Mutual Info: 0.0372 bits
- Max cross-r: 0.9364 with 'lcs_char' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'lcs_char'. Ablation test recommended before dropping. Pearson r=+0.1380 (p=3.02e-09, Bonferroni:OK), Spearman r=+0.1507, Cohen's d=0.2786, MI=0.0372 bits. REDUNDANT: max_cross_r=0.9364 with 'lcs_char' (cross_r=+0.9364)

### entity_value_prec -- MODERATE -- KEEP
- Pearson r: +0.1342  (p=8.22e-09, Bonferroni: OK)
- Spearman r: +0.1306
- Cohen's d: 0.2707
- Mutual Info: 0.0167 bits
- Max cross-r: 0.7092 with 'entity_value_rec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Moderate validated signal with independent information. Pearson r=+0.1342 (p=8.22e-09, Bonferroni:OK), Spearman r=+0.1306, Cohen's d=0.2707, MI=0.0167 bits

### jaccard -- MODERATE -- REVIEW
- Pearson r: +0.1285  (p=3.44e-08, Bonferroni: OK)
- Spearman r: +0.1756
- Cohen's d: 0.2590
- Mutual Info: 0.0737 bits
- Max cross-r: 0.9846 with 'dice' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Moderate signal but highly correlated with 'dice'. Ablation test recommended before dropping. Pearson r=+0.1285 (p=3.44e-08, Bonferroni:OK), Spearman r=+0.1756, Cohen's d=0.2590, MI=0.0737 bits. REDUNDANT: max_cross_r=0.9846 with 'dice' (cross_r=+0.9846)

### lcs_char -- WEAK -- REVIEW
- Pearson r: +0.1081  (p=3.56e-06, Bonferroni: OK)
- Spearman r: +0.1041
- Cohen's d: 0.2173
- Mutual Info: 0.0120 bits
- Max cross-r: 0.9364 with 'lcs_token' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'lcs_token'. Ablation test before dropping. Pearson r=+0.1081 (p=3.56e-06, Bonferroni:OK), Spearman r=+0.1041, Cohen's d=0.2173, MI=0.0120 bits. REDUNDANT: max_cross_r=0.9364 with 'lcs_token' (cross_r=+0.9364)

### neutral -- WEAK -- KEEP
- Pearson r: -0.0670  (p=4.10e-03, Bonferroni: FAIL)
- Spearman r: -0.0349
- Cohen's d: -0.1343
- Mutual Info: 0.0977 bits
- Max cross-r: 0.6440 with 'entailment' (redundant: no)
- Mode consistency: n/a
- **Reason:** Small but validated independent signal. Pearson r=-0.0670 (p=4.10e-03, Bonferroni:FAIL), Spearman r=-0.0349, Cohen's d=-0.1343, MI=0.0977 bits

### entity_product_value_rec -- WEAK -- REVIEW
- Pearson r: +0.0622  (p=7.73e-03, Bonferroni: FAIL)
- Spearman r: +0.0638
- Cohen's d: 0.1246
- Mutual Info: 0.0137 bits
- Max cross-r: 0.9349 with 'entity_product' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'entity_product'. Ablation test before dropping. Pearson r=+0.0622 (p=7.73e-03, Bonferroni:FAIL), Spearman r=+0.0638, Cohen's d=0.1246, MI=0.0137 bits. REDUNDANT: max_cross_r=0.9349 with 'entity_product' (cross_r=+0.9349)

### entity_percentage_value_rec -- MARGINAL -- REVIEW
- Pearson r: +0.0583  (p=1.26e-02, Bonferroni: FAIL)
- Spearman r: +0.0627
- Cohen's d: 0.1168
- Mutual Info: 0.0000 bits
- Max cross-r: 0.9273 with 'entity_percentage' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0583 (p=1.26e-02, Bonferroni:FAIL), Spearman r=+0.0627, Cohen's d=0.1168, MI=0.0000 bits. REDUNDANT: max_cross_r=0.9273 with 'entity_percentage' (cross_r=+0.9273)

### entity_product -- WEAK -- REVIEW
- Pearson r: +0.0574  (p=1.40e-02, Bonferroni: FAIL)
- Spearman r: +0.0604
- Cohen's d: 0.1149
- Mutual Info: 0.0134 bits
- Max cross-r: 0.9349 with 'entity_product_value_rec' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'entity_product_value_rec'. Ablation test before dropping. Pearson r=+0.0574 (p=1.40e-02, Bonferroni:FAIL), Spearman r=+0.0604, Cohen's d=0.1149, MI=0.0134 bits. REDUNDANT: max_cross_r=0.9349 with 'entity_product_value_rec' (cross_r=+0.9349)

### numeric_jaccard -- WEAK -- KEEP
- Pearson r: +0.0562  (p=1.61e-02, Bonferroni: FAIL)
- Spearman r: +0.0568
- Cohen's d: 0.1126
- Mutual Info: 0.0045 bits
- Max cross-r: 0.4056 with 'entity_time_value_rec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Small but validated independent signal. Pearson r=+0.0562 (p=1.61e-02, Bonferroni:FAIL), Spearman r=+0.0568, Cohen's d=0.1126, MI=0.0045 bits

### entity_percentage -- MARGINAL -- REVIEW
- Pearson r: +0.0467  (p=4.56e-02, Bonferroni: FAIL)
- Spearman r: +0.0580
- Cohen's d: 0.0935
- Mutual Info: 0.0000 bits
- Max cross-r: 0.9273 with 'entity_percentage_value_rec' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0467 (p=4.56e-02, Bonferroni:FAIL), Spearman r=+0.0580, Cohen's d=0.0935, MI=0.0000 bits. REDUNDANT: max_cross_r=0.9273 with 'entity_percentage_value_rec' (cross_r=+0.9273)

### entity_percentage_value_prec -- MARGINAL -- REVIEW
- Pearson r: +0.0355  (p=1.29e-01, Bonferroni: FAIL)
- Spearman r: +0.0459
- Cohen's d: 0.0709
- Mutual Info: 0.0051 bits
- Max cross-r: 0.8467 with 'entity_percentage_value_rec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0355 (p=1.29e-01, Bonferroni:FAIL), Spearman r=+0.0459, Cohen's d=0.0709, MI=0.0051 bits

### entity_location_value_rec -- MARGINAL -- REVIEW
- Pearson r: +0.0349  (p=1.36e-01, Bonferroni: FAIL)
- Spearman r: +0.0321
- Cohen's d: 0.0697
- Mutual Info: 0.0000 bits
- Max cross-r: 0.7840 with 'entity_location_value_prec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0349 (p=1.36e-01, Bonferroni:FAIL), Spearman r=+0.0321, Cohen's d=0.0697, MI=0.0000 bits

### entity_duration_value_prec -- MARGINAL -- REVIEW
- Pearson r: +0.0346  (p=1.39e-01, Bonferroni: FAIL)
- Spearman r: +0.0381
- Cohen's d: 0.0692
- Mutual Info: 0.0000 bits
- Max cross-r: 0.7788 with 'entity_duration_value_rec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0346 (p=1.39e-01, Bonferroni:FAIL), Spearman r=+0.0381, Cohen's d=0.0692, MI=0.0000 bits

### entity_duration_value_rec -- MARGINAL -- REVIEW
- Pearson r: +0.0309  (p=1.87e-01, Bonferroni: FAIL)
- Spearman r: +0.0357
- Cohen's d: 0.0617
- Mutual Info: 0.0198 bits
- Max cross-r: 0.7788 with 'entity_duration_value_prec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0309 (p=1.87e-01, Bonferroni:FAIL), Spearman r=+0.0357, Cohen's d=0.0617, MI=0.0198 bits

### entity_date_value_rec -- MARGINAL -- REVIEW
- Pearson r: +0.0283  (p=2.26e-01, Bonferroni: FAIL)
- Spearman r: +0.0298
- Cohen's d: 0.0566
- Mutual Info: 0.0000 bits
- Max cross-r: 0.8679 with 'entity_date_value_prec' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0283 (p=2.26e-01, Bonferroni:FAIL), Spearman r=+0.0298, Cohen's d=0.0566, MI=0.0000 bits. REDUNDANT: max_cross_r=0.8679 with 'entity_date_value_prec' (cross_r=+0.8679)

### entity_date_value_prec -- MARGINAL -- REVIEW
- Pearson r: +0.0283  (p=2.27e-01, Bonferroni: FAIL)
- Spearman r: +0.0280
- Cohen's d: 0.0565
- Mutual Info: 0.0093 bits
- Max cross-r: 0.8679 with 'entity_date_value_rec' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0283 (p=2.27e-01, Bonferroni:FAIL), Spearman r=+0.0280, Cohen's d=0.0565, MI=0.0093 bits. REDUNDANT: max_cross_r=0.8679 with 'entity_date_value_rec' (cross_r=+0.8679)

### entity_location_value_prec -- MARGINAL -- REVIEW
- Pearson r: +0.0251  (p=2.83e-01, Bonferroni: FAIL)
- Spearman r: +0.0139
- Cohen's d: 0.0502
- Mutual Info: 0.0000 bits
- Max cross-r: 0.7840 with 'entity_location_value_rec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0251 (p=2.83e-01, Bonferroni:FAIL), Spearman r=+0.0139, Cohen's d=0.0502, MI=0.0000 bits

### entity_product_value_prec -- MARGINAL -- REVIEW
- Pearson r: +0.0203  (p=3.84e-01, Bonferroni: FAIL)
- Spearman r: +0.0194
- Cohen's d: 0.0407
- Mutual Info: 0.0041 bits
- Max cross-r: 0.7345 with 'entity_product_value_rec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0203 (p=3.84e-01, Bonferroni:FAIL), Spearman r=+0.0194, Cohen's d=0.0407, MI=0.0041 bits

### entity_time_value_prec -- MARGINAL -- REVIEW
- Pearson r: +0.0075  (p=7.50e-01, Bonferroni: FAIL)
- Spearman r: +0.0091
- Cohen's d: 0.0149
- Mutual Info: 0.0071 bits
- Max cross-r: 0.7904 with 'entity_time_value_rec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0075 (p=7.50e-01, Bonferroni:FAIL), Spearman r=+0.0091, Cohen's d=0.0149, MI=0.0071 bits

### entity_time_value_rec -- MARGINAL -- REVIEW
- Pearson r: -0.0008  (p=9.73e-01, Bonferroni: FAIL)
- Spearman r: +0.0047
- Cohen's d: -0.0016
- Mutual Info: 0.0102 bits
- Max cross-r: 0.7904 with 'entity_time_value_prec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=-0.0008 (p=9.73e-01, Bonferroni:FAIL), Spearman r=+0.0047, Cohen's d=-0.0016, MI=0.0102 bits

## PCA Top Loadings (PC01-PC05)
| PC | Feature 1 | Loading | Feature 2 | Loading | Feature 3 | Loading |
|----|-----------|---------|-----------|---------|-----------|---------|
| PC01 | dice | +0.2996 | jaccard | +0.2945 | lcs_token | +0.2878 |
| PC02 | entity_percentage_value_rec | +0.4939 | entity_percentage | +0.4715 | entity_percentage_value_prec | +0.4631 |
| PC03 | entity_product_value_rec | +0.5435 | entity_product | +0.5207 | entity_product_value_prec | +0.4740 |
| PC04 | entity_date_value_rec | +0.3091 | entity_date_value_prec | +0.2936 | entity_location_value_prec | +0.2729 |
| PC05 | entity_duration_value_rec | +0.5609 | entity_duration_value_prec | +0.5429 | entity_date_value_prec | -0.3100 |

## Top-10 K-means Discriminative Features
| Feature | Centroid Distance |
|---------|-----------------|
| dice | 1.6403 |
| jaccard | 1.5889 |
| lcs_token | 1.5653 |
| lcs_char | 1.5165 |
| mixedbread-ai/mxbai-embed-large-v1 | 1.4874 |
| rouge | 1.4803 |
| PREC_mixedbread-ai/mxbai-embed-large-v1 | 1.4589 |
| REC_mixedbread-ai/mxbai-embed-large-v1 | 1.4589 |
| PREC_Qwen/Qwen3-Embedding-0.6B | 1.4352 |
| rouge3 | 1.3426 |

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
| entity_value_rec | +0.1713 |
| dice | +0.1641 |
| rouge3 | +0.1404 |
| lcs_token | +0.1380 |
| entity_value_prec | +0.1342 |
| jaccard | +0.1285 |
| lcs_char | +0.1081 |
| neutral | -0.0670 |
| entity_product_value_rec | +0.0622 |
| entity_percentage_value_rec | +0.0583 |
| entity_product | +0.0574 |
| numeric_jaccard | +0.0562 |
| entity_percentage | +0.0467 |
| entity_percentage_value_prec | +0.0355 |
| entity_location_value_rec | +0.0349 |
| entity_duration_value_prec | +0.0346 |
| entity_duration_value_rec | +0.0309 |
| entity_date_value_rec | +0.0283 |
| entity_date_value_prec | +0.0283 |
| entity_location_value_prec | +0.0251 |
| entity_product_value_prec | +0.0203 |
| entity_time_value_prec | +0.0075 |
| entity_time_value_rec | -0.0008 |