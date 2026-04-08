# Ablation Experiment Report
Generated: 2026-04-08T10:46:10.151604  |  Tag: v5.2-cvg  |  Pairs: 2331  |  Missing: 0

## Run Parameters
- Mode filter: context-vs-generated
- Split filter: all
- Framework version: 1.0
- Bonferroni threshold: p < 2.0833e-03
- Redundancy threshold: max_cross_r >= 0.85

## Global Clustering Metrics
- K-means ARI (k=2): 0.0069
- PCA PC01 explained variance: 0.291  (cumulative PC05: 0.645)

## Verdict Summary
| Verdict | Count | Features |
|---------|-------|---------|
| KEEP | 7 | rouge3, entailment, neutral, contradiction, entity_value_prec, lcs_char, entity_product_value_prec |
| REVIEW | 16 | dice, rouge, jaccard, mixedbread-ai/mxbai-embed-large-v1, PREC_mixedbread-ai/mxbai-embed-large-v1, REC_mixedbread-ai/mxbai-embed-large-v1, PREC_Qwen/Qwen3-Embedding-0.6B, entity_value_rec, lcs_token, numeric_jaccard, entity_location_value_prec, entity_location_value_rec, entity_product_value_rec, entity_time_value_prec, entity_percentage_value_prec, entity_percentage_value_rec |
| DROP | 1 | entity_time_value_rec |

## Tier Summary
| Tier | Count | Features |
|------|-------|---------|
| STRONG | 2 | entailment, contradiction |
| MODERATE | 2 | entity_value_prec, lcs_char |
| WEAK | 11 | dice, rouge3, rouge, jaccard, mixedbread-ai/mxbai-embed-large-v1, PREC_mixedbread-ai/mxbai-embed-large-v1, REC_mixedbread-ai/mxbai-embed-large-v1, PREC_Qwen/Qwen3-Embedding-0.6B, neutral, lcs_token, entity_product_value_prec |
| MARGINAL | 8 | entity_value_rec, numeric_jaccard, entity_location_value_prec, entity_location_value_rec, entity_product_value_rec, entity_time_value_prec, entity_percentage_value_prec, entity_percentage_value_rec |
| NOISE | 1 | entity_time_value_rec |

## Per-Feature Detail
(sorted by |Pearson r| descending)

### entailment -- STRONG -- KEEP
- Pearson r: +0.4019  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.4069
- Cohen's d: 0.8773
- Mutual Info: 0.1383 bits
- Max cross-r: 0.6274 with 'neutral' (redundant: no)
- Mode consistency: n/a
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=+0.4019 (p=3.32e-91, Bonferroni:OK), Spearman r=+0.4069, Cohen's d=0.8773, MI=0.1383 bits

### contradiction -- STRONG -- KEEP
- Pearson r: -0.3500  (p=< 1e-10, Bonferroni: OK)
- Spearman r: -0.3103
- Cohen's d: -0.7471
- Mutual Info: 0.1025 bits
- Max cross-r: 0.4831 with 'neutral' (redundant: no)
- Mode consistency: n/a
- **Reason:** Strong multi-measure signal. Core feature. Pearson r=-0.3500 (p=3.78e-68, Bonferroni:OK), Spearman r=-0.3103, Cohen's d=-0.7471, MI=0.1025 bits

### entity_value_prec -- MODERATE -- KEEP
- Pearson r: +0.1353  (p=< 1e-10, Bonferroni: OK)
- Spearman r: +0.1320
- Cohen's d: 0.2730
- Mutual Info: 0.0335 bits
- Max cross-r: 0.4356 with 'entity_value_rec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Moderate validated signal with independent information. Pearson r=+0.1353 (p=5.42e-11, Bonferroni:OK), Spearman r=+0.1320, Cohen's d=0.2730, MI=0.0335 bits

### lcs_char -- MODERATE -- KEEP
- Pearson r: -0.1288  (p=4.39e-10, Bonferroni: OK)
- Spearman r: -0.1556
- Cohen's d: -0.2596
- Mutual Info: 0.0272 bits
- Max cross-r: 0.7795 with 'lcs_token' (redundant: no)
- Mode consistency: n/a
- **Reason:** Moderate validated signal with independent information. Pearson r=-0.1288 (p=4.39e-10, Bonferroni:OK), Spearman r=-0.1556, Cohen's d=-0.2596, MI=0.0272 bits

### rouge -- WEAK -- REVIEW
- Pearson r: -0.1171  (p=1.41e-08, Bonferroni: OK)
- Spearman r: -0.1685
- Cohen's d: -0.2358
- Mutual Info: 0.0668 bits
- Max cross-r: 0.8706 with 'lcs_token' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'lcs_token'. Ablation test before dropping. Pearson r=-0.1171 (p=1.41e-08, Bonferroni:OK), Spearman r=-0.1685, Cohen's d=-0.2358, MI=0.0668 bits. REDUNDANT: max_cross_r=0.8706 with 'lcs_token' (cross_r=+0.8706)

### REC_mixedbread-ai/mxbai-embed-large-v1 -- WEAK -- REVIEW
- Pearson r: -0.1036  (p=5.40e-07, Bonferroni: OK)
- Spearman r: -0.1066
- Cohen's d: -0.2082
- Mutual Info: 0.0364 bits
- Max cross-r: 0.9323 with 'mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'mixedbread-ai/mxbai-embed-large-v1'. Ablation test before dropping. Pearson r=-0.1036 (p=5.40e-07, Bonferroni:OK), Spearman r=-0.1066, Cohen's d=-0.2082, MI=0.0364 bits. REDUNDANT: max_cross_r=0.9323 with 'mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9323)

### dice -- WEAK -- REVIEW
- Pearson r: -0.1024  (p=7.22e-07, Bonferroni: OK)
- Spearman r: -0.1146
- Cohen's d: -0.2058
- Mutual Info: 0.0409 bits
- Max cross-r: 0.9815 with 'jaccard' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'jaccard'. Ablation test before dropping. Pearson r=-0.1024 (p=7.22e-07, Bonferroni:OK), Spearman r=-0.1146, Cohen's d=-0.2058, MI=0.0409 bits. REDUNDANT: max_cross_r=0.9815 with 'jaccard' (cross_r=+0.9815)

### lcs_token -- WEAK -- REVIEW
- Pearson r: -0.1008  (p=1.07e-06, Bonferroni: OK)
- Spearman r: -0.1185
- Cohen's d: -0.2026
- Mutual Info: 0.0343 bits
- Max cross-r: 0.8706 with 'rouge' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'rouge'. Ablation test before dropping. Pearson r=-0.1008 (p=1.07e-06, Bonferroni:OK), Spearman r=-0.1185, Cohen's d=-0.2026, MI=0.0343 bits. REDUNDANT: max_cross_r=0.8706 with 'rouge' (cross_r=+0.8706)

### jaccard -- WEAK -- REVIEW
- Pearson r: -0.0950  (p=4.30e-06, Bonferroni: OK)
- Spearman r: -0.1127
- Cohen's d: -0.1909
- Mutual Info: 0.0273 bits
- Max cross-r: 0.9815 with 'dice' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'dice'. Ablation test before dropping. Pearson r=-0.0950 (p=4.30e-06, Bonferroni:OK), Spearman r=-0.1127, Cohen's d=-0.1909, MI=0.0273 bits. REDUNDANT: max_cross_r=0.9815 with 'dice' (cross_r=+0.9815)

### neutral -- WEAK -- KEEP
- Pearson r: -0.0857  (p=3.41e-05, Bonferroni: OK)
- Spearman r: -0.0780
- Cohen's d: -0.1720
- Mutual Info: 0.0530 bits
- Max cross-r: 0.6274 with 'entailment' (redundant: no)
- Mode consistency: n/a
- **Reason:** Small but validated independent signal. Pearson r=-0.0857 (p=3.41e-05, Bonferroni:OK), Spearman r=-0.0780, Cohen's d=-0.1720, MI=0.0530 bits

### mixedbread-ai/mxbai-embed-large-v1 -- WEAK -- REVIEW
- Pearson r: -0.0829  (p=6.10e-05, Bonferroni: OK)
- Spearman r: -0.0932
- Cohen's d: -0.1664
- Mutual Info: 0.0263 bits
- Max cross-r: 0.9323 with 'REC_mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'REC_mixedbread-ai/mxbai-embed-large-v1'. Ablation test before dropping. Pearson r=-0.0829 (p=6.10e-05, Bonferroni:OK), Spearman r=-0.0932, Cohen's d=-0.1664, MI=0.0263 bits. REDUNDANT: max_cross_r=0.9323 with 'REC_mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9323)

### PREC_mixedbread-ai/mxbai-embed-large-v1 -- WEAK -- REVIEW
- Pearson r: -0.0819  (p=7.50e-05, Bonferroni: OK)
- Spearman r: -0.0891
- Cohen's d: -0.1643
- Mutual Info: 0.0474 bits
- Max cross-r: 0.9254 with 'mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'mixedbread-ai/mxbai-embed-large-v1'. Ablation test before dropping. Pearson r=-0.0819 (p=7.50e-05, Bonferroni:OK), Spearman r=-0.0891, Cohen's d=-0.1643, MI=0.0474 bits. REDUNDANT: max_cross_r=0.9254 with 'mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9254)

### entity_product_value_prec -- WEAK -- KEEP
- Pearson r: +0.0707  (p=6.37e-04, Bonferroni: OK)
- Spearman r: +0.0601
- Cohen's d: 0.1417
- Mutual Info: 0.0150 bits
- Max cross-r: 0.7136 with 'entity_product_value_rec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Small but validated independent signal. Pearson r=+0.0707 (p=6.37e-04, Bonferroni:OK), Spearman r=+0.0601, Cohen's d=0.1417, MI=0.0150 bits

### rouge3 -- WEAK -- KEEP
- Pearson r: -0.0699  (p=7.27e-04, Bonferroni: OK)
- Spearman r: +0.0211
- Cohen's d: -0.1402
- Mutual Info: 0.0255 bits
- Max cross-r: 0.8146 with 'rouge' (redundant: no)
- Mode consistency: n/a
- **Reason:** Small but validated independent signal. Pearson r=-0.0699 (p=7.27e-04, Bonferroni:OK), Spearman r=+0.0211, Cohen's d=-0.1402, MI=0.0255 bits

### PREC_Qwen/Qwen3-Embedding-0.6B -- WEAK -- REVIEW
- Pearson r: -0.0645  (p=1.84e-03, Bonferroni: OK)
- Spearman r: -0.0679
- Cohen's d: -0.1292
- Mutual Info: 0.0214 bits
- Max cross-r: 0.8762 with 'PREC_mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'PREC_mixedbread-ai/mxbai-embed-large-v1'. Ablation test before dropping. Pearson r=-0.0645 (p=1.84e-03, Bonferroni:OK), Spearman r=-0.0679, Cohen's d=-0.1292, MI=0.0214 bits. REDUNDANT: max_cross_r=0.8762 with 'PREC_mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.8762)

### entity_time_value_prec -- MARGINAL -- REVIEW
- Pearson r: +0.0512  (p=1.34e-02, Bonferroni: FAIL)
- Spearman r: +0.0328
- Cohen's d: 0.1025
- Mutual Info: 0.0000 bits
- Max cross-r: 0.5847 with 'entity_time_value_rec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0512 (p=1.34e-02, Bonferroni:FAIL), Spearman r=+0.0328, Cohen's d=0.1025, MI=0.0000 bits

### entity_location_value_prec -- MARGINAL -- REVIEW
- Pearson r: +0.0444  (p=3.19e-02, Bonferroni: FAIL)
- Spearman r: +0.0351
- Cohen's d: 0.0889
- Mutual Info: 0.0153 bits
- Max cross-r: 0.5646 with 'entity_location_value_rec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0444 (p=3.19e-02, Bonferroni:FAIL), Spearman r=+0.0351, Cohen's d=0.0889, MI=0.0153 bits

### entity_percentage_value_rec -- MARGINAL -- REVIEW
- Pearson r: +0.0376  (p=6.94e-02, Bonferroni: FAIL)
- Spearman r: +0.0257
- Cohen's d: 0.0753
- Mutual Info: 0.0000 bits
- Max cross-r: 0.7065 with 'entity_percentage_value_prec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0376 (p=6.94e-02, Bonferroni:FAIL), Spearman r=+0.0257, Cohen's d=0.0753, MI=0.0000 bits

### entity_product_value_rec -- MARGINAL -- REVIEW
- Pearson r: +0.0361  (p=8.17e-02, Bonferroni: FAIL)
- Spearman r: +0.0291
- Cohen's d: 0.0722
- Mutual Info: 0.0215 bits
- Max cross-r: 0.7136 with 'entity_product_value_prec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0361 (p=8.17e-02, Bonferroni:FAIL), Spearman r=+0.0291, Cohen's d=0.0722, MI=0.0215 bits

### numeric_jaccard -- MARGINAL -- REVIEW
- Pearson r: +0.0275  (p=1.84e-01, Bonferroni: FAIL)
- Spearman r: +0.0337
- Cohen's d: 0.0551
- Mutual Info: 0.0011 bits
- Max cross-r: 0.5171 with 'lcs_char' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0275 (p=1.84e-01, Bonferroni:FAIL), Spearman r=+0.0337, Cohen's d=0.0551, MI=0.0011 bits

### entity_value_rec -- MARGINAL -- REVIEW
- Pearson r: -0.0236  (p=2.55e-01, Bonferroni: FAIL)
- Spearman r: -0.0431
- Cohen's d: -0.0472
- Mutual Info: 0.0313 bits
- Max cross-r: 0.5286 with 'mixedbread-ai/mxbai-embed-large-v1' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=-0.0236 (p=2.55e-01, Bonferroni:FAIL), Spearman r=-0.0431, Cohen's d=-0.0472, MI=0.0313 bits

### entity_time_value_rec -- NOISE -- DROP
- Pearson r: +0.0197  (p=3.42e-01, Bonferroni: FAIL)
- Spearman r: +0.0082
- Cohen's d: 0.0393
- Mutual Info: 0.0000 bits
- Max cross-r: 0.5847 with 'entity_time_value_prec' (redundant: no)
- Mode consistency: n/a
- **Reason:** No statistically significant or practically meaningful signal on any measure. Pearson r=+0.0197 (p=3.42e-01, Bonferroni:FAIL), Spearman r=+0.0082, Cohen's d=0.0393, MI=0.0000 bits

### entity_percentage_value_prec -- MARGINAL -- REVIEW
- Pearson r: +0.0123  (p=5.54e-01, Bonferroni: FAIL)
- Spearman r: -0.0044
- Cohen's d: 0.0245
- Mutual Info: 0.0148 bits
- Max cross-r: 0.7065 with 'entity_percentage_value_rec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0123 (p=5.54e-01, Bonferroni:FAIL), Spearman r=-0.0044, Cohen's d=0.0245, MI=0.0148 bits

### entity_location_value_rec -- MARGINAL -- REVIEW
- Pearson r: +0.0013  (p=9.49e-01, Bonferroni: FAIL)
- Spearman r: -0.0013
- Cohen's d: 0.0026
- Mutual Info: 0.0048 bits
- Max cross-r: 0.5646 with 'entity_location_value_prec' (redundant: no)
- Mode consistency: n/a
- **Reason:** Detectable association but does not survive Bonferroni correction. Retain only if domain knowledge justifies inclusion. Pearson r=+0.0013 (p=9.49e-01, Bonferroni:FAIL), Spearman r=-0.0013, Cohen's d=0.0026, MI=0.0048 bits

## PCA Top Loadings (PC01-PC05)
| PC | Feature 1 | Loading | Feature 2 | Loading | Feature 3 | Loading |
|----|-----------|---------|-----------|---------|-----------|---------|
| PC01 | jaccard | +0.3548 | dice | +0.3537 | rouge | +0.3239 |
| PC02 | numeric_jaccard | +0.3968 | lcs_char | +0.3686 | entity_location_value_rec | +0.3555 |
| PC03 | entity_value_prec | +0.3877 | contradiction | -0.3254 | neutral | +0.3224 |
| PC04 | neutral | +0.5196 | entailment | -0.4588 | entity_product_value_prec | -0.4447 |
| PC05 | entity_percentage_value_rec | +0.6227 | entity_percentage_value_prec | +0.6157 | entailment | -0.2114 |

## Top-10 K-means Discriminative Features
| Feature | Centroid Distance |
|---------|-----------------|
| dice | 1.5733 |
| mixedbread-ai/mxbai-embed-large-v1 | 1.5361 |
| PREC_mixedbread-ai/mxbai-embed-large-v1 | 1.4971 |
| jaccard | 1.4851 |
| PREC_Qwen/Qwen3-Embedding-0.6B | 1.4727 |
| REC_mixedbread-ai/mxbai-embed-large-v1 | 1.3503 |
| rouge | 1.2405 |
| entity_value_rec | 1.1803 |
| lcs_token | 1.1572 |
| rouge3 | 0.9405 |

## Per-Mode Pearson r
| Feature | context-vs-generated |
|---------|---------|
| entailment | +0.4019 |
| contradiction | -0.3500 |
| entity_value_prec | +0.1353 |
| lcs_char | -0.1288 |
| rouge | -0.1171 |
| REC_mixedbread-ai/mxbai-embed-large-v1 | -0.1036 |
| dice | -0.1024 |
| lcs_token | -0.1008 |
| jaccard | -0.0950 |
| neutral | -0.0857 |
| mixedbread-ai/mxbai-embed-large-v1 | -0.0829 |
| PREC_mixedbread-ai/mxbai-embed-large-v1 | -0.0819 |
| entity_product_value_prec | +0.0707 |
| rouge3 | -0.0699 |
| PREC_Qwen/Qwen3-Embedding-0.6B | -0.0645 |
| entity_time_value_prec | +0.0512 |
| entity_location_value_prec | +0.0444 |
| entity_percentage_value_rec | +0.0376 |
| entity_product_value_rec | +0.0361 |
| numeric_jaccard | +0.0275 |
| entity_value_rec | -0.0236 |
| entity_time_value_rec | +0.0197 |
| entity_percentage_value_prec | +0.0123 |
| entity_location_value_rec | +0.0013 |