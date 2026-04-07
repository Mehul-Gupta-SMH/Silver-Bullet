# Ablation Experiment Report
Generated: 2026-04-07T20:52:50.503191  |  Tag: v5.0-cvg  |  Pairs: 2331  |  Missing: 0

## Run Parameters
- Mode filter: context-vs-generated
- Split filter: all
- Framework version: 1.0
- Bonferroni threshold: p < 3.8462e-03
- Redundancy threshold: max_cross_r >= 0.85

## Global Clustering Metrics
- K-means ARI (k=2): 0.0098
- PCA PC01 explained variance: 0.506  (cumulative PC05: 0.936)

## Verdict Summary
| Verdict | Count | Features |
|---------|-------|---------|
| KEEP | 5 | rouge3, entailment, neutral, contradiction, lcs_char |
| REVIEW | 8 | dice, rouge, jaccard, mixedbread-ai/mxbai-embed-large-v1, PREC_mixedbread-ai/mxbai-embed-large-v1, REC_mixedbread-ai/mxbai-embed-large-v1, PREC_Qwen/Qwen3-Embedding-0.6B, lcs_token |
| DROP | 0 | — |

## Tier Summary
| Tier | Count | Features |
|------|-------|---------|
| STRONG | 2 | entailment, contradiction |
| MODERATE | 1 | lcs_char |
| WEAK | 10 | dice, rouge3, rouge, jaccard, mixedbread-ai/mxbai-embed-large-v1, PREC_mixedbread-ai/mxbai-embed-large-v1, REC_mixedbread-ai/mxbai-embed-large-v1, PREC_Qwen/Qwen3-Embedding-0.6B, neutral, lcs_token |
| MARGINAL | 0 | — |
| NOISE | 0 | — |

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

### lcs_char -- MODERATE -- KEEP
- Pearson r: -0.1288  (p=4.39e-10, Bonferroni: OK)
- Spearman r: -0.1556
- Cohen's d: -0.2596
- Mutual Info: 0.0249 bits
- Max cross-r: 0.7795 with 'lcs_token' (redundant: no)
- Mode consistency: n/a
- **Reason:** Moderate validated signal with independent information. Pearson r=-0.1288 (p=4.39e-10, Bonferroni:OK), Spearman r=-0.1556, Cohen's d=-0.2596, MI=0.0249 bits

### rouge -- WEAK -- REVIEW
- Pearson r: -0.1171  (p=1.41e-08, Bonferroni: OK)
- Spearman r: -0.1685
- Cohen's d: -0.2358
- Mutual Info: 0.0701 bits
- Max cross-r: 0.8706 with 'lcs_token' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'lcs_token'. Ablation test before dropping. Pearson r=-0.1171 (p=1.41e-08, Bonferroni:OK), Spearman r=-0.1685, Cohen's d=-0.2358, MI=0.0701 bits. REDUNDANT: max_cross_r=0.8706 with 'lcs_token' (cross_r=+0.8706)

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
- Mutual Info: 0.0458 bits
- Max cross-r: 0.9815 with 'jaccard' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'jaccard'. Ablation test before dropping. Pearson r=-0.1024 (p=7.22e-07, Bonferroni:OK), Spearman r=-0.1146, Cohen's d=-0.2058, MI=0.0458 bits. REDUNDANT: max_cross_r=0.9815 with 'jaccard' (cross_r=+0.9815)

### lcs_token -- WEAK -- REVIEW
- Pearson r: -0.1008  (p=1.07e-06, Bonferroni: OK)
- Spearman r: -0.1185
- Cohen's d: -0.2026
- Mutual Info: 0.0426 bits
- Max cross-r: 0.8706 with 'rouge' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'rouge'. Ablation test before dropping. Pearson r=-0.1008 (p=1.07e-06, Bonferroni:OK), Spearman r=-0.1185, Cohen's d=-0.2026, MI=0.0426 bits. REDUNDANT: max_cross_r=0.8706 with 'rouge' (cross_r=+0.8706)

### jaccard -- WEAK -- REVIEW
- Pearson r: -0.0950  (p=4.30e-06, Bonferroni: OK)
- Spearman r: -0.1127
- Cohen's d: -0.1909
- Mutual Info: 0.0355 bits
- Max cross-r: 0.9815 with 'dice' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'dice'. Ablation test before dropping. Pearson r=-0.0950 (p=4.30e-06, Bonferroni:OK), Spearman r=-0.1127, Cohen's d=-0.1909, MI=0.0355 bits. REDUNDANT: max_cross_r=0.9815 with 'dice' (cross_r=+0.9815)

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
- Mutual Info: 0.0473 bits
- Max cross-r: 0.9254 with 'mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'mixedbread-ai/mxbai-embed-large-v1'. Ablation test before dropping. Pearson r=-0.0819 (p=7.50e-05, Bonferroni:OK), Spearman r=-0.0891, Cohen's d=-0.1643, MI=0.0473 bits. REDUNDANT: max_cross_r=0.9254 with 'mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.9254)

### rouge3 -- WEAK -- KEEP
- Pearson r: -0.0699  (p=7.27e-04, Bonferroni: OK)
- Spearman r: +0.0211
- Cohen's d: -0.1402
- Mutual Info: 0.0182 bits
- Max cross-r: 0.8146 with 'rouge' (redundant: no)
- Mode consistency: n/a
- **Reason:** Small but validated independent signal. Pearson r=-0.0699 (p=7.27e-04, Bonferroni:OK), Spearman r=+0.0211, Cohen's d=-0.1402, MI=0.0182 bits

### PREC_Qwen/Qwen3-Embedding-0.6B -- WEAK -- REVIEW
- Pearson r: -0.0645  (p=1.84e-03, Bonferroni: OK)
- Spearman r: -0.0679
- Cohen's d: -0.1292
- Mutual Info: 0.0218 bits
- Max cross-r: 0.8762 with 'PREC_mixedbread-ai/mxbai-embed-large-v1' (redundant: yes)
- Mode consistency: n/a
- **Reason:** Small signal but near-duplicate (max_cross_r>=0.85) of higher-tier feature 'PREC_mixedbread-ai/mxbai-embed-large-v1'. Ablation test before dropping. Pearson r=-0.0645 (p=1.84e-03, Bonferroni:OK), Spearman r=-0.0679, Cohen's d=-0.1292, MI=0.0218 bits. REDUNDANT: max_cross_r=0.8762 with 'PREC_mixedbread-ai/mxbai-embed-large-v1' (cross_r=+0.8762)

## PCA Top Loadings (PC01-PC05)
| PC | Feature 1 | Loading | Feature 2 | Loading | Feature 3 | Loading |
|----|-----------|---------|-----------|---------|-----------|---------|
| PC01 | jaccard | +0.3688 | dice | +0.3675 | rouge | +0.3355 |
| PC02 | REC_mixedbread-ai/mxbai-embed-large-v1 | +0.3868 | lcs_char | -0.3764 | mixedbread-ai/mxbai-embed-large-v1 | +0.3618 |
| PC03 | neutral | +0.6869 | entailment | -0.6862 | lcs_char | +0.2070 |
| PC04 | contradiction | +0.7740 | entailment | -0.3739 | neutral | -0.2974 |
| PC05 | lcs_char | +0.6730 | rouge3 | -0.5669 | PREC_Qwen/Qwen3-Embedding-0.6B | +0.2991 |

## Top-10 K-means Discriminative Features
| Feature | Centroid Distance |
|---------|-----------------|
| dice | 1.5671 |
| mixedbread-ai/mxbai-embed-large-v1 | 1.5268 |
| PREC_mixedbread-ai/mxbai-embed-large-v1 | 1.5104 |
| PREC_Qwen/Qwen3-Embedding-0.6B | 1.5046 |
| jaccard | 1.4760 |
| REC_mixedbread-ai/mxbai-embed-large-v1 | 1.3427 |
| rouge | 1.2231 |
| lcs_token | 1.1422 |
| rouge3 | 0.9389 |
| lcs_char | 0.8090 |

## Per-Mode Pearson r
| Feature | context-vs-generated |
|---------|---------|
| entailment | +0.4019 |
| contradiction | -0.3500 |
| lcs_char | -0.1288 |
| rouge | -0.1171 |
| REC_mixedbread-ai/mxbai-embed-large-v1 | -0.1036 |
| dice | -0.1024 |
| lcs_token | -0.1008 |
| jaccard | -0.0950 |
| neutral | -0.0857 |
| mixedbread-ai/mxbai-embed-large-v1 | -0.0829 |
| PREC_mixedbread-ai/mxbai-embed-large-v1 | -0.0819 |
| rouge3 | -0.0699 |
| PREC_Qwen/Qwen3-Embedding-0.6B | -0.0645 |