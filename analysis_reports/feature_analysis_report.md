# Feature Pattern Analysis — SilverBullet

Analysis of feature signatures for **confident correct** vs **confident wrong** predictions.
- Confident correct: probability ≥ 0.85 (label=1) or ≤ 0.15 (label=0)
- Confident wrong: label=1 but probability < 0.35, or label=0 but probability > 0.65
- Scalar per feature: mean of top-5 values in the n×m feature matrix
- Delta = mean(correct) − mean(wrong); flagged if |delta| > 0.15


---

## context-vs-generated

- Test samples: 531
- Confident correct: 50 (cache found: 50, missing: 0)
- Confident wrong: 50 (cache found: 50, missing: 0)


### Feature Comparison — All Labels

| Feature | Mean Correct | Std C | Mean Wrong | Std W | Delta (C-W) | Cohen's d | Flagged |
|---------|-------------|-------|------------|-------|-------------|-----------|---------|
| `neutral` | 0.2516 | 0.3387 | 0.6219 | 0.3829 | -0.3703 | -1.024 | **YES** |
| `contradiction` | 0.5600 | 0.4342 | 0.4051 | 0.3687 | 0.1549 | 0.385 | **YES** |
| `numeric_jaccard` | 0.5670 | 0.4722 | 0.4157 | 0.4464 | 0.1513 | 0.329 | **YES** |
| `entailment` | 0.4554 | 0.4620 | 0.3138 | 0.3635 | 0.1416 | 0.341 |  |
| `dice` | 0.4452 | 0.1813 | 0.3141 | 0.2032 | 0.1311 | 0.681 |  |
| `entity_product_value_rec` | 0.9700 | 0.1568 | 0.8400 | 0.3703 | 0.1300 | 0.457 |  |
| `entity_location_value_rec` | 0.6953 | 0.4388 | 0.8200 | 0.3747 | -0.1247 | -0.306 |  |
| `entity_product` | 0.9700 | 0.1568 | 0.8500 | 0.3536 | 0.1200 | 0.439 |  |
| `rouge` | 0.3472 | 0.2263 | 0.2462 | 0.2475 | 0.1010 | 0.426 |  |
| `jaccard` | 0.3053 | 0.1680 | 0.2070 | 0.1763 | 0.0982 | 0.571 |  |
| `entity_product_value_prec` | 0.9750 | 0.1263 | 0.8800 | 0.3283 | 0.0950 | 0.382 |  |
| `entity_value_prec` | 0.6909 | 0.3689 | 0.5992 | 0.4317 | 0.0917 | 0.228 |  |
| `lcs_token` | 0.2487 | 0.1945 | 0.1605 | 0.1775 | 0.0882 | 0.474 |  |
| `entity_value_rec` | 0.4074 | 0.3346 | 0.3212 | 0.3776 | 0.0862 | 0.242 |  |
| `rouge3` | 0.1831 | 0.1861 | 0.0977 | 0.1403 | 0.0854 | 0.518 |  |
| `entity_location` | 0.7720 | 0.4014 | 0.8500 | 0.3091 | -0.0780 | -0.218 |  |
| `entity_duration` | 1.0000 | 0.0000 | 0.9267 | 0.2546 | 0.0733 | 0.407 |  |
| `entity_duration_value_rec` | 1.0000 | 0.0000 | 0.9267 | 0.2546 | 0.0733 | 0.407 |  |
| `entity_location_value_prec` | 0.8700 | 0.3182 | 0.9400 | 0.2399 | -0.0700 | -0.248 |  |
| `entity_law` | 0.9933 | 0.0471 | 0.9400 | 0.2399 | 0.0533 | 0.309 |  |
| `lcs_char` | 0.3427 | 0.2169 | 0.2899 | 0.1941 | 0.0528 | 0.257 |  |
| `entity_date_value_prec` | 0.9600 | 0.1979 | 1.0000 | 0.0000 | -0.0400 | -0.286 |  |
| `entity_time_value_prec` | 0.9200 | 0.2740 | 0.9600 | 0.1979 | -0.0400 | -0.167 |  |
| `entity_duration_value_prec` | 1.0000 | 0.0000 | 0.9600 | 0.1979 | 0.0400 | 0.286 |  |
| `entity_date_value_rec` | 0.8191 | 0.3877 | 0.8400 | 0.3703 | -0.0209 | -0.055 |  |
| `entity_time_value_rec` | 0.8000 | 0.4041 | 0.8200 | 0.3881 | -0.0200 | -0.050 |  |
| `entity_percentage_value_rec` | 1.0000 | 0.0000 | 0.9800 | 0.1414 | 0.0200 | 0.200 |  |
| `mixedbread-ai/mxbai-embed-large-v1` | 0.7698 | 0.1385 | 0.7577 | 0.1608 | 0.0121 | 0.080 |  |
| `PREC_mixedbread-ai/mxbai-embed-large-v1` | 0.7699 | 0.1384 | 0.7581 | 0.1610 | 0.0118 | 0.079 |  |
| `REC_mixedbread-ai/mxbai-embed-large-v1` | 0.7698 | 0.1385 | 0.7583 | 0.1611 | 0.0115 | 0.076 |  |
| `entity_time` | 0.8500 | 0.3536 | 0.8533 | 0.3510 | -0.0033 | -0.009 |  |
| `PREC_Qwen/Qwen3-Embedding-0.6B` | 0.7325 | 0.1521 | 0.7304 | 0.1591 | 0.0021 | 0.013 |  |
| `entity_percentage` | 1.0000 | 0.0000 | 0.9991 | 0.0066 | 0.0009 | 0.200 |  |
| `entity_percentage_value_prec` | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |  |

### Label=1 Failures (Model Predicted 0)

| Feature | Mean Correct | Std C | Mean Wrong | Std W | Delta (C-W) | Cohen's d | Flagged |
|---------|-------------|-------|------------|-------|-------------|-----------|---------|
| `entailment` | 0.7853 | 0.3310 | 0.2125 | 0.2904 | 0.5728 | 1.840 | **YES** |
| `contradiction` | 0.1626 | 0.2715 | 0.4950 | 0.3433 | -0.3324 | -1.074 | **YES** |
| `neutral` | 0.4644 | 0.3872 | 0.6920 | 0.3645 | -0.2276 | -0.605 | **YES** |
| `entity_value_prec` | 0.7743 | 0.3914 | 0.5685 | 0.4497 | 0.2059 | 0.488 | **YES** |
| `numeric_jaccard` | 0.6377 | 0.4748 | 0.4648 | 0.4777 | 0.1730 | 0.363 | **YES** |
| `dice` | 0.4000 | 0.2234 | 0.2493 | 0.1690 | 0.1507 | 0.761 | **YES** |
| `jaccard` | 0.2745 | 0.1807 | 0.1545 | 0.1239 | 0.1200 | 0.774 |  |
| `lcs_token` | 0.2225 | 0.1736 | 0.1215 | 0.1176 | 0.1011 | 0.682 |  |
| `entity_value_rec` | 0.4276 | 0.3990 | 0.3413 | 0.4129 | 0.0863 | 0.213 |  |
| `rouge` | 0.3031 | 0.2309 | 0.2189 | 0.2371 | 0.0842 | 0.360 |  |
| `rouge3` | 0.1609 | 0.1680 | 0.0877 | 0.1425 | 0.0732 | 0.470 |  |
| `entity_duration` | 1.0000 | 0.0000 | 0.9286 | 0.2607 | 0.0714 | 0.388 |  |
| `entity_duration_value_rec` | 1.0000 | 0.0000 | 0.9286 | 0.2607 | 0.0714 | 0.388 |  |
| `PREC_Qwen/Qwen3-Embedding-0.6B` | 0.6411 | 0.2194 | 0.7005 | 0.1622 | -0.0593 | -0.308 |  |
| `entity_location_value_rec` | 0.7700 | 0.4191 | 0.8214 | 0.3797 | -0.0514 | -0.129 |  |
| `entity_date_value_rec` | 0.8991 | 0.3028 | 0.8571 | 0.3542 | 0.0419 | 0.127 |  |
| `REC_mixedbread-ai/mxbai-embed-large-v1` | 0.6993 | 0.1858 | 0.7354 | 0.1607 | -0.0361 | -0.208 |  |
| `PREC_mixedbread-ai/mxbai-embed-large-v1` | 0.6994 | 0.1858 | 0.7353 | 0.1607 | -0.0359 | -0.207 |  |
| `mixedbread-ai/mxbai-embed-large-v1` | 0.6993 | 0.1858 | 0.7347 | 0.1602 | -0.0354 | -0.204 |  |
| `entity_product` | 0.9400 | 0.2399 | 0.9048 | 0.2971 | 0.0352 | 0.131 |  |
| `entity_product_value_rec` | 0.9400 | 0.2399 | 0.9048 | 0.2971 | 0.0352 | 0.131 |  |
| `entity_time` | 0.8400 | 0.3703 | 0.8730 | 0.3289 | -0.0330 | -0.094 |  |
| `entity_location_value_prec` | 0.9000 | 0.3030 | 0.9286 | 0.2607 | -0.0286 | -0.101 |  |
| `entity_time_value_prec` | 1.0000 | 0.0000 | 0.9762 | 0.1543 | 0.0238 | 0.218 |  |
| `entity_duration_value_prec` | 1.0000 | 0.0000 | 0.9762 | 0.1543 | 0.0238 | 0.218 |  |
| `entity_percentage_value_rec` | 1.0000 | 0.0000 | 0.9762 | 0.1543 | 0.0238 | 0.218 |  |
| `lcs_char` | 0.2969 | 0.2124 | 0.2822 | 0.1572 | 0.0147 | 0.079 |  |
| `entity_location` | 0.8200 | 0.3747 | 0.8333 | 0.3272 | -0.0133 | -0.038 |  |
| `entity_time_value_rec` | 0.8200 | 0.3881 | 0.8333 | 0.3772 | -0.0133 | -0.035 |  |
| `entity_product_value_prec` | 0.9650 | 0.1750 | 0.9524 | 0.2155 | 0.0126 | 0.064 |  |
| `entity_law` | 0.9933 | 0.0471 | 1.0000 | 0.0000 | -0.0067 | -0.200 |  |
| `entity_percentage` | 1.0000 | 0.0000 | 0.9989 | 0.0072 | 0.0011 | 0.218 |  |
| `entity_date_value_prec` | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |  |
| `entity_percentage_value_prec` | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |  |

### Label=0 Failures (Model Predicted 1)

| Feature | Mean Correct | Std C | Mean Wrong | Std W | Delta (C-W) | Cohen's d | Flagged |
|---------|-------------|-------|------------|-------|-------------|-----------|---------|
| `contradiction` | 0.7602 | 0.3119 | 0.1574 | 0.2865 | 0.6028 | 2.013 | **YES** |
| `entailment` | 0.0345 | 0.0669 | 0.4640 | 0.4033 | -0.4295 | -1.486 | **YES** |
| `entity_location` | 0.6496 | 0.4186 | 0.8542 | 0.3451 | -0.2045 | -0.533 | **YES** |
| `neutral` | 0.3965 | 0.4071 | 0.5927 | 0.3998 | -0.1962 | -0.486 | **YES** |
| `entity_location_value_rec` | 0.5953 | 0.4714 | 0.7708 | 0.4165 | -0.1755 | -0.395 | **YES** |
| `entity_value_prec` | 0.4871 | 0.3814 | 0.6586 | 0.3931 | -0.1715 | -0.443 | **YES** |
| `entity_location_value_prec` | 0.7667 | 0.3854 | 0.9167 | 0.2823 | -0.1500 | -0.444 |  |
| `entity_product` | 0.9900 | 0.0707 | 0.8522 | 0.3444 | 0.1378 | 0.554 |  |
| `entity_law` | 1.0000 | 0.0000 | 0.8750 | 0.3378 | 0.1250 | 0.523 |  |
| `entity_product_value_rec` | 0.9500 | 0.2082 | 0.8314 | 0.3799 | 0.1186 | 0.387 |  |
| `entity_product_value_prec` | 0.9500 | 0.2082 | 0.8333 | 0.3807 | 0.1167 | 0.380 |  |
| `rouge` | 0.3120 | 0.2322 | 0.2195 | 0.2481 | 0.0925 | 0.385 |  |
| `entity_time_value_prec` | 0.8800 | 0.3283 | 0.9583 | 0.2041 | -0.0783 | -0.287 |  |
| `rouge3` | 0.1523 | 0.2086 | 0.0810 | 0.1078 | 0.0713 | 0.430 |  |
| `lcs_char` | 0.3446 | 0.2123 | 0.2827 | 0.2175 | 0.0619 | 0.288 |  |
| `lcs_token` | 0.2318 | 0.2118 | 0.1738 | 0.2204 | 0.0579 | 0.268 |  |
| `entity_value_rec` | 0.2724 | 0.2979 | 0.3276 | 0.3594 | -0.0551 | -0.167 |  |
| `entity_time_value_rec` | 0.7791 | 0.4180 | 0.8333 | 0.3807 | -0.0543 | -0.136 |  |
| `entity_duration` | 0.9800 | 0.1414 | 0.9306 | 0.2404 | 0.0494 | 0.251 |  |
| `entity_duration_value_rec` | 0.9800 | 0.1414 | 0.9306 | 0.2404 | 0.0494 | 0.251 |  |
| `entity_date_value_rec` | 0.7867 | 0.4083 | 0.8333 | 0.3807 | -0.0467 | -0.118 |  |
| `dice` | 0.3832 | 0.1797 | 0.3411 | 0.2444 | 0.0420 | 0.196 |  |
| `numeric_jaccard` | 0.4710 | 0.4852 | 0.4939 | 0.4355 | -0.0230 | -0.050 |  |
| `entity_duration_value_prec` | 0.9800 | 0.1414 | 0.9583 | 0.2041 | 0.0217 | 0.123 |  |
| `jaccard` | 0.2557 | 0.1749 | 0.2354 | 0.2174 | 0.0203 | 0.103 |  |
| `PREC_Qwen/Qwen3-Embedding-0.6B` | 0.7568 | 0.1036 | 0.7449 | 0.1604 | 0.0119 | 0.088 |  |
| `entity_time` | 0.8424 | 0.3534 | 0.8333 | 0.3807 | 0.0091 | 0.025 |  |
| `REC_mixedbread-ai/mxbai-embed-large-v1` | 0.7761 | 0.0954 | 0.7686 | 0.1763 | 0.0075 | 0.053 |  |
| `mixedbread-ai/mxbai-embed-large-v1` | 0.7759 | 0.0954 | 0.7686 | 0.1763 | 0.0073 | 0.052 |  |
| `PREC_mixedbread-ai/mxbai-embed-large-v1` | 0.7760 | 0.0953 | 0.7687 | 0.1764 | 0.0072 | 0.051 |  |
| `entity_date_value_prec` | 0.9600 | 0.1979 | 0.9583 | 0.2041 | 0.0017 | 0.008 |  |
| `entity_percentage` | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |  |
| `entity_percentage_value_prec` | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |  |
| `entity_percentage_value_rec` | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |  |

### Top-20 Confident-Wrong Cases

#### Case 1 — label=1 (model said 0), prob=0.0219
**text1:** This study was conducted to estimate the incidence and clinical predictors of post-thoracotomy shoulder pain and to determine the effectiveness of thoracic epidural block in alleviating this pain. A p

**text2:** It is concluded that post-thoracotomy shoulder pain is a common problem, and the previously mentioned variables did not predict its appearance. Thoracic epidural block is effective in the treatment of

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `numeric_jaccard` | 0.0000 | ↓ 0.5670 |
| `neutral` | 0.4218 | ↑ 0.1702 |
| `contradiction` | 0.5737 | ↑ 0.0137 |

#### Case 2 — label=1 (model said 0), prob=0.0476
**text1:** Twelve months ago, Liverpool were a free-flowing, potent attacking force with the SS (Luis Suarez and Daniel Sturridge) firing on all cylinders - the pair would eventually end up scoring 52 of Liverpo

**text2:** Raheem Sterling, Jordan Henderson and Steven Gerrard are Liverpool's top scorers in the Premier League this season with six goals each. Seventeen of 20 Premier League clubs have top scorers with more 

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `numeric_jaccard` | 0.0294 | ↓ 0.5376 |
| `contradiction` | 0.7747 | ↑ 0.2148 |
| `neutral` | 0.2118 | ↓ 0.0399 |

#### Case 3 — label=0 (model said 1), prob=0.9437
**text1:** Cat's Cradle has genre Speculative fictionSpeculative fiction has examples: Zoe's TaleZoe's Tale is written by John Scalzi

**text2:** Great. You can try Game of Thrones. It is written by John Scalzi and also is speculative fiction.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `numeric_jaccard` | 1.0000 | ↑ 0.4330 |
| `contradiction` | 0.7881 | ↑ 0.2281 |
| `neutral` | 0.2905 | ↑ 0.0389 |

#### Case 4 — label=0 (model said 1), prob=0.9433
**text1:** Richard Tucholka (February 9, 1954 - April 27, 2017) was a writer, game designer and publisher, best known for his work in the creation of the role-playing games "Fringeworthy" & "".Fringeworthy is a 

**text2:** Richard Tucholka is best known for his work in creating the role-playing game first published in 1982 by someone.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `contradiction` | 0.0074 | ↓ 0.5526 |
| `numeric_jaccard` | 0.2000 | ↓ 0.3670 |
| `neutral` | 0.0133 | ↓ 0.2384 |

#### Case 5 — label=1 (model said 0), prob=0.0600
**text1:** Radiotherapy reduces local recurrence rates but is also capable of short- and long-term toxicity. It may also render treatment of local recurrence more challenging if it develops despite previous radi

**text2:** Patients who previously received radiotherapy for primary rectal cancer treatment have worse oncologic outcomes than those who had not received radiotherapy after pelvic exenteration for locally recur

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `numeric_jaccard` | 0.0000 | ↓ 0.5670 |
| `neutral` | 0.3183 | ↑ 0.0667 |
| `contradiction` | 0.6250 | ↑ 0.0650 |

#### Case 6 — label=1 (model said 0), prob=0.0628
**text1:** Rhonda Byrne wrote The Secret. The Secret has genre Documentary film

**text2:** Of course, she also wrote The Magic which is (The Secret, #2), have you read that?

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `neutral` | 0.9720 | ↑ 0.7204 |
| `numeric_jaccard` | 0.0000 | ↓ 0.5670 |
| `contradiction` | 0.1162 | ↓ 0.4438 |

#### Case 7 — label=1 (model said 0), prob=0.0755
**text1:** An Air Canada flight from Germany to Toronto was forced to divert to Shannon Airport in Ireland last night after an 87-year-old woman caused a disturbance on board. The pensioner, who was travelling i

**text2:** Air Canada flight ACA-877 set out from Frankfurt Airport, Germany. Pilot contacted Shannon after disturbance caused on board. Irish police confirm 87-year-old woman was taken into custody and then rel

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `numeric_jaccard` | 0.0833 | ↓ 0.4836 |
| `contradiction` | 0.9631 | ↑ 0.4031 |
| `neutral` | 0.0699 | ↓ 0.1817 |

#### Case 8 — label=1 (model said 0), prob=0.1005
**text1:** We evaluated the association of perineural invasion with disease progression in men with prostate cancer on active surveillance. We retrospectively analyzed the records of 302 men on active surveillan

**text2:** Among patients with prostate cancer on active surveillance perineural invasion was associated with an increased risk of clinical progression. The 2-year risk of clinical progression with perineural in

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `numeric_jaccard` | 0.0251 | ↓ 0.5419 |
| `neutral` | 0.5412 | ↑ 0.2896 |
| `contradiction` | 0.6993 | ↑ 0.1393 |

#### Case 9 — label=0 (model said 1), prob=0.8954
**text1:** Gary Teale has hit back at Ronny Deila’s criticism of the St Mirren Park pitch. After the Parkhead club’s 2-0 win on Friday night, Deila branded most surfaces in the Scottish Premiership ‘terrible’ an

**text2:** Gary Teale has praised the award-winning groundsman, Tommy Docherty, for maintaining the best pitch in the Scottish Premiership, despite Ronny Deila's criticism after Celtic's 2-0 win over St Mirren.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `contradiction` | 0.0026 | ↓ 0.5574 |
| `neutral` | 0.0209 | ↓ 0.2307 |
| `numeric_jaccard` | 0.5000 | ↓ 0.0670 |

#### Case 10 — label=1 (model said 0), prob=0.1130
**text1:** Memoirs of a Geisha is written by Robin Swicord

**text2:** Memoir of a Geisha was produced by Lucy Fisher and distribute by Columbia Pictures. It starred  Zhang Ziyi, Michelle Yeoh, and Gong Li.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `neutral` | 0.9593 | ↑ 0.7076 |
| `numeric_jaccard` | 1.0000 | ↑ 0.4330 |
| `contradiction` | 0.2121 | ↓ 0.3479 |

#### Case 11 — label=1 (model said 0), prob=0.1205
**text1:** {'name': 'Taqueria La Colmena', 'address': '217 Milpas St', 'city': 'Santa Barbara', 'state': 'CA', 'categories': 'Mexican, Restaurants', 'hours': {'Monday': '9:0-21:0', 'Tuesday': '9:0-21:0', 'Wednes

**text2:** Based on the provided structured data, here is an objective overview of Taqueria La Colmena:

Taqueria La Colmena is a Mexican restaurant located in Santa Barbara, California. The restaurant has a rat

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `neutral` | 0.9943 | ↑ 0.7427 |
| `numeric_jaccard` | 0.0333 | ↓ 0.5336 |
| `contradiction` | 0.1712 | ↓ 0.3888 |

#### Case 12 — label=0 (model said 1), prob=0.8762
**text1:** Lisandro Alonso (born 2 June 1975) is an Argentine film director and screenwriter.Jauja is a 2014 internationally co-produced historical drama film co-written and directed by Lisandro Alonso.

**text2:** Jauja, the historical drama film, was directed by a Peruvian.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `numeric_jaccard` | 0.0000 | ↓ 0.5670 |
| `contradiction` | 0.9973 | ↑ 0.4373 |
| `neutral` | 0.5830 | ↑ 0.3314 |

#### Case 13 — label=1 (model said 0), prob=0.1298
**text1:** Christopher McQuarrie directed The Way of the Gun. The Way of the Gun is starring Nicky Katt

**text2:** The Brave One, Secondhand Lions...

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `numeric_jaccard` | 1.0000 | ↑ 0.4330 |
| `contradiction` | 0.9644 | ↑ 0.4045 |
| `neutral` | 0.0478 | ↓ 0.2039 |

#### Case 14 — label=1 (model said 0), prob=0.1320
**text1:** snails are detrimental to plants. aim your experiment at exterminating them. they like beer and will gather at a shallow dish of beer set out in the garden. you can coll … ect them there. or you can m

**text2:** Snails play several roles in the environment. They serve as a food source for various animals like birds, turtles, fishes, insects, and crocodiles. They also contribute to the cycling of organic matte

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `neutral` | 0.9749 | ↑ 0.7233 |
| `numeric_jaccard` | 1.0000 | ↑ 0.4330 |
| `contradiction` | 0.2947 | ↓ 0.2653 |

#### Case 15 — label=1 (model said 0), prob=0.1455
**text1:** Because your calorie maintenance level already is how many calories are needed for you to maintain your current weight. You already did 99% of the work in that last step when you calculated what yours

**text2:** Based on the provided passages, the number of calories needed to maintain your current weight can be calculated using the following steps:

For women aged 19-30:

* Calculate your basal metabolic rate

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `neutral` | 0.9944 | ↑ 0.7428 |
| `numeric_jaccard` | 0.1258 | ↓ 0.4412 |
| `contradiction` | 0.9613 | ↑ 0.4013 |

#### Case 16 — label=0 (model said 1), prob=0.8396
**text1:** Vineyard wins Island Cup on last-second field goal
( Oct. 29, 2023 ) The Whalers twice took two touchdown leads Saturday against Martha’s Vineyard and appeared primed to reclaim the Island Cup but the

**text2:** Sure! Here is the summary within 111 words:

Nantucket led 14-0 early but Martha's Vineyard clawed back to tie the game with a late field goal. The Vineyarders won 24-21 on the last play of the game. 

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `neutral` | 0.9533 | ↑ 0.7016 |
| `contradiction` | 0.0549 | ↓ 0.5051 |
| `numeric_jaccard` | 0.1667 | ↓ 0.4003 |

#### Case 17 — label=1 (model said 0), prob=0.1686
**text1:** She is known for starring in the title role of the children's television series "The Secret World of Alex Mack" during the mid-1990s.The Secret World of Alex Mack is an American television series that

**text2:** Clarissa Explains It All

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `numeric_jaccard` | 0.0000 | ↓ 0.5670 |
| `neutral` | 0.0617 | ↓ 0.1900 |
| `contradiction` | 0.7104 | ↑ 0.1504 |

#### Case 18 — label=1 (model said 0), prob=0.1722
**text1:** Quantum of Solace is starring Tim Pigott-Smith

**text2:** Joseph Andrews is a romantic comedy film you might enjoy.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `numeric_jaccard` | 1.0000 | ↑ 0.4330 |
| `contradiction` | 0.8639 | ↑ 0.3039 |
| `neutral` | 0.1259 | ↓ 0.1258 |

#### Case 19 — label=1 (model said 0), prob=0.1801
**text1:** Tahereh Mafi wrote Ignite Me (Shatter Me, #3)Ignite Me (Shatter Me, #3) was released 2014. 2014 was the year that released The World of Ice and Fire2014 was the year that released You. You is written 

**text2:** You are correct! A few other books that were released that year was The World of Ice and Fire by George R.R. Martin and You by Caroline Kepnes. Do any of those interest you?

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `neutral` | 0.9955 | ↑ 0.7438 |
| `numeric_jaccard` | 1.0000 | ↑ 0.4330 |
| `contradiction` | 0.9584 | ↑ 0.3984 |

#### Case 20 — label=0 (model said 1), prob=0.8163
**text1:** Jessica Alba and Michael Weatherly reprised their roles from the series, voicing Max Guevara and Logan Cale respectively. She has won various awards for her acting, including the Choice Actress Teen C

**text2:** Jessica Alba won an award for her acting in the movie.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `contradiction` | 0.0070 | ↓ 0.5529 |
| `numeric_jaccard` | 1.0000 | ↑ 0.4330 |
| `neutral` | 0.0760 | ↓ 0.1757 |


---

## reference-vs-generated

- Test samples: 388
- Confident correct: 50 (cache found: 50, missing: 0)
- Confident wrong: 50 (cache found: 50, missing: 0)


### Feature Comparison — All Labels

| Feature | Mean Correct | Std C | Mean Wrong | Std W | Delta (C-W) | Cohen's d | Flagged |
|---------|-------------|-------|------------|-------|-------------|-----------|---------|
| `neutral` | 0.0216 | 0.0326 | 0.6516 | 0.3826 | -0.6300 | -2.320 | **YES** |
| `entity_value_prec` | 0.8847 | 0.2670 | 0.3768 | 0.4532 | 0.5079 | 1.365 | **YES** |
| `contradiction` | 0.6203 | 0.4683 | 0.1208 | 0.2276 | 0.4995 | 1.357 | **YES** |
| `entity_value_rec` | 0.6648 | 0.3977 | 0.3033 | 0.4210 | 0.3615 | 0.883 | **YES** |
| `numeric_jaccard` | 0.8749 | 0.3230 | 0.5677 | 0.4826 | 0.3072 | 0.748 | **YES** |
| `entity_location_value_rec` | 0.9000 | 0.3030 | 0.6000 | 0.4949 | 0.3000 | 0.731 | **YES** |
| `entity_location` | 0.8933 | 0.3045 | 0.6779 | 0.4592 | 0.2155 | 0.553 | **YES** |
| `lcs_char` | 0.5243 | 0.2234 | 0.3095 | 0.2489 | 0.2148 | 0.908 | **YES** |
| `entity_location_value_prec` | 0.9733 | 0.1482 | 0.7600 | 0.4314 | 0.2133 | 0.661 | **YES** |
| `rouge` | 0.4922 | 0.3068 | 0.2965 | 0.2906 | 0.1957 | 0.655 | **YES** |
| `dice` | 0.5194 | 0.2828 | 0.3279 | 0.2817 | 0.1914 | 0.678 | **YES** |
| `lcs_token` | 0.3822 | 0.2478 | 0.2091 | 0.2455 | 0.1731 | 0.702 | **YES** |
| `jaccard` | 0.4067 | 0.3032 | 0.2391 | 0.2643 | 0.1675 | 0.589 | **YES** |
| `entity_time_value_prec` | 1.0000 | 0.0000 | 0.8600 | 0.3505 | 0.1400 | 0.565 |  |
| `entity_time_value_rec` | 0.9500 | 0.2082 | 0.8200 | 0.3881 | 0.1300 | 0.417 |  |
| `PREC_Qwen/Qwen3-Embedding-0.6B` | 0.8006 | 0.1527 | 0.6714 | 0.1782 | 0.1291 | 0.778 |  |
| `entity_product` | 0.8300 | 0.3727 | 0.9287 | 0.2474 | -0.0988 | -0.312 |  |
| `entity_time` | 0.9500 | 0.2082 | 0.8600 | 0.3505 | 0.0900 | 0.312 |  |
| `mixedbread-ai/mxbai-embed-large-v1` | 0.8101 | 0.1558 | 0.7299 | 0.1389 | 0.0802 | 0.544 |  |
| `REC_mixedbread-ai/mxbai-embed-large-v1` | 0.8101 | 0.1558 | 0.7299 | 0.1389 | 0.0802 | 0.544 |  |
| `PREC_mixedbread-ai/mxbai-embed-large-v1` | 0.8101 | 0.1558 | 0.7305 | 0.1394 | 0.0797 | 0.539 |  |
| `entity_date_value_rec` | 0.9500 | 0.2082 | 0.8800 | 0.3283 | 0.0700 | 0.255 |  |
| `entity_product_value_rec` | 0.8400 | 0.3703 | 0.9000 | 0.3030 | -0.0600 | -0.177 |  |
| `entailment` | 0.3581 | 0.4672 | 0.3084 | 0.3712 | 0.0497 | 0.118 |  |
| `rouge3` | 0.1702 | 0.2303 | 0.1223 | 0.1991 | 0.0479 | 0.222 |  |
| `entity_percentage` | 1.0000 | 0.0000 | 0.9600 | 0.1979 | 0.0400 | 0.286 |  |
| `entity_product_value_prec` | 0.9400 | 0.2399 | 0.9000 | 0.3030 | 0.0400 | 0.146 |  |
| `entity_percentage_value_rec` | 1.0000 | 0.0000 | 0.9600 | 0.1979 | 0.0400 | 0.286 |  |
| `entity_duration_value_prec` | 0.9800 | 0.1414 | 1.0000 | 0.0000 | -0.0200 | -0.200 |  |
| `entity_percentage_value_prec` | 1.0000 | 0.0000 | 0.9800 | 0.1414 | 0.0200 | 0.200 |  |
| `entity_duration` | 0.9800 | 0.1414 | 0.9667 | 0.1684 | 0.0133 | 0.086 |  |
| `entity_date_value_prec` | 0.9700 | 0.1568 | 0.9600 | 0.1979 | 0.0100 | 0.056 |  |
| `entity_duration_value_rec` | 0.9600 | 0.1979 | 0.9667 | 0.1684 | -0.0067 | -0.036 |  |
| `entity_law` | 0.9800 | 0.1414 | 0.9800 | 0.1414 | 0.0000 | 0.000 |  |

### Label=1 Failures (Model Predicted 0)

| Feature | Mean Correct | Std C | Mean Wrong | Std W | Delta (C-W) | Cohen's d | Flagged |
|---------|-------------|-------|------------|-------|-------------|-----------|---------|
| `entailment` | 0.9535 | 0.0675 | 0.1226 | 0.1492 | 0.8309 | 7.175 | **YES** |
| `neutral` | 0.0434 | 0.0726 | 0.7890 | 0.2753 | -0.7456 | -3.703 | **YES** |
| `entity_value_prec` | 0.7017 | 0.4259 | 0.2245 | 0.3681 | 0.4772 | 1.199 | **YES** |
| `entity_value_rec` | 0.5569 | 0.4508 | 0.1774 | 0.3243 | 0.3795 | 0.967 | **YES** |
| `entity_location_value_rec` | 0.7800 | 0.4185 | 0.4872 | 0.5064 | 0.2928 | 0.630 | **YES** |
| `numeric_jaccard` | 0.7559 | 0.4241 | 0.5116 | 0.4885 | 0.2443 | 0.534 | **YES** |
| `entity_location_value_prec` | 0.9200 | 0.2740 | 0.6923 | 0.4676 | 0.2277 | 0.594 | **YES** |
| `entity_location` | 0.7950 | 0.3967 | 0.5870 | 0.4831 | 0.2080 | 0.471 | **YES** |
| `PREC_Qwen/Qwen3-Embedding-0.6B` | 0.7953 | 0.1428 | 0.6083 | 0.1390 | 0.1870 | 1.327 | **YES** |
| `dice` | 0.4076 | 0.2100 | 0.2217 | 0.1898 | 0.1859 | 0.929 | **YES** |
| `contradiction` | 0.0057 | 0.0098 | 0.1670 | 0.2561 | -0.1613 | -0.890 | **YES** |
| `entity_time_value_prec` | 0.9800 | 0.1414 | 0.8205 | 0.3888 | 0.1595 | 0.545 | **YES** |
| `lcs_token` | 0.2802 | 0.2126 | 0.1247 | 0.1634 | 0.1555 | 0.820 | **YES** |
| `lcs_char` | 0.3920 | 0.2286 | 0.2374 | 0.2092 | 0.1545 | 0.705 | **YES** |
| `entity_time_value_rec` | 0.9200 | 0.2740 | 0.7692 | 0.4268 | 0.1508 | 0.420 | **YES** |
| `jaccard` | 0.2787 | 0.1777 | 0.1398 | 0.1463 | 0.1390 | 0.854 |  |
| `mixedbread-ai/mxbai-embed-large-v1` | 0.8201 | 0.1153 | 0.6863 | 0.1018 | 0.1337 | 1.230 |  |
| `REC_mixedbread-ai/mxbai-embed-large-v1` | 0.8201 | 0.1153 | 0.6863 | 0.1018 | 0.1337 | 1.230 |  |
| `PREC_mixedbread-ai/mxbai-embed-large-v1` | 0.8201 | 0.1153 | 0.6871 | 0.1029 | 0.1330 | 1.217 |  |
| `rouge` | 0.3316 | 0.2336 | 0.2047 | 0.2138 | 0.1270 | 0.567 |  |
| `entity_time` | 0.9400 | 0.2399 | 0.8205 | 0.3888 | 0.1195 | 0.370 |  |
| `entity_product_value_prec` | 0.9800 | 0.1414 | 0.8718 | 0.3387 | 0.1082 | 0.417 |  |
| `entity_date_value_rec` | 0.9000 | 0.3030 | 0.8205 | 0.3888 | 0.0795 | 0.228 |  |
| `entity_percentage` | 1.0000 | 0.0000 | 0.9487 | 0.2235 | 0.0513 | 0.325 |  |
| `entity_percentage_value_rec` | 1.0000 | 0.0000 | 0.9487 | 0.2235 | 0.0513 | 0.325 |  |
| `entity_product_value_rec` | 0.9200 | 0.2740 | 0.8718 | 0.3387 | 0.0482 | 0.156 |  |
| `entity_date_value_prec` | 0.9600 | 0.1979 | 0.9231 | 0.2700 | 0.0369 | 0.156 |  |
| `entity_duration_value_rec` | 0.9400 | 0.2399 | 0.9744 | 0.1601 | -0.0344 | -0.168 |  |
| `rouge3` | 0.0912 | 0.1416 | 0.0598 | 0.1266 | 0.0314 | 0.234 |  |
| `entity_percentage_value_prec` | 1.0000 | 0.0000 | 0.9744 | 0.1601 | 0.0256 | 0.226 |  |
| `entity_duration_value_prec` | 0.9800 | 0.1414 | 1.0000 | 0.0000 | -0.0200 | -0.200 |  |
| `entity_duration` | 0.9600 | 0.1979 | 0.9744 | 0.1601 | -0.0144 | -0.080 |  |
| `entity_product` | 0.9167 | 0.2614 | 0.9087 | 0.2776 | 0.0080 | 0.030 |  |
| `entity_law` | 0.9800 | 0.1414 | 0.9744 | 0.1601 | 0.0056 | 0.037 |  |

### Label=0 Failures (Model Predicted 1)

| Feature | Mean Correct | Std C | Mean Wrong | Std W | Delta (C-W) | Cohen's d | Flagged |
|---------|-------------|-------|------------|-------|-------------|-----------|---------|
| `contradiction` | 0.9346 | 0.1259 | 0.0070 | 0.0074 | 0.9276 | 10.406 | **YES** |
| `entailment` | 0.0285 | 0.0929 | 0.9114 | 0.1336 | -0.8830 | -7.673 | **YES** |
| `numeric_jaccard` | 0.9600 | 0.1979 | 0.7083 | 0.4502 | 0.2517 | 0.724 | **YES** |
| `entity_product_value_rec` | 0.7500 | 0.4315 | 1.0000 | 0.0000 | -0.2500 | -0.819 | **YES** |
| `entity_product` | 0.7600 | 0.4194 | 1.0000 | 0.0000 | -0.2400 | -0.809 | **YES** |
| `entity_value_rec` | 0.5398 | 0.4395 | 0.7153 | 0.4300 | -0.1754 | -0.403 | **YES** |
| `dice` | 0.5170 | 0.3017 | 0.6542 | 0.2788 | -0.1372 | -0.472 |  |
| `jaccard` | 0.4115 | 0.3202 | 0.5464 | 0.3199 | -0.1349 | -0.421 |  |
| `rouge3` | 0.1887 | 0.2336 | 0.3157 | 0.2604 | -0.1270 | -0.513 |  |
| `neutral` | 0.0369 | 0.0834 | 0.1626 | 0.2935 | -0.1257 | -0.583 |  |
| `entity_location_value_rec` | 0.8800 | 0.3283 | 1.0000 | 0.0000 | -0.1200 | -0.517 |  |
| `entity_product_value_prec` | 0.8800 | 0.3283 | 1.0000 | 0.0000 | -0.1200 | -0.517 |  |
| `mixedbread-ai/mxbai-embed-large-v1` | 0.7577 | 0.1772 | 0.8741 | 0.1446 | -0.1164 | -0.720 |  |
| `PREC_mixedbread-ai/mxbai-embed-large-v1` | 0.7577 | 0.1772 | 0.8741 | 0.1446 | -0.1164 | -0.720 |  |
| `REC_mixedbread-ai/mxbai-embed-large-v1` | 0.7577 | 0.1772 | 0.8741 | 0.1446 | -0.1164 | -0.720 |  |
| `PREC_Qwen/Qwen3-Embedding-0.6B` | 0.7618 | 0.1603 | 0.8720 | 0.1304 | -0.1102 | -0.754 |  |
| `entity_value_prec` | 0.7530 | 0.3946 | 0.8611 | 0.3321 | -0.1081 | -0.296 |  |
| `entity_location` | 0.9133 | 0.2761 | 1.0000 | 0.0000 | -0.0867 | -0.444 |  |
| `lcs_token` | 0.3813 | 0.2524 | 0.4672 | 0.2858 | -0.0859 | -0.319 |  |
| `rouge` | 0.4878 | 0.3208 | 0.5726 | 0.3340 | -0.0848 | -0.259 |  |
| `entity_duration` | 1.0000 | 0.0000 | 0.9444 | 0.1925 | 0.0556 | 0.408 |  |
| `entity_duration_value_rec` | 1.0000 | 0.0000 | 0.9444 | 0.1925 | 0.0556 | 0.408 |  |
| `entity_location_value_prec` | 0.9533 | 0.2021 | 1.0000 | 0.0000 | -0.0467 | -0.326 |  |
| `entity_date_value_prec` | 0.9700 | 0.1568 | 1.0000 | 0.0000 | -0.0300 | -0.271 |  |
| `entity_date_value_rec` | 0.9700 | 0.1568 | 1.0000 | 0.0000 | -0.0300 | -0.271 |  |
| `entity_time_value_prec` | 0.9800 | 0.1414 | 1.0000 | 0.0000 | -0.0200 | -0.200 |  |
| `entity_time` | 0.9300 | 0.2476 | 0.9167 | 0.2887 | 0.0133 | 0.050 |  |
| `entity_time_value_rec` | 0.9300 | 0.2476 | 0.9167 | 0.2887 | 0.0133 | 0.050 |  |
| `lcs_char` | 0.5340 | 0.2199 | 0.5212 | 0.2533 | 0.0128 | 0.054 |  |
| `entity_law` | 0.9900 | 0.0707 | 1.0000 | 0.0000 | -0.0100 | -0.200 |  |
| `entity_percentage` | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |  |
| `entity_duration_value_prec` | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |  |
| `entity_percentage_value_prec` | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |  |
| `entity_percentage_value_rec` | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |  |

### Top-20 Confident-Wrong Cases

#### Case 1 — label=0 (model said 1), prob=0.9717
**text1:** A man is playing a wooden flute while several other men play bongo drums.

**text2:** A man is playing a flute.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `contradiction` | 0.0026 | ↓ 0.6176 |
| `entity_value_rec` | 1.0000 | ↑ 0.3352 |
| `lcs_char` | 0.3288 | ↓ 0.1955 |
| `numeric_jaccard` | 1.0000 | ↑ 0.1251 |
| `rouge` | 0.3750 | ↓ 0.1172 |

#### Case 2 — label=0 (model said 1), prob=0.9540
**text1:** The recommended pricing strategy sets rates 20% above the market median, targets enterprise accounts exclusively through direct sales and channel partners, and includes a value-based ROI calculator as

**text2:** The recommended pricing strategy sets rates 20% above the market median and targets enterprise accounts through direct sales and channel partners.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `contradiction` | 0.0021 | ↓ 0.6181 |
| `entity_value_rec` | 1.0000 | ↑ 0.3352 |
| `dice` | 0.7273 | ↑ 0.2079 |
| `lcs_token` | 0.5625 | ↑ 0.1803 |
| `jaccard` | 0.5714 | ↑ 0.1648 |

#### Case 3 — label=0 (model said 1), prob=0.9464
**text1:** The competitive response plan has three phases: a 90-day containment phase (price matching, accelerated roadmap), a six-month differentiation phase (exclusive features, enterprise SLAs), and a 12-mont

**text2:** The competitive response begins with a 90-day containment phase involving price matching and an accelerated product roadmap.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `contradiction` | 0.0045 | ↓ 0.6157 |
| `numeric_jaccard` | 0.5000 | ↓ 0.3749 |
| `entity_value_rec` | 0.3333 | ↓ 0.3315 |
| `rouge` | 0.2807 | ↓ 0.2115 |
| `lcs_char` | 0.3657 | ↓ 0.1586 |

#### Case 4 — label=1 (model said 0), prob=0.0623
**text1:** . What is your definition of "life force"?

**text2:** What is your definition of "soul"?

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `entity_value_rec` | 1.0000 | ↑ 0.3352 |
| `lcs_token` | 0.7143 | ↑ 0.3321 |
| `rouge` | 0.8000 | ↑ 0.3078 |
| `dice` | 0.8235 | ↑ 0.3042 |
| `jaccard` | 0.7000 | ↑ 0.2933 |

#### Case 5 — label=0 (model said 1), prob=0.9304
**text1:** The acquisition rationale rests on three factors: access to the target's proprietary technology, elimination of a key competitor, and capturing $40M in annual cost synergies through shared infrastruct

**text2:** The acquisition is justified by the target's proprietary technology and the opportunity to eliminate a direct competitor from the market.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `numeric_jaccard` | 0.0000 | ↓ 0.8749 |
| `entity_value_rec` | 0.0000 | ↓ 0.6648 |
| `contradiction` | 0.0029 | ↓ 0.6174 |
| `neutral` | 0.1997 | ↑ 0.1781 |
| `rouge` | 0.3158 | ↓ 0.1764 |

#### Case 6 — label=1 (model said 0), prob=0.0711
**text1:** A person is walking down a stone path.

**text2:** A man is walking down a sidewalk.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `entity_location_value_prec` | 0.0000 | ↓ 0.9733 |
| `entity_location_value_rec` | 0.0000 | ↓ 0.9000 |
| `entity_value_prec` | 0.0000 | ↓ 0.8847 |
| `entity_value_rec` | 0.0000 | ↓ 0.6648 |
| `lcs_token` | 0.6250 | ↑ 0.2428 |

#### Case 7 — label=0 (model said 1), prob=0.9276
**text1:** What process signals the need for the prime minister to resign in Australia?

**text2:** In Australia, the Prime Minister is expected to step down if s

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `contradiction` | 0.0117 | ↓ 0.6085 |
| `entity_value_rec` | 1.0000 | ↑ 0.3352 |
| `neutral` | 0.2917 | ↑ 0.2700 |
| `numeric_jaccard` | 1.0000 | ↑ 0.1251 |
| `entity_value_prec` | 1.0000 | ↑ 0.1153 |

#### Case 8 — label=1 (model said 0), prob=0.1825
**text1:** What book first contained the term biological diversity?

**text2:** A Different Kind of Country advocating conservation.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `entity_location_value_prec` | 0.0000 | ↓ 0.9733 |
| `entity_location_value_rec` | 0.0000 | ↓ 0.9000 |
| `entity_location` | 0.0000 | ↓ 0.8933 |
| `entity_value_prec` | 0.0000 | ↓ 0.8847 |
| `entity_value_rec` | 0.0000 | ↓ 0.6648 |

#### Case 9 — label=1 (model said 0), prob=0.2053
**text1:** What was considered unreliable?

**text2:** Airborne Interception radar (AI) was unreliable.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `neutral` | 0.9406 | ↑ 0.9189 |
| `entity_value_prec` | 0.0000 | ↓ 0.8847 |
| `entity_value_rec` | 0.0000 | ↓ 0.6648 |
| `contradiction` | 0.0560 | ↓ 0.5643 |
| `jaccard` | 0.1818 | ↓ 0.2248 |

#### Case 10 — label=1 (model said 0), prob=0.2100
**text1:** (meta data) TITLE: (HK elections) Nathan Law elected as youngest lawmaker; Ricky Wong falls short | The Standard (meta data) AUTHOR: The Standard (meta data) PUBLISHER: The Standard Aug 21, 17:0332°C6

**text2:** Law received 50,818 votes, the second-highest among all candidates for the six-seat Hong Kong Island constituency, and was elected.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `entity_location_value_prec` | 0.0000 | ↓ 0.9733 |
| `neutral` | 0.9767 | ↑ 0.9550 |
| `entity_location_value_rec` | 0.0000 | ↓ 0.9000 |
| `entity_location` | 0.0000 | ↓ 0.8933 |
| `entity_value_prec` | 0.0000 | ↓ 0.8847 |

#### Case 11 — label=1 (model said 0), prob=0.2112
**text1:** How to drive a turbo diesel car with manual transmission efficiently<br>Choose the right gas station to trial and stick with it for duration of the steps below. Not all diesel fuels are equal quality.

**text2:** The main topic of this article is automobiles.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `neutral` | 0.9422 | ↑ 0.9205 |
| `entity_location_value_rec` | 0.0000 | ↓ 0.9000 |
| `entity_location` | 0.0000 | ↓ 0.8933 |
| `entity_value_prec` | 0.0000 | ↓ 0.8847 |
| `entity_value_rec` | 0.0000 | ↓ 0.6648 |

#### Case 12 — label=0 (model said 1), prob=0.7843
**text1:** There are 4 playable characters , each with a unique ability and also a different combat style .

**text2:** There are 4 playable characters , each with a different ability and a unique fighting style .

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `contradiction` | 0.0022 | ↓ 0.6180 |
| `jaccard` | 0.8235 | ↑ 0.4169 |
| `rouge` | 0.8824 | ↑ 0.3902 |
| `dice` | 0.9032 | ↑ 0.3839 |
| `lcs_token` | 0.7647 | ↑ 0.3825 |

#### Case 13 — label=1 (model said 0), prob=0.2232
**text1:** A U.S. military transport plane carrying humanitarian aid meant for Venezuelans landed in the Colombian border city of Cucuta on Saturday, where food and medicine are being stored amid uncertainty ove

**text2:** America helps hispanic countries as stated in this article

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `entity_location_value_prec` | 0.0000 | ↓ 0.9733 |
| `entity_location_value_rec` | 0.0000 | ↓ 0.9000 |
| `entity_value_prec` | 0.0000 | ↓ 0.8847 |
| `neutral` | 0.8288 | ↑ 0.8071 |
| `entity_value_rec` | 0.0000 | ↓ 0.6648 |

#### Case 14 — label=1 (model said 0), prob=0.2300
**text1:** (meta data) TITLE: French striker smashes transfer record - News - Sheffield United Skip to main content Sheffield United badge - Link to home # Sheffield United Official club partner Fan Engagement S

**text2:** He graduated to Le Havre II where he impressed scoring 14 goals before making his debut for Le Havre first team, again scoring 14 goals in 34 games.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `neutral` | 0.9829 | ↑ 0.9613 |
| `entity_value_prec` | 0.0000 | ↓ 0.8847 |
| `numeric_jaccard` | 0.0741 | ↓ 0.8008 |
| `entity_value_rec` | 0.0000 | ↓ 0.6648 |
| `contradiction` | 0.0120 | ↓ 0.6083 |

#### Case 15 — label=1 (model said 0), prob=0.2319
**text1:** (meta data) TITLE: Sleeman Breweries – Better beer for all. Contact us Who we are Our Story Where we brew Your Career Latest News Contact usLegal See All Brand Are you of legal drinking age? Sorry, yo

**text2:** Two brands are manufactured: Upper Canada Lager, a German-style lager and Upper Canada Dark Ale, "with a robust malty character and a rich chestnut colour".

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `entity_value_prec` | 0.0000 | ↓ 0.8847 |
| `numeric_jaccard` | 0.0000 | ↓ 0.8749 |
| `entity_value_rec` | 0.0000 | ↓ 0.6648 |
| `neutral` | 0.5642 | ↑ 0.5425 |
| `lcs_char` | 0.0384 | ↓ 0.4859 |

#### Case 16 — label=1 (model said 0), prob=0.2342
**text1:** (meta data) TITLE: USSSP: Scoutmaster.org - COPE ## C.O.P.E. What is C.O.P.E. ? Challenging Outdoor Personal Experience Project C.O.P.E. is a series of inter-related events that challenge on an indivi

**text2:** It was championed by Pony Express Council Executive Parvin Bishop who expanded the program when he became Director of Program at the National Office.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `entity_location_value_rec` | 0.0000 | ↓ 0.9000 |
| `entity_location` | 0.0000 | ↓ 0.8933 |
| `numeric_jaccard` | 0.0000 | ↓ 0.8749 |
| `entity_value_prec` | 0.3333 | ↓ 0.5513 |
| `neutral` | 0.5462 | ↑ 0.5245 |

#### Case 17 — label=1 (model said 0), prob=0.2408
**text1:** That One Night is a 2008 Canadian comedy film directed, written and produced by Rick Alyea. This film stars Crystal Lowe, Amanda Crew and Sam Easton, who were all from the 2006 horror film "Final Dest

**text2:** Crystal Lowe has been in at least one horror movie and at least one comedy movie.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `neutral` | 0.9485 | ↑ 0.9268 |
| `numeric_jaccard` | 0.0000 | ↓ 0.8749 |
| `contradiction` | 0.0435 | ↓ 0.5767 |
| `entity_value_rec` | 0.2383 | ↓ 0.4266 |
| `rouge` | 0.1599 | ↓ 0.3323 |

#### Case 18 — label=1 (model said 0), prob=0.2411
**text1:** What liberal arts type colleges are in Cork?

**text2:** CIT also incorporates the Cork School of Music and Crawford College of Art and Design as constituent schools.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `neutral` | 0.9803 | ↑ 0.9587 |
| `entity_location_value_rec` | 0.0000 | ↓ 0.9000 |
| `entity_location` | 0.0000 | ↓ 0.8933 |
| `entity_value_prec` | 0.0000 | ↓ 0.8847 |
| `entity_value_rec` | 0.0000 | ↓ 0.6648 |

#### Case 19 — label=1 (model said 0), prob=0.2424
**text1:** In which season did Ford Motor Company become a sponsor of American Idol?

**text2:** The sponsorship deal cost around $10 million in season one, rising to $35 million by season 7, and between $50 to $60 million in season 10.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `neutral` | 0.9975 | ↑ 0.9758 |
| `entity_value_prec` | 0.0000 | ↓ 0.8847 |
| `numeric_jaccard` | 0.0000 | ↓ 0.8749 |
| `entity_value_rec` | 0.0000 | ↓ 0.6648 |
| `contradiction` | 0.0015 | ↓ 0.6188 |

#### Case 20 — label=1 (model said 0), prob=0.2431
**text1:** (meta data) TITLE: Khabar: Shanti – let the sensations wash over you About Us Business Directory Community Press Releases Home > Magazine > Around Town > Shanti – let the sensations wash over you # Sh

**text2:** Shanti premiered in 2004 and recently celebrated its 10th anniversary in 2014.

**Feature deviations (from correct group mean):**
| Feature | Value | Deviation from correct |
|---------|-------|----------------------|
| `neutral` | 0.9135 | ↑ 0.8919 |
| `entity_value_prec` | 0.2500 | ↓ 0.6347 |
| `contradiction` | 0.0140 | ↓ 0.6062 |
| `lcs_char` | 0.0201 | ↓ 0.5042 |
| `rouge` | 0.0109 | ↓ 0.4813 |


---

## Cross-Mode Observations

### Features flagged in both CVG and RVG (|delta| > 0.15)
| Feature | CVG delta | RVG delta |
|---------|-----------|-----------|
| `neutral` | -0.3703 | -0.6300 |
| `contradiction` | 0.1549 | 0.4995 |
| `numeric_jaccard` | 0.1513 | 0.3072 |

### Features flagged in CVG only
_None_


### Features flagged in RVG only
| Feature | RVG delta |
|---------|-----------|
| `entity_value_prec` | 0.5079 |
| `entity_value_rec` | 0.3615 |
| `entity_location_value_rec` | 0.3000 |
| `entity_location` | 0.2155 |
| `lcs_char` | 0.2148 |
| `entity_location_value_prec` | 0.2133 |
| `rouge` | 0.1957 |
| `dice` | 0.1914 |
| `lcs_token` | 0.1731 |
| `jaccard` | 0.1675 |

---

## Actionable Conclusions

The following observations are drawn from the feature comparison tables and failure case review.

### Most discriminative features by mode

**CVG (Context vs Generated):**
- `neutral`: delta=-0.3703, Cohen's d=-1.024
- `contradiction`: delta=0.1549, Cohen's d=0.385
- `numeric_jaccard`: delta=0.1513, Cohen's d=0.329
- `entailment`: delta=0.1416, Cohen's d=0.341
- `dice`: delta=0.1311, Cohen's d=0.681

**RVG (Reference vs Generated):**
- `neutral`: delta=-0.6300, Cohen's d=-2.32
- `entity_value_prec`: delta=0.5079, Cohen's d=1.365
- `contradiction`: delta=0.4995, Cohen's d=1.357
- `entity_value_rec`: delta=0.3615, Cohen's d=0.883
- `numeric_jaccard`: delta=0.3072, Cohen's d=0.748

### Feature improvement suggestions
1. **Features with high delta in wrong direction** (delta < −0.15): the model is over-relying on these features in the wrong direction — consider penalised feature selection or interaction terms.
2. **Features with low delta** (|delta| < 0.05): may be adding noise without signal — candidates for removal in next ablation.
3. **NLI features** (entailment/neutral/contradiction): consistently the strongest signals across modes. Consider adding more NLI-heavy training pairs.
4. **Entity value features**: high sparsity means top-5 aggregation may still give 1.0 for both-empty pairs. Consider a presence/absence flag as additional feature.
5. **Data augmentation**: failure cases with numeric values (dates, percentages) that the model gets confidently wrong suggest adding adversarial numeric-swap pairs to all splits.
