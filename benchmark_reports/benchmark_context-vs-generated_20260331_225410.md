# Benchmark Report — context-vs-generated
Generated: 2026-03-31T22:50:38.620036 | Test pairs: 216

## Metric Comparison

| Metric | SilverBullet | NLI-DeBERTa-v3-base | Delta | STS-RoBERTa-base | Delta |
|--------|-------------|--------|-------|--------|-------|
| Accuracy | 0.8981 | 0.7407 | +0.1574 | 0.4583 | +0.4398 |
| ROC-AUC | 0.9532 | 0.8279 | +0.1254 | 0.8518 | +0.1014 |
| AUPRC | 0.9429 | 0.8162 | +0.1267 | 0.8416 | +0.1013 |
| MCC | 0.8017 | 0.4904 | +0.3113 | 0.0000 | +0.8017 |
| F1 (@0.5) | 0.8952 | 0.6585 | +0.2367 | 0.6286 | +0.2667 |
| Brier Score | 0.0844 | 0.2479 | +0.1634 | 0.2484 | +0.1640 |

## Inference Latency (ms / pair, warm cache)

| Model | ms/pair |
|-------|--------|
| SilverBullet (precomputed features) | 0.56 |
| NLI-DeBERTa-v3-base | 105.45 |
| STS-RoBERTa-base | 73.31 |

## Failure Case Analysis

### vs NLI-DeBERTa-v3-base

**sb wrong NLI-DeBERTa-v3-base right** (13 pairs)

- true=0  SB=1 (0.615)  RR=0 (0.001)
  - *text1:* DNA is a double-helix molecule with adenine pairing with thymine and guanine pairing with cytosine.
  - *text2:* DNA is a double-helix molecule where adenine pairs with cytosine and guanine pairs with thymine — held together by hydro

- true=0  SB=1 (0.862)  RR=0 (0.011)
  - *text1:* Worst of all, the Kamakura warriors, resenting the way the Kyoto court referred to them as  Eastern barbarians,  sought 
  - *text2:* Sick of being referred to as barbarians, the Kamakura warriors wanted to improve their imagine by doing things to appear

- true=0  SB=1 (0.613)  RR=0 (0.000)
  - *text1:* The Amazon River carries roughly 20% of the world's fresh water discharge.
  - *text2:* The Amazon River carries roughly 20% of the world's fresh water discharge and is the longest river in the world, stretch


**sb right NLI-DeBERTa-v3-base wrong** (47 pairs)

- true=1  SB=1 (0.649)  RR=0 (0.001)
  - *text1:* Cross-entropy loss measures the difference between a predicted probability distribution and the true distribution, commo
  - *text2:* For classification, cross-entropy loss quantifies how far the model's predicted probabilities diverge from the actual on

- true=1  SB=1 (0.846)  RR=0 (0.000)
  - *text1:* The Best: Make the Music Go Bang! The album includes liner notes by Tony Alva, K. K. Barrett, Elissa Bello, Tito Larriva
  - *text2:* vertical skateboarding

- true=1  SB=1 (0.600)  RR=0 (0.005)
  - *text1:* Syria Regime Agrees to Attend Peace Conference
  - *text2:* Russia: Syria Agrees to Participate in Conference


**both wrong** (9 pairs)

- true=1  SB=0 (0.206)  RR=0 (0.001)
  - *text1:* Compound interest is interest calculated on both the principal and the accumulated interest from previous periods.
  - *text2:* Unlike simple interest, compound interest grows on the original principal plus any interest already earned, leading to e

- true=0  SB=1 (0.680)  RR=1 (0.750)
  - *text1:* Children’s National Medical Center (formerly DC Children’s Hospital) is ranked among the top 10 children’s hospitals in 
  - *text2:* Children's National Medical Center is a top-ranked hospital.

- true=1  SB=0 (0.467)  RR=0 (0.000)
  - *text1:* the 5 permanent united nations security council members are britain, china, france, russia and the united states.
  - *text2:* p5+1 members are britain, china, france, russia, germany and the united states.


### vs STS-RoBERTa-base

**sb wrong STS-RoBERTa-base right** (5 pairs)

- true=1  SB=0 (0.206)  RR=1 (0.637)
  - *text1:* Compound interest is interest calculated on both the principal and the accumulated interest from previous periods.
  - *text2:* Unlike simple interest, compound interest grows on the original principal plus any interest already earned, leading to e

- true=1  SB=0 (0.467)  RR=1 (0.679)
  - *text1:* the 5 permanent united nations security council members are britain, china, france, russia and the united states.
  - *text2:* p5+1 members are britain, china, france, russia, germany and the united states.

- true=1  SB=0 (0.264)  RR=1 (0.698)
  - *text1:* The judge also refused to postpone the trial date of Sept. 29.
  - *text2:* Obus also denied a defense motion to postpone the Sept. 29 trial date.


**sb right STS-RoBERTa-base wrong** (100 pairs)

- true=0  SB=0 (0.046)  RR=1 (0.620)
  - *text1:* Isaac Newton developed the laws of motion and universal gravitation in the 17th century.
  - *text2:* Albert Einstein developed the laws of motion and universal gravitation in the early 20th century, replacing Newton's ear

- true=0  SB=0 (0.034)  RR=1 (0.569)
  - *text1:* A man is playing on his keyboard.
  - *text2:* A man is playing a guitar.

- true=0  SB=0 (0.054)  RR=1 (0.632)
  - *text1:* Newton's first law of motion states that an object at rest stays at rest, and an object in motion continues in motion at
  - *text2:* Newton's second law of motion states that force equals mass times acceleration (F = ma), and an object at rest stays at 


**both wrong** (17 pairs)

- true=0  SB=1 (0.615)  RR=1 (0.676)
  - *text1:* DNA is a double-helix molecule with adenine pairing with thymine and guanine pairing with cytosine.
  - *text2:* DNA is a double-helix molecule where adenine pairs with cytosine and guanine pairs with thymine — held together by hydro

- true=0  SB=1 (0.862)  RR=1 (0.676)
  - *text1:* Worst of all, the Kamakura warriors, resenting the way the Kyoto court referred to them as  Eastern barbarians,  sought 
  - *text2:* Sick of being referred to as barbarians, the Kamakura warriors wanted to improve their imagine by doing things to appear

- true=0  SB=1 (0.680)  RR=1 (0.673)
  - *text1:* Children’s National Medical Center (formerly DC Children’s Hospital) is ranked among the top 10 children’s hospitals in 
  - *text2:* Children's National Medical Center is a top-ranked hospital.

