# Benchmark Report — reference-vs-generated
Generated: 2026-03-31T22:54:11.674853 | Test pairs: 216

## Metric Comparison

| Metric | SilverBullet | NLI-DeBERTa-v3-base | Delta | STS-RoBERTa-base | Delta |
|--------|-------------|--------|-------|--------|-------|
| Accuracy | 0.8056 | 0.7361 | +0.0694 | 0.4583 | +0.3472 |
| ROC-AUC | 0.8705 | 0.7937 | +0.0768 | 0.8397 | +0.0308 |
| AUPRC | 0.8291 | 0.8049 | +0.0243 | 0.8395 | -0.0103 |
| MCC | 0.6112 | 0.5140 | +0.0971 | 0.0000 | +0.6112 |
| F1 (@0.5) | 0.7941 | 0.6122 | +0.1819 | 0.6286 | +0.1655 |
| Brier Score | 0.1439 | 0.2625 | +0.1186 | 0.2407 | +0.0967 |

## Inference Latency (ms / pair, warm cache)

| Model | ms/pair |
|-------|--------|
| SilverBullet (precomputed features) | 0.44 |
| NLI-DeBERTa-v3-base | 63.20 |
| STS-RoBERTa-base | 46.48 |

## Failure Case Analysis

### vs NLI-DeBERTa-v3-base

**sb wrong NLI-DeBERTa-v3-base right** (24 pairs)

- true=0  SB=1 (0.772)  RR=0 (0.001)
  - *text1:* DNA is a double-helix molecule with adenine pairing with thymine and guanine pairing with cytosine.
  - *text2:* DNA is a double-helix molecule where adenine pairs with cytosine and guanine pairs with thymine — held together by hydro

- true=0  SB=1 (0.979)  RR=0 (0.011)
  - *text1:* Worst of all, the Kamakura warriors, resenting the way the Kyoto court referred to them as  Eastern barbarians,  sought 
  - *text2:* Sick of being referred to as barbarians, the Kamakura warriors wanted to improve their imagine by doing things to appear

- true=0  SB=1 (0.816)  RR=0 (0.000)
  - *text1:* The Amazon River carries roughly 20% of the world's fresh water discharge.
  - *text2:* The Amazon River carries roughly 20% of the world's fresh water discharge and is the longest river in the world, stretch


**sb right NLI-DeBERTa-v3-base wrong** (39 pairs)

- true=1  SB=1 (0.704)  RR=0 (0.001)
  - *text1:* Cross-entropy loss measures the difference between a predicted probability distribution and the true distribution, commo
  - *text2:* For classification, cross-entropy loss quantifies how far the model's predicted probabilities diverge from the actual on

- true=1  SB=1 (0.894)  RR=0 (0.000)
  - *text1:* How many feet tall was the proposed statue of Schwarzenegger?
  - *text2:* In tribute to Schwarzenegger in 2002, Forum Stadtpark, a local cultural association, proposed plans to build a 25-meter 

- true=1  SB=1 (0.743)  RR=0 (0.001)
  - *text1:* What is the only object identified with Neptune's trailing L5 Lagrangian point?
  - *text2:* The first and so far only object identified as associated with Neptune's trailing L5 Lagrangian point is 2008 LC18.


**both wrong** (18 pairs)

- true=1  SB=0 (0.372)  RR=0 (0.001)
  - *text1:* Compound interest is interest calculated on both the principal and the accumulated interest from previous periods.
  - *text2:* Unlike simple interest, compound interest grows on the original principal plus any interest already earned, leading to e

- true=1  SB=0 (0.334)  RR=0 (0.005)
  - *text1:* Syria Regime Agrees to Attend Peace Conference
  - *text2:* Russia: Syria Agrees to Participate in Conference

- true=1  SB=0 (0.421)  RR=0 (0.000)
  - *text1:* Due to his reporting skills, some give Russell credit for doing what?
  - *text2:* Some credit Russell with prompting the resignation of the sitting British government through his reporting of the lacklu


### vs STS-RoBERTa-base

**sb wrong STS-RoBERTa-base right** (18 pairs)

- true=1  SB=0 (0.372)  RR=1 (0.637)
  - *text1:* Compound interest is interest calculated on both the principal and the accumulated interest from previous periods.
  - *text2:* Unlike simple interest, compound interest grows on the original principal plus any interest already earned, leading to e

- true=1  SB=0 (0.334)  RR=1 (0.730)
  - *text1:* Syria Regime Agrees to Attend Peace Conference
  - *text2:* Russia: Syria Agrees to Participate in Conference

- true=1  SB=0 (0.421)  RR=1 (0.627)
  - *text1:* Due to his reporting skills, some give Russell credit for doing what?
  - *text2:* Some credit Russell with prompting the resignation of the sitting British government through his reporting of the lacklu


**sb right STS-RoBERTa-base wrong** (93 pairs)

- true=0  SB=0 (0.030)  RR=1 (0.620)
  - *text1:* Isaac Newton developed the laws of motion and universal gravitation in the 17th century.
  - *text2:* Albert Einstein developed the laws of motion and universal gravitation in the early 20th century, replacing Newton's ear

- true=0  SB=0 (0.011)  RR=1 (0.569)
  - *text1:* A man is playing on his keyboard.
  - *text2:* A man is playing a guitar.

- true=0  SB=0 (0.020)  RR=1 (0.632)
  - *text1:* Newton's first law of motion states that an object at rest stays at rest, and an object in motion continues in motion at
  - *text2:* Newton's second law of motion states that force equals mass times acceleration (F = ma), and an object at rest stays at 


**both wrong** (24 pairs)

- true=0  SB=1 (0.772)  RR=1 (0.676)
  - *text1:* DNA is a double-helix molecule with adenine pairing with thymine and guanine pairing with cytosine.
  - *text2:* DNA is a double-helix molecule where adenine pairs with cytosine and guanine pairs with thymine — held together by hydro

- true=0  SB=1 (0.979)  RR=1 (0.676)
  - *text1:* Worst of all, the Kamakura warriors, resenting the way the Kyoto court referred to them as  Eastern barbarians,  sought 
  - *text2:* Sick of being referred to as barbarians, the Kamakura warriors wanted to improve their imagine by doing things to appear

- true=0  SB=1 (0.816)  RR=1 (0.663)
  - *text1:* The Amazon River carries roughly 20% of the world's fresh water discharge.
  - *text2:* The Amazon River carries roughly 20% of the world's fresh water discharge and is the longest river in the world, stretch

