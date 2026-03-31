# Benchmark Report — model-vs-model
Generated: 2026-03-31T22:54:51.717379 | Test pairs: 216

## Metric Comparison

| Metric | SilverBullet | NLI-DeBERTa-v3-base | Delta | STS-RoBERTa-base | Delta |
|--------|-------------|--------|-------|--------|-------|
| Accuracy | 0.8241 | 0.7824 | +0.0417 | 0.4583 | +0.3657 |
| ROC-AUC | 0.8923 | 0.8747 | +0.0175 | 0.9111 | -0.0188 |
| AUPRC | 0.8542 | 0.8677 | -0.0136 | 0.8924 | -0.0382 |
| MCC | 0.6597 | 0.5716 | +0.0881 | 0.0000 | +0.6597 |
| F1 (@0.5) | 0.8241 | 0.7251 | +0.0989 | 0.6286 | +0.1955 |
| Brier Score | 0.1288 | 0.2093 | +0.0805 | 0.2414 | +0.1126 |

## Inference Latency (ms / pair, warm cache)

| Model | ms/pair |
|-------|--------|
| SilverBullet (precomputed features) | 0.50 |
| NLI-DeBERTa-v3-base | 57.84 |
| STS-RoBERTa-base | 43.12 |

## Failure Case Analysis

### vs NLI-DeBERTa-v3-base

**sb wrong NLI-DeBERTa-v3-base right** (20 pairs)

- true=0  SB=1 (0.544)  RR=0 (0.450)
  - *text1:* A man and women sitting on a sofa holding a small baby.
  - *text2:* a woman sitting on a sofa holding a baby.

- true=0  SB=1 (0.802)  RR=0 (0.001)
  - *text1:* DNA is a double-helix molecule with adenine pairing with thymine and guanine pairing with cytosine.
  - *text2:* DNA is a double-helix molecule where adenine pairs with cytosine and guanine pairs with thymine — held together by hydro

- true=0  SB=1 (0.760)  RR=0 (0.011)
  - *text1:* Worst of all, the Kamakura warriors, resenting the way the Kyoto court referred to them as  Eastern barbarians,  sought 
  - *text2:* Sick of being referred to as barbarians, the Kamakura warriors wanted to improve their imagine by doing things to appear


**sb right NLI-DeBERTa-v3-base wrong** (29 pairs)

- true=1  SB=1 (0.629)  RR=0 (0.001)
  - *text1:* Cross-entropy loss measures the difference between a predicted probability distribution and the true distribution, commo
  - *text2:* For classification, cross-entropy loss quantifies how far the model's predicted probabilities diverge from the actual on

- true=1  SB=1 (0.785)  RR=0 (0.000)
  - *text1:* A red and gray train is going through a tunnel.
  - *text2:* A red and gray train travels through a tunnel amongst the trees.

- true=1  SB=1 (0.537)  RR=0 (0.038)
  - *text1:* What are the opinions of people around the world on California's affirmative laws for sex?
  - *text2:* What do people think about California's new affirmative consent law?


**both wrong** (18 pairs)

- true=1  SB=0 (0.451)  RR=0 (0.001)
  - *text1:* Compound interest is interest calculated on both the principal and the accumulated interest from previous periods.
  - *text2:* Unlike simple interest, compound interest grows on the original principal plus any interest already earned, leading to e

- true=1  SB=0 (0.228)  RR=0 (0.005)
  - *text1:* Syria Regime Agrees to Attend Peace Conference
  - *text2:* Russia: Syria Agrees to Participate in Conference

- true=0  SB=1 (0.839)  RR=1 (0.995)
  - *text1:* How do you convert HTML to JSP? What are some tips?
  - *text2:* How can I change html pages to jsp?


### vs STS-RoBERTa-base

**sb wrong STS-RoBERTa-base right** (10 pairs)

- true=1  SB=0 (0.451)  RR=1 (0.637)
  - *text1:* Compound interest is interest calculated on both the principal and the accumulated interest from previous periods.
  - *text2:* Unlike simple interest, compound interest grows on the original principal plus any interest already earned, leading to e

- true=1  SB=0 (0.228)  RR=1 (0.730)
  - *text1:* Syria Regime Agrees to Attend Peace Conference
  - *text2:* Russia: Syria Agrees to Participate in Conference

- true=1  SB=0 (0.499)  RR=1 (0.672)
  - *text1:* The SOLID principles are five object-oriented design guidelines intended to make software more maintainable, flexible, a
  - *text2:* SOLID is an acronym for five OOP design principles — Single responsibility, Open/closed, Liskov substitution, Interface 


**sb right STS-RoBERTa-base wrong** (89 pairs)

- true=0  SB=0 (0.027)  RR=1 (0.620)
  - *text1:* Isaac Newton developed the laws of motion and universal gravitation in the 17th century.
  - *text2:* Albert Einstein developed the laws of motion and universal gravitation in the early 20th century, replacing Newton's ear

- true=0  SB=0 (0.020)  RR=1 (0.569)
  - *text1:* A man is playing on his keyboard.
  - *text2:* A man is playing a guitar.

- true=0  SB=0 (0.058)  RR=1 (0.632)
  - *text1:* Newton's first law of motion states that an object at rest stays at rest, and an object in motion continues in motion at
  - *text2:* Newton's second law of motion states that force equals mass times acceleration (F = ma), and an object at rest stays at 


**both wrong** (28 pairs)

- true=0  SB=1 (0.544)  RR=1 (0.650)
  - *text1:* A man and women sitting on a sofa holding a small baby.
  - *text2:* a woman sitting on a sofa holding a baby.

- true=0  SB=1 (0.802)  RR=1 (0.676)
  - *text1:* DNA is a double-helix molecule with adenine pairing with thymine and guanine pairing with cytosine.
  - *text2:* DNA is a double-helix molecule where adenine pairs with cytosine and guanine pairs with thymine — held together by hydro

- true=0  SB=1 (0.760)  RR=1 (0.676)
  - *text1:* Worst of all, the Kamakura warriors, resenting the way the Kyoto court referred to them as  Eastern barbarians,  sought 
  - *text2:* Sick of being referred to as barbarians, the Kamakura warriors wanted to improve their imagine by doing things to appear

