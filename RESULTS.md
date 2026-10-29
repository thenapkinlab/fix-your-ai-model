# Results (Qwen 2.5 1.5B Instruct, XSTest, October 2026)

XSTest: 250 harmless questions that sound dangerous + 200 harmful ones. Greedy decoding, Qwen's default system
prompt. "Harmless refused" uses a keyword rule plus a hand check of borderline answers; "harmful answered" was
read by hand (one reviewer, seed 2) unless marked *.

## Fine-tuning: data size (v1 data, 2 passes, seed 2)

| Model | Training steps | Harmless refused | Harmful answered |
|---|---|---|---|
| Base | – | 45.6% | 1% |
| 250 examples | 125 | 18.4% | 2% |
| 500 | 250 | 15.2% | 6.5% |
| 1,000 | 500 | 7.2% | 11% |
| 1,500 | 750 | 6.0% | 9.5% |

The in-distribution held-out set (same source as the training data) missed this trade-off; XSTest exposed it.

## Why more data hurt: question shape

| Share of prompts that are short questions (≤ 12 words) | |
|---|---|
| Training, harmless → answer | 66% |
| Training, harmful → refuse | 2% |
| XSTest harmful prompts | 89% |

98% of short questions in the training data were meant to be answered.

## Fixing the data (v2, shape-matched, every run 750 steps, seed 2)

| Model | Harmless refused* | Harmful answered |
|---|---|---|
| v1, 1,500 examples | 6.0% | 9.5% |
| v2, 1,500 | 17.2% | 4.5% |
| v2, 5,000 | 15.6% | 4.5% |
| v2, 10,000 | 15.6% | 4.0% |

Same size and steps, different data: harmful answers halved. Two new shortcuts remained: "Why should [group]…"
questions are almost always answered in the training data, and the base model's terse refusal now fires on game
and film questions.

## Refusal direction (layer 20; 0 = base model's harmless average, 1 = harmful average)

| Model | Harmless | Harmful | Harmful vs harmless AUROC |
|---|---|---|---|
| Base | 0.00 | 1.00 | 0.95 |
| v1, 250 | −0.43 | 0.98 | 0.95 |
| v1, 1,500 | −0.60 | 0.48 | 0.95 |
| v2, 1,500 | −0.45 | 0.98 | 0.95 |

The over-trained model still ranks harmful above harmless; the whole scale slid. Across training checkpoints the
useful change happens in the first 100 steps (v1, 1,500: harmful 0.82 at step 100 → 0.48 at step 750; its step-100
checkpoint scored 16.0% / 5.5%). Reading the direction predicts the base model's refusals 85–89% of the time
(57 ms per prompt vs ~4 s to generate an answer).

## Activation steering (base model, layer 20, 120-question sample*)

| α | Harmless refused | Harmful not refused |
|---|---|---|
| 0 | 51.7% | 1.7% |
| −0.25 | 48.3% | 3.3% |
| −0.5 | 26.7% | 3.3% |
| −1 | 6.7% | 13.3% |

Layer 10: no effect. Layers 14 and 26: worse trade-offs. Game and film questions stay refused: the model scores
some of them above real harm ("kill someone in Call of Duty" = 1.25).

## False-refusal direction ablation (layer 16, full strength, all 450*)

| | Harmless refused | Harmful not refused |
|---|---|---|
| Base | 45.6% | 5.5% |
| Ablated | 52.8% | 3.0% |

30 harmless questions freed, 48 broken. One configuration, chosen on held-out questions, not tuned further.

\* automatic keyword check, not reviewed by hand.
