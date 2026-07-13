# Results: Does J-space workspace routing predict answer correctness?

**Outcome under the preregistered decision rules: NEGATIVE.** The workspace
scores did not beat the logprob baselines on either dataset. Per the
[preregistration](preregistration-v1.2.md) (v1.2, frozen 2026-07-08, section
4.4), this is reported as a negative result. Completed 2026-07-13.

Model: Llama-3.1-8B-Instruct, chain-of-thought chat prompts, greedy decoding.
Lens: Jacobian lens fitted on 100 wikitext prompts (approximate "cheap path").
Decomposition: batched nonnegative gradient pursuit, k=25, validated at 99.7%
synthetic support recovery on the real 128,256-atom dictionary.

## AUC at predicting answer correctness (95% bootstrap CI, 10,000 resamples)

### GSM8K (test, n=1319, model accuracy 86.6%)

| score | AUC | 95% CI | shuffle ctrl |
|---|---|---|---|
| S1 routing (produced anchor) | 0.526 | [0.478, 0.573] | 0.495 |
| S1-gold (secondary) | 0.526 | [0.478, 0.573] | 0.501 |
| S2 stability | 0.427 | [0.380, 0.475] | 0.501 |
| S3 coverage | 0.467 | [0.421, 0.515] | 0.497 |
| B1 mean answer logprob | 0.711 | [0.672, 0.750] | 0.497 |
| **B2 first-token logprob** | **0.759** | [0.719, 0.797] | 0.498 |
| B3 verbalized confidence | 0.632 | [0.598, 0.667] | 0.503 |

### TriviaQA (rc.nocontext validation sample, n=1703, model accuracy 75.1%)

| score | AUC | 95% CI | shuffle ctrl |
|---|---|---|---|
| S1 routing (produced anchor) | 0.627 | [0.596, 0.658] | 0.501 |
| S1-gold (secondary) | 0.610 | [0.581, 0.637] | 0.503 |
| S2 stability | 0.495 | [0.464, 0.527] | 0.501 |
| S3 coverage | 0.563 | [0.530, 0.594] | 0.499 |
| B1 mean answer logprob | 0.649 | [0.615, 0.682] | 0.499 |
| B2 first-token logprob | 0.710 | [0.681, 0.738] | 0.500 |
| **B3 verbalized confidence** | **0.722** | [0.695, 0.749] | 0.498 |

## Reading the table

- **The headline:** no workspace score approaches the best baseline on either
  dataset. The preregistered SIGNAL threshold (+0.05 AUC over best baseline,
  non-overlapping CIs, both datasets) is not remotely met; on GSM8K the gap
  runs the other way by ~0.23.
- **Workspace routing is not empty, just uncompetitive:** S1 on TriviaQA
  (0.627, CI excluding chance) shows that the answer's concept being visibly
  loaded in the workspace before emission does carry real information about
  correctness - the model's own probability/confidence signals simply carry
  more.
- **One unexpected inversion:** S2 (workspace stability) on GSM8K is
  significantly BELOW chance (0.427, CI [0.380, 0.475]) - i.e., workspace
  CHURN weakly predicts being right on math. Post-hoc speculation, labeled as
  such: correct multi-step solutions may march through different concepts
  step by step, while wrong ones perseverate. Not preregistered; treat as a
  hypothesis for someone else to test.
- **Controls:** all shuffled-label AUCs sit at 0.495-0.503, as they should.
  Baselines landed in their literature-typical range, so the incumbent was
  healthy when it won.

## Flags and exclusions (reported per prereg)

- TriviaQA exclusion rate 14.9% (297/2000 items never produced the required
  "Answer:" marker). Above the prereg's 5% flag threshold - reported
  prominently; TriviaQA was pre-designated the secondary dataset.
- GSM8K used an lm-eval-style flexible-extract fallback (last number in the
  generation) when the "####" marker was absent; each item records which
  extraction path applied. GSM8K accuracy (86.6%) came in slightly above the
  prereg's 55-85% sanity band, leaving 177 incorrect items for
  discrimination.
- B4 (self-consistency baseline) was budget-optional and not run.

## Limitations

Approximate 100-prompt lens fit (not the paper's exact 1000-prompt Jacobian);
single-vocabulary-token anchors (the method's published limitation); one
model (Llama-3.1-8B-Instruct); one lens fit; 8B scale, where the only
independent replication of the workspace paper (Nanda, Qwen 27B) already
found weaker effects than the original. Any of these could suppress a real
signal. The claim supported here is narrow: *under these conditions,
workspace readouts did not beat logprob baselines* - not "deliberation
scoring cannot work."

## Data

Summary statistics: [w3_results.json](w3_results.json). Per-item scores
(all 3,022 graded items, every score and baseline):
[w3_rows.jsonl](w3_rows.jsonl). The raw generation outputs and activations
(24GB) are retained offline and available on request.
