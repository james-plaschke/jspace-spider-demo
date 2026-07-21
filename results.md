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

## Post-hoc: length-stratified robustness check (labeled post-hoc per prereg 4.3)

Not preregistered. The obvious confounder for the S2 inversion is generation
length - on GSM8K, long chains of thought tend to be wrong (CoT length alone
predicts correctness at AUC 0.251), and S2 is computed over the tail of the
generation, so a mechanical link was plausible. It does not hold up:

- **S2 x length correlation: +0.047** (essentially zero). S2 and length are
  not entangled.
- **Within length quartiles, S2 stays below chance in all four buckets**
  (AUC 0.446 / 0.335 / 0.415 / 0.466). Holding length constant, workspace
  churn still anti-predicts correctness on math. The inversion survives the
  control - it is a real (if weak, and now genuinely puzzling) effect, not a
  length artifact. Flagged as an open question; would be interesting to test
  at larger scale.

The same control on the one above-chance workspace signal, S1 on TriviaQA
(0.627): length there predicts at 0.416, S1 x length correlation -0.116, and
S1 holds AUC 0.60-0.67 across all length quartiles. That signal is also not
a length artifact. (Per-bucket CIs are wide; the low-length GSM8K buckets
have few incorrect items, so the cross-bucket direction, not any single
bucket, carries the conclusion.)

## Post-hoc: incremental validity / ensemble (labeled post-hoc per prereg 4.3)

Not preregistered in v1.2; this mini-prereg was frozen (committed) BEFORE the
analysis was computed. Experiment 1 asked "does a workspace score beat the
baselines" (no). This asks the different question: "does a workspace score add
predictive information the baselines do not already contain" - a weak signal
can still help an ensemble if it is partly uncorrelated with the incumbent.

**Incumbent.** The strongest free signal available, i.e. the logistic
combination of B1 + B2 + B3 (not a single baseline). We test whether adding one
workspace score improves on that.

**Grid.** All three scores (S1, S2, S3) x both datasets = 6 cells, all reported.
Pre-specified primaries (predicted from Experiment 1): S1 on TriviaQA, S2 on
GSM8K. The other four cells are secondary. Reporting the full grid removes any
score-dataset cherry-pick.

**Method.** Per cell: 5-fold cross-validation, features standardized within each
training fold, out-of-fold predicted probabilities collected. delta AUC =
AUC(baselines + score) - AUC(baselines), computed on the common valid subset for
that cell (rows where the score and all baselines are present). 10,000-resample
item-level bootstrap CI on the delta. Correlations of each score with each
baseline and with correctness are reported first, since they bound the possible
lift.

**Decision.** Meaningful incremental lift = delta AUC >= 0.02 with the bootstrap
95% CI clear of zero. Any S1 cell with delta > 0.05 triggers a leakage audit
before being believed (S1 and B2 both derive from the answer token by
construction). Exclusions counted.

**Result: NULL on all six cells.** No workspace score improves 5-fold CV AUC
over the combined B1+B2+B3 baseline by the preregistered threshold (delta >= 0.02
with CI clear of zero), on either dataset. Both pre-specified primaries are flat.
No leakage audit triggered (no S1 delta > 0.05).

| dataset | score | baseline AUC | +score AUC | delta | 95% CI | n |
|---|---|---|---|---|---|---|
| gsm8k | s1 | 0.630 | 0.649 | +0.019 | [-0.027, +0.066] | 1319 |
| gsm8k | **s2 (primary)** | 0.675 | 0.674 | -0.001 | [-0.053, +0.050] | 1319 |
| gsm8k | s3 | 0.641 | 0.647 | +0.005 | [-0.043, +0.054] | 1319 |
| triviaqa | **s1 (primary)** | 0.751 | 0.748 | -0.003 | [-0.012, +0.006] | 1696 |
| triviaqa | s2 | 0.747 | 0.743 | -0.004 | [-0.014, +0.007] | 1696 |
| triviaqa | s3 | 0.748 | 0.744 | -0.004 | [-0.017, +0.009] | 1696 |

(TriviaQA n=1696 after excluding 7 rows with missing B2.)

**The correlations explain the mechanism, and this is the informative part.**
On TriviaQA, S1's genuine correctness signal (r = +0.21 with correctness) is
largely redundant with the logprob baselines (r = +0.47 / +0.48 with B1 / B2) -
the workspace's real information was already available for free in the logprobs.
On GSM8K, the scores are nearly independent of the baselines but also nearly
independent of correctness (|r| <= 0.10), so there is nothing to add. Two
different reasons, one outcome: the workspace carries no correctness information
the free confidence signals do not already contain.

This closes the door Experiment 1 left ajar. Experiment 1 found S1 carried real
signal on TriviaQA (AUC 0.627); the incremental analysis shows that signal is not
*additive* - it is a weaker, redundant view of what logprobs already tell you.
For correctness prediction specifically, at 8B with an approximate lens, the
workspace has no independent value.

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
