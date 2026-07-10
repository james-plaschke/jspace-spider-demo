# J-Space Deliberation Experiment - Full Handoff Spec

Version 1.2 - 2026-07-08
Author: James Plaschke (with Claude)
Status: Not started. This document is the complete handoff package.
Changelog v1.2 (residual ambiguity fixes, design now FROZEN): S2 layer
pinned to layer 20 (the "middle layer of the mid-band" was layer 19, which
the every-other-layer downsampling would skip); S3 layer pinned to layer 20
(same as S2); S3 stopword list pinned to NLTK English; S2 short-CoT edge
case rule added (fewer than 3 pre-answer positions = missing, counted with
exclusions); SIGNAL decision rule clarified to use headline S1 only
(S1-gold explicitly does not count).
Changelog v1.1 (after peer review by a second Claude instance): S1 headline
variant fixed to produced-answer anchor (gold anchor demoted to secondary -
circularity risk); S2/S3 formulas fully specified; answer-span location
rules added; mid-band layers pinned; B1 named the real incumbent baseline;
length-stratified robustness added as labeled post-hoc; instruction added
to implement from primary sources.

---

## 0. How to use this document (instructions for the assistant receiving it)

You are picking up a fully-scoped ML experiment. The human operator (James) is
smart and technical-adjacent (sales engineer at Ketryx, a regulated-software
ALM company) but has NEVER done an ML experiment: he does not know PyTorch,
has never rented a GPU, and learned what "logprob" means last week. Your job
is to write all code, drive all technical decisions, and explain as you go.
His job is to run commands, paste outputs back to you, pay for the GPU, and
make judgment calls when you surface them.

Working preferences:
- Plain ASCII punctuation in all output (no em-dashes, curly quotes, arrows,
  or ellipsis characters - his clipboard mojibakes multi-byte unicode).
- Explain jargon the first time you use it. There is a glossary in section 9.
- Show him the full terminal output of things you run.
- He likes analogies and understanding WHY, not just what to type.
- Prove things work before declaring them done.

Read this whole file before doing anything. Then start at Weekend 1
(section 5). Do not re-litigate the design unless you find a real flaw;
the scoring definitions in section 4 are preregistered on purpose.

IMPORTANT: the paper (July 2026) and the jacobian-lens repo post-date most
models' training data. Your first move in Weekend 1 is to fetch the repo
README and the paper's methods section and implement from those primary
sources. This document is the map, not the territory.

---

## 1. The experiment in one paragraph (plain language)

Anthropic proved you can watch an AI model "think": a small set of named
concepts (its "workspace") lights up inside the model when it deliberates,
and stays dark when it answers on reflex. We download a model whose insides
we are allowed to look at (Meta's Llama), point the published lens at it,
ask it a few thousand questions with known answers, and test one thing:
does "the model actually deliberated" predict "the answer is correct"
better than the model's own confidence number (the current industry
standard)? The output is one number (AUC). Nobody has published this
measurement. Any result - positive, null, or negative - is worth writing up.

## 2. Why this matters (context, 3 minutes)

- The science: Anthropic's paper "Verbalizable Representations Form a Global
  Workspace in Language Models" (Gurnee et al., Transformer Circuits,
  July 6, 2026) showed: (a) models do higher-order reasoning through a
  sparse workspace of ~20-25 named concepts (~6-10% of activation variance);
  (b) suppressing the workspace collapses multi-hop reasoning to near zero
  while parsing/recall survive; (c) automatic processing (fluent reflexes)
  bypasses the workspace entirely; (d) swapping one workspace concept
  (spider -> ant) flips downstream answers (8 legs -> 6 legs).
- The gap: nobody has tested whether workspace routing PREDICTS answer
  correctness on standard benchmarks. That predictive question is the
  foundation of several potential products (AI trust scoring, deliberation-
  gated model routing, audit evidence). James ran a week of adversarial
  market research on those products (verdicts: all "not now" - science too
  new, market access constraints); this experiment is the cheap option that
  stays valuable in every future branch.
- Expectation calibration: Neel Nanda partially replicated the paper on
  Qwen 3.6 27B with WEAKER, NOISIER results (some sub-experiments failed).
  At 8B scale expect a modest effect. A null result is publishable and
  useful; do not torture the data to avoid one.

## 3. The one-sentence hypothesis

H1: A deliberation score derived from J-space readouts (did the answer's
concepts route through the workspace before emission) achieves higher AUC
at predicting answer correctness than mean-token-logprob baselines, on
GSM8K and TriviaQA, for Llama-3.1-8B-Instruct.

## 4. Preregistered design (FROZEN - do not adjust after seeing correlations)

### 4.1 Scores (computed per question-answer pair)

All computed from J-space decompositions of mid-band layers (30%-90% of
depth), at token positions BEFORE the final answer span is emitted.

- S1 ROUTING (HEADLINE variant = produced-answer anchor): max workspace
  coefficient, across the mid-band and across all positions PRIOR to the
  answer span, of the first token of the MODEL'S PRODUCED answer (for
  GSM8K numeric answers: max over the digit tokens of the produced
  number). This is the deployable variant - at inference time you never
  have the gold answer.
  S1-gold (SECONDARY, mechanistic only): same computation anchored on the
  gold answer's first token. Known issue, declared now: this variant is
  partially circular (if the gold concept is loaded pre-emission the model
  likely emits it, so it partly measures "knew the answer"), so it may
  post a flattering AUC for the wrong reason. It is reported for
  mechanistic interest and is NOT the headline number.
- S2 STABILITY: for each generated position in the final 20 tokens before
  the answer span, take the top-10 concepts by coefficient (from the k=25
  decomposition) at layer 20 (pinned; the naive "middle of the mid-band"
  would be layer 19, which every-other-layer capture skips - layer 20 is
  the 6th of the 10 captured layers); compute Jaccard similarity between
  consecutive positions' top-10 sets; S2 = the mean of those Jaccard
  values. Higher = stabler = more committed.
  Edge case, pinned: if fewer than 20 pre-answer generated positions
  exist, use all available; if fewer than 3 positions (i.e. fewer than 2
  consecutive Jaccard pairs), mark S2 as missing for that item and count
  it with the exclusions (same reporting rule as span-location failures).
- S3 COVERAGE: computed at layer 20, same layer as S2 (pinned). Q-set =
  union of top-10 concept sets across the question's content-token
  positions (stopwords excluded via the NLTK English stopword list,
  pinned) while reading the question; G-set = union of top-10 concept
  sets across all pre-answer generated positions. S3 = Jaccard(Q-set,
  G-set).

Mid-band definition, pinned: layers 10 through 28 of Llama-3.1-8B's 32
layers (approx 30-90% depth), captured every other layer (10 layers total)
if storage requires downsampling.

Answer-span location rules, pinned:
- GSM8K: prompt template instructs the model to end with "#### <number>".
  Answer span = tokens after the final "####". 
- TriviaQA: prompt template instructs a final line "Answer: <answer>".
  Answer span = first line after the final "Answer:".
- Items where the parser cannot locate the span are EXCLUDED AND COUNTED.
  Report the exclusion rate per dataset; if it exceeds 5%, flag it
  prominently - systematic span-detection failure would bias scores.

Report S1, S1-gold, S2, S3 SEPARATELY (no blended score in v1 - cleaner
science and tells us which signal matters).

### 4.2 Baselines (the incumbents to beat)

- B1: mean token logprob over the generated answer. THIS IS THE REAL
  INCUMBENT - the comparison that matters.
- B2: logprob of the answer's first token.
- B3: verbalized confidence (append "How confident are you, 0-100?" as a
  second pass; parse the number). Known-weak baseline: 8B verbalized
  confidence is notoriously miscalibrated; beating B3 alone means little.
- B4 (subset only, 500 items, if budget allows): self-consistency - sample
  k=5 answers at temperature 0.7, score = agreement fraction.

### 4.3 Metrics and controls

- Primary metric: AUC (area under ROC) of each score at classifying
  correct vs incorrect answers, per dataset. Bootstrap 95% CIs (10,000
  resamples).
- Control 1: shuffled-labels AUC must be ~0.50 (pipeline leak check).
- Control 2: report accuracy of the model itself per dataset (sanity:
  GSM8K 8B CoT should land roughly 55-85% depending on prompt; if it is
  near 0% or 100% the dataset is unusable for discrimination).
- Permitted post-hoc analysis (labeled as such, NOT part of the prereg):
  if results are ambiguous, a length-stratified robustness check (do the
  scores and baselines both just track answer length/difficulty?) may be
  run and must be reported as post-hoc.

### 4.4 Decision rules (set before data)

- SIGNAL: any of S1-S3 beats best baseline by >= 0.05 AUC with
  non-overlapping bootstrap CIs, on BOTH datasets -> write up as positive.
  Here "S1" means the HEADLINE produced-answer-anchor variant only;
  S1-gold does NOT count toward the SIGNAL rule (it is partially circular
  by construction and is reported for mechanistic interest only).
- NULL: within +/- 0.05 of baselines -> write up as informative null
  (first public measurement; still valuable).
- NEGATIVE: below baselines -> write up as negative, done.
- In ALL cases the deliverable is a written report. The writeup is the
  point, not the number.

## 5. The build plan (three weekends)

### Weekend 1 - "Can I see the workspace?" (~$15 GPU)

Goal: the lens works on Llama and shows an unspoken concept.

1. Accounts (operator tasks, ~30 min):
   - huggingface.co account. Request access to meta-llama/Llama-3.1-8B-Instruct
     (it is LICENSE-GATED: click through Meta's license on the model page;
     approval is usually minutes-to-hours). Create an HF access token.
   - runpod.io account, add ~$25 credit.
2. Rent: RunPod Secure Cloud, 1x H100 80GB (or A100 80GB to save money),
   PyTorch template, 200GB volume. Cost ~$2-3/hr. ALWAYS STOP THE POD when
   done for the day (assistant: remind him).
3. Setup: clone https://github.com/anthropics/jacobian-lens , install,
   download the model (~16GB).
4. Verify toolchain: run the repo's example / walkthrough.ipynb on its
   default model untouched. Confirm readouts render.
5. Port to Llama-3.1-8B-Instruct. Repo states other HF decoders adapt
   cleanly. Fit the lens with the cheap path first: repo guidance says
   ~100 prompts is usable (paper spec is 1000 x 128 tokens; scale up only
   if readouts look noisy). Fitting cost: single-digit GPU-hours.
   NOTE the fork: the paper defines an exact autodiff Jacobian (expensive,
   ~30 GPU-hrs at 8B); the repo/community path uses cheaper fitting. Use
   the cheap path; state it as a limitation in the writeup.
6. GO/NO-GO ACCEPTANCE TEST: prompt "The number of legs on the animal that
   spins webs is" and inspect mid-layer readouts (~layers 10-22 of 32).
   PASS = "spider" (or close variants) visibly ranked in the lens readout
   at positions before the answer. Screenshot it (this image is also
   LinkedIn gold). If PASS, weekend 1 is done. If FAIL after honest
   debugging, try 2-3 other two-hop prompts from the paper's style before
   concluding; if still failing, stop and reassess (do not proceed to
   weekend 2 on a broken lens).

### Weekend 2 - "Turn readouts into a score." (~$20-40 GPU)

Goal: working sparse decomposition + frozen score implementations.

1. Implement gradient pursuit sparse decomposition (NOT in the released
   repo - build from the paper's description): given activation h and the
   J-lens vector dictionary, find nonnegative coefficients, at most k=25
   nonzero, minimizing reconstruction error. Iterative greedy selection +
   nonneg least squares on the active set is acceptable. MUST be batched
   torch (naive per-token python loops will blow the budget; there are
   ~1-2M token-layer decompositions in weekend 3).
2. Validate decomposition on synthetic data: compose known sparse
   nonnegative combinations of dictionary vectors + noise; recovery of the
   true support >= ~90% at k=5-10 -> pass.
3. Implement S1/S2/S3 exactly as section 4.1. Unit-test on 20 hand-checked
   examples (5 obviously-deliberated two-hop questions, 5 trivial reflex
   completions) - the scores should differ in the expected direction, but
   DO NOT tune definitions on these; they are smoke tests only.
4. Known limitation to document, not solve: the J-lens dictionary is
   single-vocabulary-token by construction. Multi-token answers (numbers,
   entity names) are anchored by first/digit tokens. This is the method's
   published limitation; note it prominently in the writeup.

### Weekend 3 - "The number." (~$30-60 GPU)

1. Generation run: Llama-3.1-8B-Instruct, chain-of-thought prompting,
   greedy decoding, over GSM8K test (1,319 items) + TriviaQA sample
   (2,000 items, rc.nocontext validation split). Use HuggingFace
   transformers with forward hooks capturing mid-band residual streams.
   (NOT vLLM - it does not expose activations.) Persist activations for
   the final 64 tokens per item + question-reading positions (disk budget:
   plan ~100-200GB; downsample layers to every other one if needed).
2. Grading: EleutherAI lm-evaluation-harness task configs for gsm8k
   (exact match on final number) and triviaqa. Reuse their extraction
   regexes rather than writing your own.
3. Compute S1/S2/S3 + B1-B3 (B4 on a 500-item subset if time), AUCs,
   bootstrap CIs, shuffle control.
4. Write the report: method, the exact/approx-Jacobian caveat, single-token
   caveat, results table, decision-rule outcome, and 3 example items with
   workspace readouts shown (one deliberated-and-right, one reflexed-and-
   wrong, one interesting failure).

Optional Weekend 4 (only if results are null AND curiosity remains):
70B spot-check on 300 items, 4x GPU node, ~$200-500. Nanda's replication
suggests signal grows with scale; a null at 8B + positive trend at 70B is
itself a finding.

## 6. Budget and time summary

| Item | Estimate |
|---|---|
| GPU, happy path | $50-100 total |
| GPU, realistic (debugging 3-5x) | $150-400 |
| Optional 70B check | +$200-500 |
| Calendar | 3 weekends part-time (or ~2 focused weeks) |
| Operator skill required | none beyond terminal copy-paste; assistant codes |

## 7. All links (verified 2026-07-07/08)

- The paper: https://transformer-circuits.pub/2026/workspace/index.html
  (Gurnee, Sofroniew, ... Lindsey. Anthropic. July 6, 2026)
- Official code: https://github.com/anthropics/jacobian-lens (Apache-2.0;
  includes lens fitting + visualization; does NOT include gradient pursuit)
- Independent replication (calibrate expectations):
  https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S/a-global-workspace-in-language-models
- Prior-art lens (fallback fitting approach):
  https://github.com/AlignmentResearch/tuned-lens
- Model: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct (gated;
  accept license; needs HF token)
- Eval harness: https://github.com/EleutherAI/lm-evaluation-harness
  (gsm8k and triviaqa tasks built in)
- GPU rental: https://www.runpod.io/pricing (H100 ~$2-3/hr; alternatives:
  https://lambda.ai/pricing , https://vast.ai/pricing )
- Community port datapoint (regression-fit feasibility, 4B on a 3090):
  https://starlog.is/articles/developer-tools/ninjahawk-subtext

## 8. Risk register

| Risk | Mitigation |
|---|---|
| Lens port to Llama fails / readouts garbage | Verify on repo's default model first; try tuned-lens-style regression fit; ask on repo issues |
| Gradient pursuit bugs | Synthetic-recovery validation (5.W2.2) before any real data |
| Score measures an artifact | Preregistration (sec 4), shuffle control, report S1-S3 separately |
| Weak 8B signal (expected per Nanda) | Decision rules make null publishable; optional 70B check |
| GPU cost runaway | Stop pods daily; cheap fitting path; downsample layers/items before scaling up |
| Single-token anchor too crude for TriviaQA entities | Report GSM8K (numeric) as primary; TriviaQA as secondary |
| Operator discouragement mid-project | Weekend 1 ends on the spider screenshot - a visible, shareable win |

## 9. Glossary (for the operator)

- Llama: Meta's downloadable AI model. We use it because we can see inside
  a brain we are holding; API models (Claude/GPT) only send finished text.
- Parameters / 8B: the model's size. 8 billion = small-but-real; cheap.
- GPU / H100: the special graphics card models run on. Rented, not bought.
- RunPod: hourly GPU rental website (Airbnb for supercomputers).
- HuggingFace (HF): the app store for open models and datasets.
- Activations / residual stream: the model's internal state-of-mind numbers
  at each layer while it processes text. What the lens reads.
- J-lens: Anthropic's technique - an averaged "wiggle test" that maps
  internal states to the vocabulary words they push toward.
- J-space / workspace: the ~25 named concepts an internal state decomposes
  into. The model's legible "thought recipe."
- Logprob: the model's built-in confidence in each word it produced. The
  incumbent trust signal; our baseline.
- GSM8K / TriviaQA: standard test sets (grade-school math; trivia) with
  known correct answers.
- AUC: grading score for predictors. Pick one right + one wrong answer at
  random; AUC = probability your score ranks the right one higher.
  0.5 = coin flip, 1.0 = perfect.
- Bootstrap CI: error bars computed by resampling; tells you whether a
  difference is real or luck.
- Chain-of-thought (CoT): prompting the model to reason step by step
  before answering.
- Preregistration: freezing your measurement definitions before seeing
  results, so you cannot fool yourself.

## 10. Deliverables ("done" looks like)

1. The spider screenshot (weekend 1).
2. A repo with: lens fitting for Llama, gradient-pursuit decomposition
   (with synthetic validation), scoring, eval pipeline. MIT or Apache
   license if ever published.
3. results.md: AUC table (S1-S3 vs B1-B4, both datasets, CIs), controls,
   3 worked examples with readout visuals.
4. A 1-2 page writeup in plain language (James will adapt for LinkedIn /
   an internal Ketryx pitch). State the caveats honestly: approximate lens
   fitting, single-token anchors, 8B scale, one model family.

## 11. Broader context (one paragraph, for the record)

This experiment came out of a July 2026 investigation into startup ideas on
top of the workspace paper (AI flight recorder, deliberation trust scoring,
insurer telematics, ontology installation, workspace persistence). All got
"not now" verdicts after adversarial research: the science is days old and
partially replicated, activation access is shrinking in the enterprise
(closed APIs won compliance certifications), funded players hold the
horizontal positions (Glacis, Vorlon, Goodfire, Purview), and regulatory
deadlines slipped (EU AI Act high-risk logging now Dec 2027+). The
experiment is the cheap option that appreciates in every branch: evidence
for an internal Ketryx "decision records + workspace trace" pitch, personal
credibility in the space, and a head start if the watchlist moves (watch:
Goodfire product announcements, Glacis traction, CTGT, replication papers,
enterprise open-weight share, EU AI Act dates).
