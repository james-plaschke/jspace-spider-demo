# Llama-3.1-8B thinks "spider" without saying it

**[Open the interactive explorer](https://james-plaschke.github.io/jspace-spider-demo/)** (works on desktop; ~1MB, fully self-contained).

## What you are looking at

Give Llama-3.1-8B-Instruct the prompt:

> Fact: The number of legs on the animal that spins webs is

The model answers "eight." The word *spider* appears nowhere in the prompt and
nowhere in the answer - but to get from "spins webs" to "eight," the model has
to think it. This page reads that thought out of the model's internal states,
mid-computation.

The heatmap at the bottom shows the rank of the token ` spider` in the lens
readout at every (position, layer) of the prompt. It surges to **rank 1 out of
128,256 vocabulary tokens** around layer 20, at the word "webs" - the middle
of the network, right when the clue arrives - and fades before the output
layers. Click any cell in the top grid to pin other tokens and explore.

## How it was made

- Method: the Jacobian lens from ["Verbalizable Representations Form a Global
  Workspace in Language Models"](https://transformer-circuits.pub/2026/workspace/index.html)
  (Gurnee et al., Anthropic, July 2026), using Anthropic's open-source
  [jacobian-lens](https://github.com/anthropics/jacobian-lens) code (Apache-2.0),
  which also generated this page.
- Model: [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
  To our knowledge this is the first public J-lens fitted for a Llama model
  (the paper studied Claude models; the released lenses cover Qwen).
- Fit: 100 wikitext prompts on a rented A100 (the repo's "cheap path" - the
  paper's full spec is 1000 prompts x 128 tokens). Overnight, roughly $14 of
  GPU time.

## Honest caveats

- This demonstrates that **the instrument works on Llama** - it is not an
  experimental result. It is Weekend 1 of a preregistered experiment testing
  whether workspace routing predicts answer correctness better than logprob
  baselines (GSM8K / TriviaQA); results will be written up separately,
  whatever they show.
- The lens fit is the approximate 100-prompt path, not the paper's exact
  spec. Readouts are single-vocabulary-token by construction.

*James Plaschke, with Claude. 2026-07-09.*
