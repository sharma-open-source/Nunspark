# Streaming 30B/70B LLM Inference on a 16 GB Apple M4 using Weight Streaming + Speculative Decoding

## Overview

This report summarizes experiments performed on a **16 GB Apple M4
MacBook Pro** to evaluate whether very large language models (32B--70B
class) can be executed using a streaming runtime with a small resident
weight cache.

Rather than loading the entire model into memory, weights are streamed
on demand while speculative decoding reduces verifier work.

------------------------------------------------------------------------

# Hardware

  Component        Value
  ---------------- ------------------------
  Machine          Apple MacBook Pro (M4)
  Unified Memory   16 GB

------------------------------------------------------------------------

# Models Evaluated

## Target Models

-   Qwen2.5-32B-Instruct-4bit
-   Llama-3.3-70B-Instruct-4bit

## Draft Models

-   Qwen2.5-0.5B
-   Qwen2.5-3B
-   Qwen3-1.7B
-   Llama-3.2-1B

------------------------------------------------------------------------

# Major Findings

## 1. Resident cache size is not the primary bottleneck

Increasing the resident cache from 512 MB up to 12 GB produced only
marginal improvements in throughput and, in some cases, reduced
performance due to memory pressure.

The dominant factor was **speculative acceptance**, not weight
residency.

------------------------------------------------------------------------

## 2. Draft model quality matters more than draft size

  -----------------------------------------------------------------------
  Draft Model                                 Result
  ------------------------------------------- ---------------------------
  Qwen2.5-0.5B                                Low acceptance, \~0.43
                                              tok/s

  Qwen2.5-3B                                  Significant improvement
                                              (\~0.67 tok/s)

  Qwen3-1.7B                                  Best overall draft model
                                              (\~0.92--1.13 tok/s
                                              depending on workload)

  Qwen2.5-7B                                  Slower despite being larger
  -----------------------------------------------------------------------

The newer Qwen3-1.7B consistently outperformed larger Qwen2.5 draft
models.

------------------------------------------------------------------------

## 3. Optimal speculative window is draft-model dependent

### Qwen2.5-3B

Increasing draft tokens beyond 8 produced almost no improvement.

### Qwen3-1.7B

Performance improved steadily:

  Draft Tokens     Throughput
  -------------- ------------
  4                0.56 tok/s
  8                0.73 tok/s
  12               0.80 tok/s
  16               0.86 tok/s
  20               0.88 tok/s
  24               0.92 tok/s

Beyond 24 tokens, performance saturated.

------------------------------------------------------------------------

# Workload Dependence

Performance varied dramatically depending on the prompt.

  Workload                   Acceptance   Throughput
  ------------------------ ------------ ------------
  Vector DB comparison             2.99   0.49 tok/s
  SaaS billing design              3.45   0.55 tok/s
  JS alert function                5.41   0.92 tok/s
  React WebSocket hook             6.06   1.03 tok/s
  LRU Cache (TypeScript)           7.14   1.13 tok/s

The streaming engine remained constant; throughput tracked speculative
agreement between draft and target models.

Approximate relationship observed:

> Throughput ≈ Verifier Speed × Acceptance Multiplier

------------------------------------------------------------------------

# Fast Approximate Verification (accept-top-k)

Allowing acceptance of draft tokens within the target's top-k
predictions substantially reduced verifier passes.

## Qwen 32B

  Mode            TPS   Acceptance   Deviation
  ------------ ------ ------------ -----------
  Exact          1.13         7.14          0%
  Top-k = 5      2.83        18.18       21.7%
  Top-k = 10     2.65        18.18       26.7%

Top-k=5 offered the best speed/quality tradeoff among tested values.

------------------------------------------------------------------------

# Llama 70B Results

## Exact

-   Throughput: **0.89 tok/s**
-   Acceptance: **11.76**
-   Deviation: **0%**

## Top-k = 5

-   Throughput: **1.45 tok/s**
-   Acceptance: **20.0**
-   Deviation: **9.8%**

The Llama family exhibited much higher draft-target agreement, likely
because the draft and target models share architecture, tokenizer, and
training lineage.

------------------------------------------------------------------------

# Memory Efficiency

## Qwen2.5-32B

-   Model size: 18.43 GB
-   Resident weights: \~1.10 GB
-   Peak unified memory: \~3.1 GB

## Llama-3.3-70B

-   Model size: 39.69 GB
-   Resident weights: \~1.44 GB
-   Peak unified memory: \~3.4 GB

Both models operated while keeping only a small fraction of weights
resident.

------------------------------------------------------------------------

# Key Insights

-   Weight streaming is practical on 16 GB Apple Silicon.
-   Speculative decoding dominates overall performance.
-   Draft-target alignment is a stronger predictor of throughput than
    cache size.
-   Performance is highly workload dependent.
-   Approximate verification (top-k acceptance) enables significant
    throughput gains with controllable divergence.

------------------------------------------------------------------------

# Recommended Configurations

## Highest Exact Throughput

-   Target: Qwen2.5-32B-Instruct-4bit
-   Draft: Qwen3-1.7B-4bit
-   Draft Tokens: 24
-   Cache Budget: 1 GB

Typical throughput:

**0.9--1.1 tok/s**

------------------------------------------------------------------------

## Highest Approximate Throughput

-   Target: Qwen2.5-32B-Instruct-4bit
-   Draft: Qwen3-1.7B-4bit
-   Draft Tokens: 24
-   Accept Top-k: 5

Typical throughput:

**2.6--2.8 tok/s**

------------------------------------------------------------------------

## Highest Fidelity Large Model

-   Target: Llama-3.3-70B
-   Draft: Llama-3.2-1B
-   Accept Top-k: 1

Typical throughput:

**\~0.9 tok/s**

------------------------------------------------------------------------

# Future Work

-   Benchmark additional prompt categories (reasoning, RAG, refactoring,
    summarization).
-   Measure acceptance histograms rather than only averages.
-   Evaluate semantic correctness under approximate verification.
-   Explore adaptive top-k based on confidence.
-   Investigate smarter layer prefetching and cache policies.

------------------------------------------------------------------------

# Conclusion

These experiments demonstrate that large language models well beyond
physical memory size can be executed on a 16 GB Apple M4 using streamed
weights and speculative decoding.

The strongest finding is that **draft-target agreement---not resident
cache size---is the primary determinant of throughput**. With
appropriate draft models and speculative verification, a streamed 32B
model achieved over **1 tok/s losslessly** and nearly **3 tok/s** in
approximate mode, while a streamed 70B model remained usable with only
\~1.4 GB of resident weights.
