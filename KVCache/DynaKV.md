# DynaKV: Token-Wise Adaptive KV Cache Compression — A First Principles Deep Dive

## the problem:

When an LLM generates text, it doesn't "think about the whole sentence at once." It processes tokens one by one. At every step, the attention mechanism asks: which earlier tokens should I pay attention to right now?
To answer that, every previously-seen token must have its Key and Value vectors accessible. Computing them again each step would be catastrophically slow — O(n²) in compute every decoding step. So we cache them. That cache is the KV Cache.

These get stored in GPU VRAM and grown with every generated token.

Memory math:

For LLaMA-3-8B: 32 layers, 8 KV heads, head_dim=128, bfloat16 (2 bytes):

Memory per token = 2 × 32 × 8 × 128 × 2 bytes = 131,072 bytes ≈ 128KB

100K tokens × 128KB = ~12.8 GB  ← just the cache, before model weights

That's the wall. A 70B model is worse. Long-context RAG, agentic loops, multi-turn dialogue — they all hit it.

## the prior landscape:
two main camps for the exisitng solutions:

* Camp A : Seq level pruning - just throw away the tokens we think aren't important. This works by throwing out those tokens that have less attention scores but it has one problem - once evicted that information is gone forever

* Camp B : Dim Reduction - you keep all the tokens but reduce (shrink) each K/V from the 128 dim to some rank=r. It again has a problem that each token gets the same rank r (one fits all problem ) and at aggressive compression (v small r values) the quality collapses badly.

## the key insight from the paper

the information distribution is highly non uniform within these KV representaions and thus they vary significantly across different tokens. 

Think about what this means physically. Consider the sentence:

> "The mitochondria, which was discovered in 1857 by Albert von Kölliker, is the powerhouse of the cell."

The word "mitochondria" and "powerhouse" carry dense semantic content. The word "which" carries almost none. Yet uniform compression assigns them identical storage budgets. That's wasteful for content-rich tokens and unnecessarily compressive for content-poor ones.

This is the fundamental research bet: semantic content ≠ uniform across tokens, so compression should not be uniform either.

## The Architecture

there are 3 components in the architecture:

1. PCA Basis Transformation :

Raw K/V vectors have correlated dimensions. If you just drop the last d - r dimensions of a raw vector, you lose information somewhat randomly and there's no guarantee the first dimensions carry more signal than the last.
PCA solves this by rotating your coordinate system so that:

* Dimension 1 captures the most variance
* Dimension 2 captures the second most
...
* Dimension d captures the least

After this rotation, the trailing dimensions are genuinely the least informative. Now you can safely drop them.

Mechanically:

* Collect a calibration set of K/V activations (128M tokens for an 8B model)
* Compute the covariance matrix of these activations
* Decompose via SVD: C = UΣVᵀ
* The columns of U are your new basis vectors, ordered by variance captured
* Store matrix P (the top principal components) per layer and head

