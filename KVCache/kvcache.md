# KV Cache: Study Notes

These are my short notes about whatever I studied about KV cache. How did it come into being and what are the new techniques that are out there related to this.

## Background: The Problem with Transformers

Transformers use self-attention to compare every word in a sentence. The text generation in Transformers is **auto-regressive**, i.e., for every token that is generated it needs to take into account all the previous tokens that have been generated.

This means that for the Transformer decoding of the 100th token, attention between all the other 99 tokens is calculated from scratch — K, V, Q for every single token is calculated and the time complexity is $O(n^2)$ which is very expensive.

## Why KV Cache Works

If you think about it, the keys and the values are calculated using:

$$
K = \text{input} \times W_k
$$

$$
V = \text{input} \times W_v
$$

In the above, you can see that the input remains the same and the model weights for the key and value are also fixed. So there is no change present and hence we can simply store and then use them for subsequent calculations. There is no need to re-calculate!

Then the question came: to generate the 100th token, why are we re-calculating the previous 99 tokens? So the solution was proposed: let us store all the keys and values of every token that we have generated so far. Turns out by using this approach the time complexity was reduced to $O(1)$.

## How Cache Works

### Step 1: Prefill
- Calculate Q, K, V for all the input tokens
- Save the K and V matrices to the cache

### Step 2: Incremental Decoding (Generation)

For every token that is generated:
- Calculate $Q_{new}$, $K_{new}$ and $V_{new}$ and append only the new K and V to the existing cache
- Perform attention using single $Q_{new}$ and entire $K_{cache}$ and $V_{cache}$

### Standard Attention Formula

$$
\text{Head}(h) = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_k}}\right) V_h
$$

**KV cache size** = $H \times \text{sequence\_length} \times d_{head}$

## The Problem

There is a trade-off between memory and speed — because although we reduced the time complexity by preventing the re-calculation of the K and V matrices, we also introduced space limitations as the KV cache storage takes a lot of VRAM.

## Improvements

### 1. Multi Query Attention (MQA)

**Problem:** Standard MHA gave every single attention head its own K and V → bulky on VRAM

**Insight:** What if all Q heads shared a single K and V value — meaning that we will calculate K and V once and then the same value will be shared across all the Q heads.

$$
Q_h = X \cdot W_{q,h}
$$

$$
K = X \cdot W_k
$$

$$
V = X \cdot W_v
$$

$$
\text{Head}(h) = \text{softmax}\left(\frac{Q_h K^T}{\sqrt{d_k}}\right) V
$$

**KV cache size** = $1 \times \text{seq\_length} \times d_{head}$

**Why do this?**
- Saves memory
- Higher throughput
- **Trade-off:** There is a slight drop in the model's performance as the heads share the same K and V values and lose the ability to attend to different types of relationships simultaneously

### 2. Grouped Query Attention (GQA)

It was the sweet spot between standard MHA and MQA.

Let $H = 4$ and the number of groups, $G = 2$

**Standard MHA:**
- $K_1, V_1$ = $H_1$
- $K_2, V_2$ = $H_2$
- $K_3, V_3$ = $H_3$
- $K_4, V_4$ = $H_4$

Total stored items = $2 \times 4 = 8$

**GQA:**
- $G = 2$
- $K_{GA}, V_{GA}$ = $H_1$ and $H_2$
- $K_{GB}, V_{GB}$ = $H_3$ and $H_4$

Total stored items = $2 \times 2 = 4$

### 3. Paged Attention

#### Understanding External and Internal Fragmentation

Let us say we have to store 100 tokens and for that we need to empty 100 slots in the GPU. If 50 slots are empty at address A and the other 50 are at address J, then the model will crash — this is called **external fragmentation**.

Let us say the max length is 2048 and we reserved this memory in the GPU, but then we only used 10 slots and the remaining 2038 were wasted — this is called **internal fragmentation**.

#### How Does Paged Attention Solve the Problem of Contiguous Memory?

It is inspired from how OS works. What it does is that it takes sequences and breaks them down into blocks (pages). The block size is fixed, let us say 4 tokens.

So if we find 4 empty slots in the GPU, we will simply fit our 4 pages (blocks) into it. And then if we find 4 empty slots somewhere else, the next 4 pages will be occupied there.

To store the address of the block we use a **block table** that stores the address of the blocks.

#### Paged Attention Loop

1. **Block lookup:** Find the address of the block in CPU
2. **Memory fetch:** GPU kernel will pick up the number of tokens it is supposed to from the address given with the help of block lookup table
3. **Local math:** Find $Q \cdot K^T$ of the number of tokens
4. **Accumulate:** Store the results and move on to the next address and repeat the same

### 4. Prompt Caching

Need to study — [paper](https://proceedings.mlsys.org/paper_files/paper/2024/file/a66caa1703fe34705a4368c3014c1966-Paper-Conference.pdf)