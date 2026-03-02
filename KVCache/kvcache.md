# KV Cache — Short Notes

These are my short notes about whatever I studied about KV cache — how it came into being and what are the new techniques that are out there related to this.

------

## 1. Why KV Cache Exists

Transformers use **self-attention** to compare every word in a sentence. Text generation in Transformers is **auto-regressive**, i.e., for every token that is generated, it needs to take into account all the previous tokens that have been generated.

This means that during decoding of the 100th token, attention between all the previous 99 tokens is calculated from scratch.
 The **K, V, Q** for every single token is calculated again.

Time complexity:

O(n2)O(n^2)O(n2)

This is very expensive.

------

## 2. Why KV Caching Works

If you think about it, the keys and the values are calculated using:

K=input⋅WkK = \text{input} \cdot W_kK=input⋅WkV=input⋅WvV = \text{input} \cdot W_vV=input⋅Wv

In the above equations:

- The **input remains the same**
- The **model weights are fixed**

So there is no change. Hence, we can simply store them and reuse them for subsequent calculations.

There is no need to recompute them.

------

## 3. Core Idea Behind KV Cache

The question came:

> To generate the 100th token, why are we recalculating the previous 99 tokens?

The solution was:

Store all the **keys and values** of every token generated so far.

Using this approach, the decoding time complexity reduces to:

O(1)O(1)O(1)

(per token step during generation)

------

## 4. How KV Cache Works

### Step 1: Prefill Phase

- Calculate Q,K,VQ, K, VQ,K,V for all input tokens.
- Save the KKK and VVV matrices to the cache.

------

### Step 2: Incremental Decoding (Generation)

For every newly generated token:

- Calculate:

  Qnew,  Knew,  VnewQ_{\text{new}}, \; K_{\text{new}}, \; V_{\text{new}}Qnew,Knew,Vnew

- Append only the new KKK and VVV to the existing cache.

- Perform attention using:

  - Single QnewQ_{\text{new}}Qnew
  - Entire KcacheK_{\text{cache}}Kcache
  - Entire VcacheV_{\text{cache}}Vcache

------

## 5. The Trade-Off

There is a trade-off between **memory and speed**.

We reduced time complexity by avoiding recomputation of KKK and VVV, but:

- KV cache storage consumes a large amount of VRAM.
- Memory grows linearly with sequence length.

------

## 6. Attention Equation

For head hhh:

Head(h)=softmax(QhKhTdk)Vh\text{Head}(h) = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_k}}\right) V_hHead(h)=softmax(dkQhKhT)Vh

KV cache size:

KV Cache Size=H×sequence_length×dhead\text{KV Cache Size} = H \times \text{sequence\_length} \times d_{\text{head}}KV Cache Size=H×sequence_length×dhead

------

# Improvements Over Standard KV Cache

------

## 1. Multi-Query Attention (MQA)

### Problem

In standard Multi-Head Attention (MHA):

- Every head has its own KKK and VVV.
- This is bulky on VRAM.

------

### Insight

What if all query heads shared a single KKK and VVV?

That means:

- Compute KKK and VVV once.
- Share across all QhQ_hQh.

Qh=X⋅Wq,hQ_h = X \cdot W_{q,h}Qh=X⋅Wq,hK=X⋅WkK = X \cdot W_kK=X⋅WkV=X⋅WvV = X \cdot W_vV=X⋅WvHead(h)=softmax(QhKTdk)V\text{Head}(h) = \text{softmax}\left(\frac{Q_h K^T}{\sqrt{d_k}}\right) VHead(h)=softmax(dkQhKT)V

KV cache size becomes:

1×sequence_length×dhead1 \times \text{sequence\_length} \times d_{\text{head}}1×sequence_length×dhead

------

### Why Do This?

- Saves memory
- Higher throughput

Trade-off:

- Slight drop in performance
- Heads lose ability to attend to different relationship types independently

------

## 2. Grouped-Query Attention (GQA)

This is the sweet spot between standard MHA and MQA.

Let:

H=4(number of heads)H = 4 \quad \text{(number of heads)}H=4(number of heads)G=2(number of groups)G = 2 \quad \text{(number of groups)}G=2(number of groups)

------

### Standard MHA

Each head has its own K and V:

- K1,V1→H1K_1, V_1 \rightarrow H_1K1,V1→H1
- K2,V2→H2K_2, V_2 \rightarrow H_2K2,V2→H2
- K3,V3→H3K_3, V_3 \rightarrow H_3K3,V3→H3
- K4,V4→H4K_4, V_4 \rightarrow H_4K4,V4→H4

Total stored items:

2×4=82 \times 4 = 82×4=8

------

### GQA

Heads are divided into groups:

Group A:

- Shared KGA,VGAK_{GA}, V_{GA}KGA,VGA for H1,H2H_1, H_2H1,H2

Group B:

- Shared KGB,VGBK_{GB}, V_{GB}KGB,VGB for H3,H4H_3, H_4H3,H4

Total stored items:

2×2=42 \times 2 = 42×2=4

So memory usage is reduced by half.

------

## 3. Paged Attention

### Understanding Fragmentation

#### External Fragmentation

Suppose:

- We need 100 contiguous memory slots in GPU.
- 50 empty slots are at address A.
- 50 empty slots are at address J.

Even though total memory is available, it is not contiguous.

Model crashes.

This is called **external fragmentation**.

------

#### Internal Fragmentation

Suppose:

- Max sequence length = 2048.
- We reserve memory for 2048 tokens.
- We use only 10 tokens.

Remaining 2038 slots are wasted.

This is **internal fragmentation**.

------

### How Paged Attention Solves This

Inspired by how operating systems manage memory.

Idea:

- Break sequences into fixed-size blocks (pages).
- Example: block size = 4 tokens.

If we find 4 empty GPU slots anywhere:

- Store 4-token block there.
- Next 4-token block can go somewhere else.

We maintain a **block table**:

- Stores addresses of each block.

------

### Paged Attention Loop

1. **Block lookup**

   - Find address of block (via block table).

2. **Memory fetch**

   - GPU kernel fetches tokens from that address.

3. **Local math**

   - Compute:

     QKTQ K^TQKT

4. **Accumulate**

   - Store results.
   - Move to next block.
   - Repeat.

------

## 4. Prompt Caching

Need to study further:

Paper:
 https://proceedings.mlsys.org/paper_files/paper/2024/file/a66caa1703fe34705a4368c3014c1966-Paper-Conference.pdf