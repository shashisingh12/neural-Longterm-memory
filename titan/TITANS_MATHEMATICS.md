# Titans Memory Architecture — Complete Mathematics

> Based on *Behrouz et al. (2501.00663)* — **"Titans: Learning to Memorize at Test Time"**
>
> This document provides the full mathematical formulation behind the Titans memory system,
> with proper notation, detailed variable descriptions, and direct code references.

---

## Table of Contents

1. [Notation and Variable Glossary](#1-notation-and-variable-glossary)
2. [Input Projections (Eq. 11)](#2-input-projections-eq-11)
3. [Memory MLP — The Storage Medium](#3-memory-mlp--the-storage-medium)
4. [Inner Loop — Surprise-Driven Memory Update](#4-inner-loop--surprise-driven-memory-update)
5. [Outer Loop — Meta-Learning (FOMAML)](#5-outer-loop--meta-learning-fomaml)
6. [Parallel Memory Updates (§3.2)](#6-parallel-memory-updates-§32)
7. [MAC Architecture — Memory as Context (§4.1)](#7-mac-architecture--memory-as-context-§41)
8. [Text Encoding Pipeline](#8-text-encoding-pipeline)
9. [Text Retrieval — Nearest-Neighbor Decoder](#9-text-retrieval--nearest-neighbor-decoder)
10. [Persistent Memory (§3.3)](#10-persistent-memory-§33)
11. [Hyperparameter Summary](#11-hyperparameter-summary)
12. [Code-to-Math Reference Table](#12-code-to-math-reference-table)

---

## 1. Notation and Variable Glossary

### 1.1 Scalars

| Symbol | Name | Default | Description |
|--------|------|---------|-------------|
| $d$ | Model dimension | 128 | The dimensionality of all embedding vectors, keys, values, queries, and MLP hidden layers. Every vector in the system lives in $\mathbb{R}^d$. Controls memory capacity — larger $d$ means more information per vector but slower computation. |
| $d_h$ | Attention head dimension | $d / H$ | The dimension each attention head operates in. Computed as $d_h = d / H$ where $H$ is the number of heads. For default $d=128$, $H=4$: $d_h = 32$. |
| $d_{\text{bert}}$ | BERT hidden size | 768 | The output dimensionality of the pretrained BERT encoder. Fixed by the chosen HuggingFace model. The projection layer maps from $d_{\text{bert}}$ down to $d$. |
| $t$ | Time step | — | An integer index representing the current turn or token position in a sequence. At time $t$, the system has processed inputs $x_1, x_2, \ldots, x_t$. |
| $\theta$ | Inner-loop learning rate | 0.01 | Controls how aggressively the MLP memory weights are updated on each write. Higher $\theta$ means faster memorization but more interference with existing memories. In the paper, $\theta_t$ can be data-dependent; this implementation uses a constant. |
| $\eta$ | Momentum decay coefficient | 0.9 | Controls how long the "surprise signal" persists after a surprising observation. When $\eta = 0.9$, after 10 steps the momentum retains $0.9^{10} \approx 0.35$ of its original magnitude. Higher $\eta$ means longer memory of recent surprises. |
| $\alpha$ | Forgetting gate | 0.02 | The fraction of old memory weight that is forgotten per step. Acts as weight decay on MLP parameters. After $n$ steps without reinforcement, a weight decays to $(1-\alpha)^n$ of its original value. At $\alpha = 0.02$: after 50 steps → 36%, after 100 steps → 13%. |
| $\lambda$ | Outer-loop learning rate | $1 \times 10^{-4}$ | The AdamW optimizer learning rate for the outer-loop parameters $\{W_K, W_V, W_Q, W_{\text{proj}}\}$. Much smaller than $\theta$ because the outer loop makes careful adjustments over many episodes. |
| $c$ | Gradient clip norm | 1.0 | Maximum allowed L2 norm for the outer-loop gradient vector. If $\|\nabla_\Phi \mathcal{L}\| > c$, the gradient is rescaled to have norm $c$. Prevents exploding gradients during meta-learning. |
| $N$ | Episode length | 8 | Number of texts sampled per training episode. Split into write and eval subsets. Must be $\leq$ total training texts. |
| $N_w$ | Write set size | 6 | Number of texts used for the write phase: $N_w = N \times (1 - r_{\text{eval}})$ where $r_{\text{eval}} = 0.25$. So $N_w = 8 \times 0.75 = 6$. |
| $N_e$ | Eval set size | 2 | Number of texts used for the eval phase: $N_e = N \times r_{\text{eval}} = 8 \times 0.25 = 2$. |
| $r_{\text{eval}}$ | Eval ratio | 0.25 | Fraction of each episode reserved for evaluation. Higher means more eval signal but less write data per episode. |
| $E$ | Number of epochs | 10 | How many full passes over the training corpus for the outer loop. Each epoch shuffles texts and creates $\lfloor |\text{texts}| / N \rfloor$ episodes. |
| $L$ | Sequence length (tokens) | $\leq 512$ | Maximum number of sub-word tokens the BERT tokenizer produces per text. Texts longer than $L$ are truncated. |
| $N_l$ | Number of MLP layers | 2 | Depth of the memory MLP. More layers = more expressive memory but harder to train. Default is 2 (one hidden + one output). |
| $H$ | Number of attention heads | 4 | For the MAC architecture only. Number of parallel attention heads in the self-attention module. Each head independently attends over the concatenated sequence. |
| $N_p$ | Number of persistent vectors | 4 | For the MAC architecture only. Number of learnable parameter vectors $\{p_1, \ldots, p_{N_p}\}$ that encode task-level knowledge. |
| $b$ | Chunk size | 16 | For parallel memory updates. Number of tokens processed simultaneously within one chunk. All gradients in a chunk are computed w.r.t. the same starting weights. |
| $K_{\text{top}}$ | Top-K retrieval | 3 | Number of nearest-neighbor text snippets retrieved from the decoder store per query. |
| $\tau$ | Similarity threshold | 0.05 | Minimum cosine similarity for a retrieved text to be included. Filters out irrelevant matches. |

### 1.2 Vectors

| Symbol | Shape | Description |
|--------|-------|-------------|
| $x_t$ | $\in \mathbb{R}^d$ | The input embedding at time step $t$. Produced by the text encoder: raw text → BERT → pooling → projection → L2-normalize. This is the starting point for all memory operations. |
| $k_t$ | $\in \mathbb{R}^d$ | The **key** vector at time $t$, computed as $k_t = W_K x_t + b_K$. Determines *under what address* the information is stored in the MLP memory. The MLP learns to map $k_t \mapsto v_t$. |
| $v_t$ | $\in \mathbb{R}^d$ | The **value** vector at time $t$, computed as $v_t = W_V x_t + b_V$. Represents *what information* to store. The MLP is trained so that $M(k_t) \approx v_t$. |
| $q_t$ | $\in \mathbb{R}^d$ | The **query** vector at time $t$, computed as $q_t = W_Q x_t + b_Q$. Used to *retrieve* from memory: $y_t = M(q_t)$. Separate from $k_t$ so that the system can learn different representations for writing vs. reading. |
| $\hat{v}_t$ | $\in \mathbb{R}^d$ | The MLP's prediction: $\hat{v}_t = M_{t-1}(k_t)$. The difference $\hat{v}_t - v_t$ is the "error" or "surprise" — how far the memory's current output is from the desired value. |
| $y_t$ | $\in \mathbb{R}^d$ | The read output: $y_t = M_t^*(q_t)$. The result of querying the memory MLP with $q_t$, without modifying weights. This is the "retrieved memory" vector. |
| $o_t$ | $\in \mathbb{R}^d$ | The gated output in MAC architecture: $o_t = y_t \odot M_t^*(y_t)$. Element-wise product of the attention output and a memory read of that output. |
| $h_t$ | $\in \mathbb{R}^d$ | The memory read vector in MAC: $h_t = M_{t-1}^*(q_t)$. One of the three branches concatenated for attention. |
| $z_i$ | $\in \mathbb{R}^d$ | Pre-activation output of MLP layer $i$: $z_i = W_i h_{i-1} + b_i$. Cached during the forward pass for use in backpropagation. |
| $h_i$ | $\in \mathbb{R}^d$ | Post-activation output of MLP layer $i$: $h_i = \sigma(z_i)$ for hidden layers, $h_i = z_i$ for the final layer. |
| $\delta_i$ | $\in \mathbb{R}^d$ | The error signal backpropagated to layer $i$. For the output layer: $\delta_2 = 2(\hat{v}_t - v_t)$. For hidden layers: $\delta_i = (W_{i+1}^\top \delta_{i+1}) \odot \sigma'(z_i)$. |
| $g_t$ | (collection) | The full gradient $\nabla_{\theta_M} \ell_t$ — a collection of gradient tensors, one for each MLP parameter. Contains $\{\partial\ell/\partial W_1, \partial\ell/\partial b_1, \partial\ell/\partial W_2, \partial\ell/\partial b_2\}$. |
| $p_i$ | $\in \mathbb{R}^d$ | The $i$-th learnable persistent memory vector in the MAC architecture. Input-independent, trained during the outer loop, fixed at inference. Encodes task-level knowledge (like "I am a technical assistant"). |

### 1.3 Matrices and Parameter Sets

| Symbol | Shape | Description |
|--------|-------|-------------|
| $W_K$ | $\in \mathbb{R}^{d \times d}$ | Key projection weight matrix. An **outer-loop** parameter. Maps input embeddings to the key space for memory writes. Trained by AdamW, not by the Titans inner-loop rule. |
| $b_K$ | $\in \mathbb{R}^d$ | Key projection bias vector. Paired with $W_K$. |
| $W_V$ | $\in \mathbb{R}^{d \times d}$ | Value projection weight matrix. An **outer-loop** parameter. Maps input embeddings to the value space — determines what information the memory should store. |
| $b_V$ | $\in \mathbb{R}^d$ | Value projection bias vector. Paired with $W_V$. |
| $W_Q$ | $\in \mathbb{R}^{d \times d}$ | Query projection weight matrix. An **outer-loop** parameter. Maps input embeddings to the query space for memory reads. Can differ from $W_K$ so that writes and reads use different representations. |
| $b_Q$ | $\in \mathbb{R}^d$ | Query projection bias vector. Paired with $W_Q$. |
| $W_i$ | $\in \mathbb{R}^{d \times d}$ | Weight matrix of the $i$-th MLP layer (for $i = 1, \ldots, N_l$). These are **inner-loop** parameters — their values ARE the memory. Updated at inference time by the Titans surprise rule. |
| $b_i$ | $\in \mathbb{R}^d$ | Bias vector of the $i$-th MLP layer. Also an inner-loop parameter, updated alongside $W_i$. |
| $S_t$ | (same shape as $\theta_M$) | The momentum buffer at time $t$. Maintains a running average of recent gradients, carrying the "surprise signal" forward in time. One buffer per MLP parameter: $S_{W_1}^{(t)}, S_{b_1}^{(t)}, S_{W_2}^{(t)}, S_{b_2}^{(t)}$. Initialized to zero at the start of each episode. |
| $M_t$ | (collection of $W_i, b_i$) | The full memory state at time $t$ — the set of all MLP layer weights and biases after $t$ write operations. $M_0$ is the initial (random or checkpoint-loaded) state. |
| $W_{\text{proj}}$ | $\in \mathbb{R}^{d \times d_{\text{bert}}}$ | Encoder projection matrix. Maps from the BERT embedding space ($\mathbb{R}^{768}$) down to the memory space ($\mathbb{R}^d$). An outer-loop trainable parameter. |
| $b_{\text{proj}}$ | $\in \mathbb{R}^d$ | Encoder projection bias. Paired with $W_{\text{proj}}$. |
| $\Phi$ | (set) | The complete set of outer-loop parameters: $\Phi = \{W_K, b_K, W_V, b_V, W_Q, b_Q, W_{\text{proj}}, b_{\text{proj}}\}$. These are what the meta-learning outer loop optimizes. |
| $\theta_M$ | (set) | The complete set of inner-loop (MLP memory) parameters: $\theta_M = \{W_1, b_1, W_2, b_2\}$. These are NOT optimized by a standard optimizer — they are updated in-place by the Titans rule. |

### 1.4 Functions and Operators

| Symbol | Description |
|--------|-------------|
| $M(\cdot)$ | The memory MLP forward function. $M(x)$ maps an input $x \in \mathbb{R}^d$ through the MLP layers to produce an output $\in \mathbb{R}^d$. The subscript $M_t$ denotes the MLP state after $t$ updates. The asterisk $M_t^*$ emphasizes "current state, no weight update." |
| $\sigma(\cdot)$ | The SiLU (Sigmoid Linear Unit) activation function: $\sigma(z) = z \cdot \text{sigmoid}(z) = \frac{z}{1 + e^{-z}}$. Applied element-wise. Used in all MLP hidden layers. Also called "Swish" in some literature. |
| $\sigma'(\cdot)$ | The derivative of SiLU: $\sigma'(z) = \text{sigmoid}(z) + z \cdot \text{sigmoid}(z) \cdot (1 - \text{sigmoid}(z))$. Needed for backpropagation through MLP hidden layers. |
| $\text{sg}(\cdot)$ | Stop-gradient operator (PyTorch `.detach()`). Returns the same value but blocks gradient flow. Used in the eval phase: $\text{sg}(W_V x)$ provides a target without letting gradients flow back through $W_V$. |
| $\nabla_\theta f$ | The gradient of scalar $f$ with respect to parameters $\theta$. Returns a collection of tensors matching the shapes of $\theta$. |
| $\odot$ | Element-wise (Hadamard) product. $[a \odot b]_i = a_i \cdot b_i$. Used in MAC output gating and SiLU derivative computation. |
| $\|\cdot\|^2$ | Squared L2 norm: $\|x\|^2 = \sum_i x_i^2$. Used in the MSE loss function. |
| $\|\cdot\|$ | L2 norm: $\|x\| = \sqrt{\sum_i x_i^2}$. Used for gradient clipping and L2 normalization. |
| $\text{softmax}(\cdot)$ | $\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$. Normalizes a vector to a probability distribution. Used in attention weights. |
| $\text{MSE}(a, b)$ | Mean squared error: $\text{MSE}(a, b) = \frac{1}{d}\sum_{i=1}^{d}(a_i - b_i)^2$. Note: PyTorch's `F.mse_loss` divides by $d$; the paper's Eq. 12 uses the sum (no division). The gradient direction is the same. |

### 1.5 Index Conventions

| Index | Range | Meaning |
|-------|-------|---------|
| $t$ | $1, 2, 3, \ldots$ | Time step (conversation turn or token position) |
| $i$ | $1, \ldots, N_w$ | Index within the write set of a training episode |
| $j$ | $1, \ldots, N_e$ | Index within the eval set of a training episode |
| $l$ | $1, \ldots, N_l$ | MLP layer index ($N_l = 2$ by default) |
| $h$ | $1, \ldots, H$ | Attention head index (MAC only) |
| $n$ | $1, \ldots, N_p$ | Persistent vector index (MAC only) |

---

## 2. Input Projections (Eq. 11)

Given an input embedding $x_t \in \mathbb{R}^d$, three learned linear projections transform it:

$$k_t = W_K x_t + b_K \quad \in \mathbb{R}^d \qquad \text{(key — what to store under)}$$

$$v_t = W_V x_t + b_V \quad \in \mathbb{R}^d \qquad \text{(value — what to store)}$$

$$q_t = W_Q x_t + b_Q \quad \in \mathbb{R}^d \qquad \text{(query — what to look up)}$$

**Variable roles:**
- $x_t$: The raw input embedding from the encoder. Contains the semantic meaning of the text, but in a generic BERT-derived space.
- $k_t$: The key — projects $x_t$ into a "storage address" space. The MLP learns to associate this address with the corresponding value.
- $v_t$: The value — projects $x_t$ into a "stored content" space. This is what the MLP should output when given the key.
- $q_t$: The query — projects $x_t$ into a "retrieval address" space. Separate from $k_t$ so writing and reading can use different representations (e.g., a question maps to the same retrieval address as the answer that was written).

**Why separate $W_K$, $W_V$, $W_Q$?** Consider storing "Python was created by Guido." The key $k_t$ might encode "Python-creator" (the address), the value $v_t$ might encode "Guido" (the content), and later a query "Who created Python?" would produce $q_t$ that points to the same MLP region as $k_t$.

**Code** — `titans/neural_memory.py:64-66`:
```python
self.W_K = nn.Linear(self.d_model, self.d_model)  # d×d weight + d bias
self.W_V = nn.Linear(self.d_model, self.d_model)
self.W_Q = nn.Linear(self.d_model, self.d_model)
```

---

## 3. Memory MLP — The Storage Medium

The memory $M$ is a multi-layer perceptron whose **weights are the memory**. For the default 2-layer architecture:

### 3.1 Forward Pass

$$z_1 = W_1 x + b_1 \quad \in \mathbb{R}^d$$

$$h_1 = \sigma(z_1) \quad \in \mathbb{R}^d$$

$$z_2 = W_2 h_1 + b_2 \quad \in \mathbb{R}^d$$

$$M(x) = z_2 \quad \in \mathbb{R}^d$$

**Variable roles:**
- $z_1$: Pre-activation at layer 1. The raw linear transformation before nonlinearity. Cached for backpropagation.
- $h_1$: Post-activation at layer 1. The SiLU nonlinearity introduces the capacity for nonlinear key→value mappings.
- $z_2$: Output layer (no activation). The final MLP output, which should approximate $v_t$ when given $k_t$.

### 3.2 SiLU Activation Function

$$\sigma(z) = z \cdot \text{sigmoid}(z) = \frac{z}{1 + e^{-z}}$$

**Properties:**
- Smooth, non-monotonic (dips slightly below zero around $z \approx -1.28$)
- $\sigma(0) = 0$, $\lim_{z \to \infty} \sigma(z) = z$, $\lim_{z \to -\infty} \sigma(z) = 0$
- Self-gated: the sigmoid acts as a learned gate on the linear pass-through

### 3.3 SiLU Derivative (needed for backpropagation)

$$\sigma'(z) = \text{sigmoid}(z) + z \cdot \text{sigmoid}(z) \cdot \big(1 - \text{sigmoid}(z)\big)$$

Let $s = \text{sigmoid}(z) = \frac{1}{1 + e^{-z}}$, then:

$$\sigma'(z) = s \cdot (1 + z \cdot (1 - s))$$

### 3.4 Memory Dimensions

| Parameter | Shape | Count |
|-----------|-------|-------|
| $W_1$ | $\mathbb{R}^{d \times d}$ | $d^2 = 16{,}384$ |
| $b_1$ | $\mathbb{R}^d$ | $d = 128$ |
| $W_2$ | $\mathbb{R}^{d \times d}$ | $d^2 = 16{,}384$ |
| $b_2$ | $\mathbb{R}^d$ | $d = 128$ |
| **Total** | | **$2d^2 + 2d = 33{,}024$** |

**Code** — `titans/neural_memory.py:171-186`:
```python
def _mlp_forward(self, x):
    h = x
    for i, layer in enumerate(self.mlp_layers):
        h = layer(h)                       # z_i = W_i·h + b_i
        if i < self.num_layers - 1:
            h = F.silu(h)                  # h_i = σ(z_i) for hidden layers
    return h                               # final layer: no activation
```

---

## 4. Inner Loop — Surprise-Driven Memory Update

The inner loop runs at **inference time**, updating MLP weights on-the-fly as new information arrives.

### 4.1 Associative Memory Loss (Eq. 12)

$$\ell_t = \| M_{t-1}(k_t) - v_t \|^2 = \sum_{i=1}^{d} \big( [M_{t-1}(k_t)]_i - [v_t]_i \big)^2$$

**Variable roles:**
- $\ell_t \in \mathbb{R}$: The **surprise** — a scalar measuring how far the memory's prediction is from the desired value. High $\ell_t$ = novel/surprising information. Low $\ell_t$ = familiar/already-memorized.
- $M_{t-1}(k_t)$: The MLP's current prediction. What the memory *currently thinks* the value for key $k_t$ should be.
- $v_t$: The desired value. What the memory *should* output for this key.

**Code** — `titans/neural_memory.py:264-265`:
```python
output = self._mlp_forward(k)          # M_{t-1}(k_t)
loss = F.mse_loss(output, v)           # ℓ_t = ||output - v||² / d
```

### 4.2 Gradient Computation — Full Backpropagation Derivation

We compute $g_t = \nabla_{\theta_M} \ell_t$ — the gradient of the surprise loss with respect to all MLP parameters.

**Step 1 — Forward pass** (cache intermediates):

$$z_1 = W_1 k_t + b_1 \qquad h_1 = \sigma(z_1) \qquad z_2 = W_2 h_1 + b_2 \qquad \hat{v}_t = z_2$$

- $\hat{v}_t \in \mathbb{R}^d$: The MLP's prediction (same as $M_{t-1}(k_t)$, just renamed for clarity in the derivation).

**Step 2 — Output error signal:**

$$\delta_2 = \frac{\partial \ell_t}{\partial z_2} = 2(\hat{v}_t - v_t) \quad \in \mathbb{R}^d$$

- $\delta_2$: The error at the output layer. Each element $[\delta_2]_i = 2([\hat{v}_t]_i - [v_t]_i)$ is proportional to how far the $i$-th output dimension is from target. The factor of 2 comes from differentiating the squared error.

**Step 3 — Output layer parameter gradients:**

$$\frac{\partial \ell_t}{\partial W_2} = \delta_2 \cdot h_1^\top \quad \in \mathbb{R}^{d \times d}$$

$$\frac{\partial \ell_t}{\partial b_2} = \delta_2 \quad \in \mathbb{R}^d$$

- $\delta_2 \cdot h_1^\top$: An outer product. Element $[i, j]$ equals $[\delta_2]_i \cdot [h_1]_j$ — how much the error in output dimension $i$ is attributed to the activation in hidden dimension $j$.

**Step 4 — Backpropagate through SiLU activation:**

$$\delta_1 = \big( W_2^\top \delta_2 \big) \odot \sigma'(z_1) \quad \in \mathbb{R}^d$$

- $W_2^\top \delta_2 \in \mathbb{R}^d$: The error signal projected back through the output layer's weights.
- $\sigma'(z_1) \in \mathbb{R}^d$: The element-wise SiLU derivative at the pre-activation values.
- $\odot$: Element-wise multiplication. Each dimension's error is scaled by how sensitive the activation was at that point.

**Step 5 — Hidden layer parameter gradients:**

$$\frac{\partial \ell_t}{\partial W_1} = \delta_1 \cdot k_t^\top \quad \in \mathbb{R}^{d \times d}$$

$$\frac{\partial \ell_t}{\partial b_1} = \delta_1 \quad \in \mathbb{R}^d$$

**Complete gradient collection:**

$$g_t = \nabla_{\theta_M} \ell_t = \left\{ \frac{\partial \ell_t}{\partial W_1},\; \frac{\partial \ell_t}{\partial b_1},\; \frac{\partial \ell_t}{\partial W_2},\; \frac{\partial \ell_t}{\partial b_2} \right\}$$

**Code** — `titans/neural_memory.py:270-274`:
```python
mlp_grads = torch.autograd.grad(
    loss, mlp_params,
    retain_graph=True,     # keep computation graph for outer-loop backward
    create_graph=False,    # first-order approximation: no Hessian
)
```

### 4.3 Momentum Update (Eq. 14)

$$S_t = \eta \cdot S_{t-1} - \theta \cdot g_t$$

**Variable roles:**
- $S_t$: The momentum buffer at time $t$. A collection of tensors with the same shapes as $\theta_M$. Acts as a "surprise accumulator" — when the memory encounters a surprising input, the gradient $g_t$ is large, and $S_t$ carries this signal forward.
- $\eta = 0.9$: Momentum decay. Determines how quickly past surprise fades. After $n$ steps: $|S_t| \propto \eta^n |S_{t-n}|$.
- $\theta = 0.01$: Inner-loop learning rate. Scales how much each gradient contributes.
- $S_0 = \mathbf{0}$: Momentum is zero-initialized at the start of each episode or conversation.

**Expanded per parameter:**

$$S_{W_1}^{(t)} = \eta \cdot S_{W_1}^{(t-1)} - \theta \cdot \frac{\partial \ell_t}{\partial W_1} \quad \in \mathbb{R}^{d \times d}$$

$$S_{b_1}^{(t)} = \eta \cdot S_{b_1}^{(t-1)} - \theta \cdot \frac{\partial \ell_t}{\partial b_1} \quad \in \mathbb{R}^d$$

$$S_{W_2}^{(t)} = \eta \cdot S_{W_2}^{(t-1)} - \theta \cdot \frac{\partial \ell_t}{\partial W_2} \quad \in \mathbb{R}^{d \times d}$$

$$S_{b_2}^{(t)} = \eta \cdot S_{b_2}^{(t-1)} - \theta \cdot \frac{\partial \ell_t}{\partial b_2} \quad \in \mathbb{R}^d$$

**Why momentum?** After a surprising observation (large $g_t$), the tokens that follow are likely related. Momentum carries the surprise signal forward so post-surprise tokens get memorized even if their individual surprise is low.

**Code** — `titans/neural_memory.py:281-282`:
```python
momentum = self._get_momentum_buffer(name)                      # S_{t-1}
new_momentum = self.momentum_decay * momentum - self.lr * grad  # S_t = η·S_{t-1} - θ·g_t
```

### 4.4 Weight Update with Forgetting (Eq. 13)

$$M_t = (1 - \alpha) \cdot M_{t-1} + S_t$$

**Expanded per parameter:**

$$W_1^{(t)} = (1 - \alpha) \cdot W_1^{(t-1)} + S_{W_1}^{(t)}$$

$$b_1^{(t)} = (1 - \alpha) \cdot b_1^{(t-1)} + S_{b_1}^{(t)}$$

$$W_2^{(t)} = (1 - \alpha) \cdot W_2^{(t-1)} + S_{W_2}^{(t)}$$

$$b_2^{(t)} = (1 - \alpha) \cdot b_2^{(t-1)} + S_{b_2}^{(t)}$$

**Variable role of $\alpha$ (forgetting gate):**
- $(1 - \alpha) \cdot M_{t-1}$: Decays all existing weights by a factor of $(1 - \alpha) = 0.98$. This slowly erases old memories.
- $S_t$: Adds the momentum-weighted gradient signal. This writes new information.
- **Decay schedule**: After $n$ steps without reinforcement, a weight decays to $(1 - \alpha)^n$:

| Steps $n$ | $(1-0.02)^n$ | Remaining Memory |
|-----------|--------------|------------------|
| 10 | $0.98^{10} = 0.817$ | 81.7% |
| 50 | $0.98^{50} = 0.364$ | 36.4% |
| 100 | $0.98^{100} = 0.133$ | 13.3% |
| 200 | $0.98^{200} = 0.018$ | 1.8% |

**Combined single-step formula** (substituting Eq. 14 into Eq. 13):

$$M_t = (1 - \alpha) \cdot M_{t-1} + \eta \cdot S_{t-1} - \theta \cdot \nabla_{\theta_M} \ell_t$$

**Code** — `titans/neural_memory.py:283`:
```python
param.data.mul_(1.0 - self.forget_gate).add_(new_momentum)
# equivalent to: param = (1-α)·param + S_t
```

### 4.5 Memory Read (Eq. 15)

$$y_t = M_t^*(q_t) \quad \in \mathbb{R}^d$$

Expanding through the 2-layer MLP with current weights:

$$y_t = W_2^{(t)} \cdot \sigma\!\left( W_1^{(t)} \cdot q_t + b_1^{(t)} \right) + b_2^{(t)}$$

**Variable roles:**
- $q_t = W_Q x_t + b_Q$: The query vector — projects the input into the retrieval address space.
- $M_t^*$: The MLP at its current weight state, used in inference mode (no updates).
- $y_t$: The retrieved memory vector. Not directly human-readable — it's a $d$-dimensional vector that needs to be decoded back to text via nearest-neighbor search.

**Code** — `titans/neural_memory.py:190-205`:
```python
def read(self, x, training=False):
    with torch.no_grad():             # no weight updates
        q = self.project_query(x)     # q_t = W_Q·x + b_Q
        return self._mlp_forward(q)   # y_t = M*(q_t)
```

### 4.6 Combined Inner-Loop Algorithm

$$\boxed{ \begin{aligned} &\textbf{Input: } x_t \in \mathbb{R}^d,\; M_{t-1},\; S_{t-1} \\ &\textbf{Output: } M_t,\; S_t,\; \ell_t \\ \\ &1.\quad k_t = W_K x_t + b_K \\ &2.\quad v_t = W_V x_t + b_V \\ &3.\quad \hat{v}_t = M_{t-1}(k_t) \\ &4.\quad \ell_t = \| \hat{v}_t - v_t \|^2 \\ &5.\quad g_t = \nabla_{\theta_M} \ell_t \\ &6.\quad S_t = \eta \cdot S_{t-1} - \theta \cdot g_t \\ &7.\quad M_t = (1 - \alpha) \cdot M_{t-1} + S_t \end{aligned} }$$

---

## 5. Outer Loop — Meta-Learning (FOMAML)

The outer loop trains $\Phi = \{W_K, b_K, W_V, b_V, W_Q, b_Q, W_{\text{proj}}, b_{\text{proj}}\}$ so that the inner loop (Section 4) works effectively.

### 5.1 Episodic Training Structure

$$\textbf{For } e = 1 \text{ to } E \text{ (epochs):}$$

$$\quad \text{Shuffle training texts}$$

$$\quad \textbf{For each episode of } N \text{ texts } \{t_1, \ldots, t_N\}:$$

$$\qquad 1.\; M_0 \leftarrow \text{RandomInit}(),\quad S_0 \leftarrow \mathbf{0}$$

$$\qquad 2.\; x_i = \text{Encoder}(t_i) \quad \forall\, i = 1, \ldots, N$$

$$\qquad 3.\; \text{Write set} = \{x_1, \ldots, x_{N_w}\},\quad \text{Eval set} = \{x_{N_w+1}, \ldots, x_N\}$$

$$\qquad 4.\; \text{Compute write losses } \ell_i^{\text{write}} \text{ for } i = 1, \ldots, N_w$$

$$\qquad 5.\; \text{Compute eval losses } \ell_j^{\text{eval}} \text{ for } j = 1, \ldots, N_e$$

$$\qquad 6.\; \text{Outer gradient step on } \Phi$$

**Why reset $M_0$ each episode?** Each episode simulates a fresh conversation. The MLP starts from scratch, forcing the outer loop to learn projections that generalize — not ones that only work for a specific weight initialization or text ordering.

**Code** — `titans/trainer.py:102-120`

### 5.2 Write Phase Loss

For each text $i = 1, \ldots, N_w$ in the write set:

$$k_i = W_K x_i + b_K \qquad \text{(computation graph alive through } W_K \text{)}$$

$$v_i = W_V x_i + b_V \qquad \text{(computation graph alive through } W_V \text{)}$$

$$\ell_i^{\text{write}} = \big\| M_{i-1}(k_i) - v_i \big\|^2$$

Then apply the inner-loop update (Eq. 13–14) to obtain $M_i$ from $M_{i-1}$.

**Gradient with respect to $W_K$** (via chain rule):

$$\frac{\partial \ell_i^{\text{write}}}{\partial W_K} = 2\big(\hat{v}_i - v_i\big)^\top \cdot \frac{\partial M_{i-1}(k_i)}{\partial k_i} \cdot \frac{\partial k_i}{\partial W_K}$$

The Jacobian of the MLP with respect to its input:

$$\frac{\partial M(k)}{\partial k} = W_2 \cdot \text{diag}\!\big(\sigma'(W_1 k + b_1)\big) \cdot W_1 \quad \in \mathbb{R}^{d \times d}$$

Combining:

$$\frac{\partial \ell_i^{\text{write}}}{\partial W_K} = 2\big(\hat{v}_i - v_i\big)^\top \cdot W_2 \cdot \text{diag}\!\big(\sigma'(z_1)\big) \cdot W_1 \cdot x_i^\top$$

**Gradient with respect to $W_V$** (simpler — $v_i$ appears directly in the loss):

$$\frac{\partial \ell_i^{\text{write}}}{\partial W_V} = -2\big(M_{i-1}(k_i) - v_i\big) \cdot x_i^\top \quad \in \mathbb{R}^{d \times d}$$

**Code** — `titans/trainer.py:148-152`:
```python
for i in range(n_write):
    x_t = write_embeds[i]
    loss_t = self.memory.write(x_t, training=True)  # graph alive for W_K, W_V
    write_losses.append(loss_t)
```

### 5.3 Eval Phase Loss

After all $N_w$ writes, the MLP is at state $M_{N_w}$. Now test retrieval quality using held-out eval texts $j = 1, \ldots, N_e$:

$$q_j = W_Q x_j + b_Q \qquad \text{(query projection)}$$

$$\hat{v}_j = M_{N_w}(q_j) \qquad \text{(read from updated MLP)}$$

$$v_j^* = \text{sg}(W_V x_j + b_V) \qquad \text{(target with stop-gradient)}$$

$$\ell_j^{\text{eval}} = \big\| \hat{v}_j - v_j^* \big\|^2$$

**Variable roles:**
- $q_j$: The query for the $j$-th eval text. Produced by $W_Q$ — the parameter this loss is specifically training.
- $\hat{v}_j$: What the MLP returns when queried. If the write phase worked well AND $W_Q$ maps to the right region, this should be close to $v_j^*$.
- $v_j^* = \text{sg}(W_V x_j)$: The target value, with **stop-gradient**. This ensures $\ell_j^{\text{eval}}$ only trains $W_Q$, not $W_V$ (which was already trained by the write losses).
- $\text{sg}(\cdot)$: Stop-gradient — the value passes through but no gradients flow back during `.backward()`.

**Gradient with respect to $W_Q$:**

$$\frac{\partial \ell_j^{\text{eval}}}{\partial W_Q} = 2\big(\hat{v}_j - v_j^*\big)^\top \cdot \frac{\partial M_{N_w}(q_j)}{\partial q_j} \cdot x_j^\top$$

$$= 2\big(\hat{v}_j - v_j^*\big)^\top \cdot W_2^{(N_w)} \cdot \text{diag}\!\Big(\sigma'\!\big(W_1^{(N_w)} q_j + b_1^{(N_w)}\big)\Big) \cdot W_1^{(N_w)} \cdot x_j^\top$$

**Code** — `titans/trainer.py:156-161`:
```python
for i in range(n_eval):
    x_t = eval_embeds[i]
    retrieved = self.memory.read(x_t, training=True)     # M_{Nw}(W_Q·x)
    target_v = self.memory.project_value(x_t)            # W_V·x
    eval_loss = F.mse_loss(retrieved, target_v.detach()) # .detach() = sg()
```

### 5.4 Outer Loss and Parameter Update

**Total outer loss:**

$$\mathcal{L}_{\text{outer}} = \sum_{i=1}^{N_w} \ell_i^{\text{write}} + \sum_{j=1}^{N_e} \ell_j^{\text{eval}}$$

**Outer gradient:**

$$\nabla_\Phi \mathcal{L}_{\text{outer}} = \sum_{i=1}^{N_w} \frac{\partial \ell_i^{\text{write}}}{\partial \Phi} + \sum_{j=1}^{N_e} \frac{\partial \ell_j^{\text{eval}}}{\partial \Phi}$$

**Gradient clipping:**

$$\text{If } \|\nabla_\Phi \mathcal{L}_{\text{outer}}\| > c: \quad \nabla_\Phi \mathcal{L}_{\text{outer}} \leftarrow c \cdot \frac{\nabla_\Phi \mathcal{L}_{\text{outer}}}{\|\nabla_\Phi \mathcal{L}_{\text{outer}}\|}$$

**Parameter update (AdamW):**

$$\Phi \leftarrow \text{AdamW}\!\big(\Phi,\; \nabla_\Phi \mathcal{L}_{\text{outer}},\; \text{lr}=\lambda\big)$$

The AdamW update rule internally computes adaptive moment estimates $m_t, \hat{v}_t$ and applies weight decay $w$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\Phi \mathcal{L}$$

$$\hat{v}_t = \beta_2 \hat{v}_{t-1} + (1 - \beta_2) (\nabla_\Phi \mathcal{L})^2$$

$$\Phi \leftarrow \Phi - \lambda \left( \frac{m_t / (1 - \beta_1^t)}{\sqrt{\hat{v}_t / (1 - \beta_2^t)} + \epsilon} + w \cdot \Phi \right)$$

where $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $w = 0$ (default).

**Code** — `titans/trainer.py:165-178`:
```python
outer_loss = sum(write_losses) + sum(eval_losses)
self.optimizer.zero_grad()
outer_loss.backward()
torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip)
self.optimizer.step()
```

### 5.5 First-Order vs Second-Order Approximation

**Full second-order (MAML)** would differentiate through the inner-loop updates:

$$\frac{\partial \ell_j^{\text{eval}}}{\partial W_K} = \frac{\partial \ell_j^{\text{eval}}}{\partial M_{N_w}} \cdot \frac{\partial M_{N_w}}{\partial M_{N_w - 1}} \cdot \frac{\partial M_{N_w - 1}}{\partial M_{N_w - 2}} \cdots \frac{\partial M_1}{\partial W_K}$$

Each Jacobian $\frac{\partial M_i}{\partial M_{i-1}}$ involves second derivatives (the Hessian):

$$\frac{\partial M_i}{\partial M_{i-1}} = (1 - \alpha) \cdot I + \eta \cdot \frac{\partial S_{i-1}}{\partial M_{i-1}} - \theta \cdot \frac{\partial^2 \ell_i}{\partial M_{i-1}^2}$$

- $\frac{\partial^2 \ell_i}{\partial M_{i-1}^2}$: The **Hessian** of the associative loss with respect to MLP parameters. Computing this is expensive: $O(|\theta_M|^2)$ per step.
- $I$: Identity matrix of appropriate dimension.

**First-order approximation (FOMAML)** — what this code uses:

$$\frac{\partial M_i}{\partial M_{i-1}} \approx \mathbf{0} \quad \text{(inner-loop updates treated as constants)}$$

This means:
- Write losses $\ell_i^{\text{write}}$ → gradients flow to $W_K$, $W_V$ through the **loss computation** (not through MLP updates)
- Eval losses $\ell_j^{\text{eval}}$ → gradients flow to $W_Q$ through the **read computation**
- **No gradients** flow through the chain $M_0 \to M_1 \to \cdots \to M_{N_w}$

Implemented by `create_graph=False` in `torch.autograd.grad`.

### 5.6 Gradient Flow Diagram

```
                    OUTER LOSS  𝓛 = Σ(write) + Σ(eval)
                         │
    ┌────────────────────┼──────────────────────────────────┐
    │                    │                                  │
    │   ℓ₁ʷ = ‖M₀(W_K·x₁) - W_V·x₁‖²                   │
    │     │         │            │                          │
    │     │      ∂/∂W_K       ∂/∂W_V                       │
    │     │                                                 │
    │     ▼  inner update: M₀ → M₁  ✗ (no grad through)   │
    │                                                       │
    │   ℓ₂ʷ = ‖M₁(W_K·x₂) - W_V·x₂‖²                   │
    │     │         │            │                          │
    │     │      ∂/∂W_K       ∂/∂W_V                       │
    │     │                                                 │
    │     ▼  inner updates: M₁ → ··· → M₆  ✗              │
    │                                                       │
    │   ℓ₁ᵉ = ‖M₆(W_Q·x₇) - sg(W_V·x₇)‖²              │
    │                  │                                    │
    │               ∂/∂W_Q                                  │
    │                                                       │
    │   ℓ₂ᵉ = ‖M₆(W_Q·x₈) - sg(W_V·x₈)‖²              │
    │                  │                                    │
    │               ∂/∂W_Q                                  │
    │                                                       │
    └──────────────────┬────────────────────────────────────┘
                       │
                       ▼
            AdamW step on Φ = {W_K, W_V, W_Q, W_proj}
```

---

## 6. Parallel Memory Updates (§3.2)

The sequential inner-loop update processes one token at a time ($O(T)$ sequential steps). Parallel updates process tokens in **chunks of size $b$**, enabling GPU-friendly batched computation.

### 6.1 Chunked Gradient Computation (Eq. 17)

**Key insight**: Within a chunk of $b$ tokens, all gradients are computed w.r.t. the **same starting weights** $M_0$ (the weights at the beginning of the chunk).

For a single linear layer $W_0$ (simplified case), the per-token gradient is:

$$\nabla_{W_0} \ell(W_0;\, x_i) = 2 \cdot (W_0 k_i - v_i) \cdot k_i^\top \quad \in \mathbb{R}^{d \times d}$$

**Variable roles:**
- $W_0 \in \mathbb{R}^{d \times d}$: The weight matrix at the **start** of the chunk. Same for all $b$ tokens.
- $k_i \in \mathbb{R}^d$: The $i$-th key in the chunk.
- $v_i \in \mathbb{R}^d$: The $i$-th value in the chunk.
- $W_0 k_i \in \mathbb{R}^d$: What the memory predicts for key $k_i$.
- $(W_0 k_i - v_i) \in \mathbb{R}^d$: The prediction error for the $i$-th token.

**Batched as matrix operations** (all $b$ tokens at once):

$$K \in \mathbb{R}^{b \times d} \quad \text{(stacked keys)} \qquad V \in \mathbb{R}^{b \times d} \quad \text{(stacked values)}$$

$$\text{Predictions} = W_0 K^\top \quad \in \mathbb{R}^{d \times b}$$

$$\text{Errors} = \text{Predictions} - V^\top \quad \in \mathbb{R}^{d \times b}$$

$$G_i = 2 \cdot \text{Errors}_{:,\,i} \cdot K_{i,\,:}^\top \quad \in \mathbb{R}^{d \times d} \qquad \text{for } i = 1, \ldots, b$$

**Code** — `titans/parallel_memory.py:176-201`:
```python
predictions = W_0 @ keys.T                             # (d, b)
errors = predictions - values.T                         # (d, b)
grads = errors.T.unsqueeze(2) * keys.unsqueeze(1)      # (b, d, d)
return 2.0 * grads
```

### 6.2 Cumulative Decay Factors

The forgetting gate $\alpha$ introduces cumulative weight decay within a chunk:

$$\beta_t = \prod_{j=1}^{t} (1 - \alpha_j) \qquad \text{(cumulative decay up to position } t \text{)}$$

For constant $\alpha$:

$$\beta_t = (1 - \alpha)^t$$

**Variable roles:**
- $\beta_t \in (0, 1]$: How much of the original weights $M_0$ remain after $t$ forgetting steps. $\beta_1 = (1-\alpha)$, $\beta_b = (1-\alpha)^b$.
- $\alpha_j$: Per-token forgetting rate (constant in this implementation).

**Decay ratios** used to scale gradients:

$$B_i = \frac{\beta_b}{\beta_i} = \frac{(1-\alpha)^b}{(1-\alpha)^i} = (1-\alpha)^{b-i}$$

- $B_i$: Scales the $i$-th token's gradient to account for how much it will decay by the end of the chunk. Earlier tokens ($i$ small) get **larger** scaling because their effect decays more.

**Numerically stable computation** (log-space to avoid underflow):

$$\log \beta_t = \sum_{j=1}^{t} \log(1 - \alpha_j)$$

$$\beta_t = \exp\!\big(\text{cumsum}(\log(1 - \alpha))\big)$$

**Code** — `titans/parallel_memory.py:128-154`:
```python
def compute_cumulative_decay(alphas):
    log_decay = torch.log1p(-alphas)               # log(1-α) for each token
    cum_log_decay = torch.cumsum(log_decay, dim=0)  # cumulative sum in log-space
    return torch.exp(cum_log_decay)                 # β_t = exp(Σ log(1-α_j))

def compute_beta_ratios(betas):
    beta_T = betas[-1]                              # β_b
    return beta_T / betas.clamp(min=1e-10)          # β_b / β_i
```

### 6.3 Parallel Associative Scan (Eq. 18)

The momentum recurrence is a **first-order linear recurrence**:

$$S_t = \eta \cdot S_{t-1} + u_t \qquad \text{where } u_t = -\theta \cdot g_t$$

**Variable roles:**
- $S_t$: Momentum at position $t$ within the chunk.
- $\eta$: Momentum decay (gate value). Scalar, same for all positions.
- $u_t = -\theta \cdot g_t$: The scaled negative gradient input at position $t$.

This can be computed in parallel using the **associative operator**:

$$(a_1, b_1) \circ (a_2, b_2) = (a_1 \cdot a_2,\;\; a_2 \cdot b_1 + b_2)$$

where $a_t = \eta$ (gate) and $b_t = u_t$ (input).

**Proof of associativity** (required for parallel scan correctness):

$$\big((a_1, b_1) \circ (a_2, b_2)\big) \circ (a_3, b_3)$$

$$= (a_1 a_2,\; a_2 b_1 + b_2) \circ (a_3, b_3)$$

$$= (a_1 a_2 a_3,\; a_3(a_2 b_1 + b_2) + b_3)$$

$$= (a_1 a_2 a_3,\; a_2 a_3 b_1 + a_3 b_2 + b_3)$$

$$(a_1, b_1) \circ \big((a_2, b_2) \circ (a_3, b_3)\big)$$

$$= (a_1, b_1) \circ (a_2 a_3,\; a_3 b_2 + b_3)$$

$$= (a_1 a_2 a_3,\; a_2 a_3 b_1 + a_3 b_2 + b_3) \quad \checkmark \text{ (same result)}$$

**Blelloch parallel prefix scan algorithm:**

Computes all $S_1, S_2, \ldots, S_b$ in $O(\log b)$ sequential steps (with $O(b)$ total work):

**Up-sweep (reduce):**

$$\text{For } l = 0, 1, \ldots, \lceil\log_2 b\rceil - 1:$$

$$\quad \text{stride} = 2^{l+1}$$

$$\quad \text{For all } i \text{ where } i \bmod \text{stride} = \text{stride} - 1:$$

$$\qquad j = i - 2^l$$

$$\qquad (a_i, b_i) \leftarrow (a_j, b_j) \circ (a_i, b_i) = (a_j \cdot a_i,\;\; a_i \cdot b_j + b_i)$$

**Down-sweep:** Extract all prefix sums from the tree structure.

For small chunks ($b \leq 64$), sequential scan is used (overhead of parallelism exceeds benefit):

$$S_0 = u_0, \qquad S_t = \eta \cdot S_{t-1} + u_t \quad \text{for } t = 1, \ldots, b-1$$

**Code** — `titans/parallel_memory.py:37-125`

### 6.4 Final Chunk Update (Eq. 16)

After computing all momentum values $S_1, \ldots, S_b$ via the scan, the final weight update:

$$M_b = \beta_b \cdot M_0 + S_b$$

**Variable roles:**
- $M_b$: Memory weights at the end of the chunk.
- $\beta_b = (1-\alpha)^b$: Total decay over the chunk — how much of the original $M_0$ survives.
- $M_0$: Weight matrix at the start of the chunk.
- $S_b$: Final accumulated momentum — encodes all $b$ gradient updates.

**Closed-form for $S_b$** (expanding the recurrence):

$$S_b = \eta^b \cdot S_0 - \theta \sum_{i=1}^{b} \eta^{b-i} \cdot g_i$$

**Variable roles in the closed form:**
- $\eta^b \cdot S_0$: The "carry" from the previous chunk's momentum, decayed by $\eta^b$.
- $\eta^{b-i}$: How much the $i$-th token's gradient decays by position $b$. Token 1's gradient decays by $\eta^{b-1}$; token $b$'s gradient has decay $\eta^0 = 1$.

**Geometric sum of momentum** (when all $\eta$ are equal and $\eta \neq 1$):

$$\sum_{i=0}^{b-1} \eta^i = \frac{1 - \eta^b}{1 - \eta}$$

**Code** — `titans/parallel_memory.py:370-378`:
```python
if abs(eta - 1.0) > 1e-8:
    eta_sum = (1.0 - eta ** chunk_size) / (1.0 - eta)   # geometric sum
else:
    eta_sum = float(chunk_size)                           # η=1 edge case

new_momentum = (eta ** chunk_size) * momentum - theta * eta_sum * grad
param.data.mul_(chunk_beta).add_(new_momentum)   # M_b = β_b·M_0 + S_b
```

---

## 7. MAC Architecture — Memory as Context (§4.1)

The MAC variant integrates memory into a self-attention mechanism with three information branches.

### 7.1 Three-Branch Input (Eq. 21–22)

**Eq. 21 — Query and memory read:**

$$q_t = W_Q \cdot S^{(t)} + b_Q \quad \in \mathbb{R}^d$$

$$h_t = M_{t-1}^*(q_t) \quad \in \mathbb{R}^d$$

**Variable roles:**
- $S^{(t)} \in \mathbb{R}^d$: The encoded current input (from the text encoder).
- $q_t$: Query derived from the current input.
- $h_t$: Historical memory read — what the MLP remembers that is relevant to the current input.

**Eq. 22 — Concatenate three branches:**

$$\tilde{S}^{(t)} = \big[\, p_1,\; p_2,\; \ldots,\; p_{N_p} \;\big\|\; h_t \;\big\|\; S^{(t)} \,\big] \quad \in \mathbb{R}^{(N_p + 2) \times d}$$

**Variable roles:**
- $p_1, \ldots, p_{N_p} \in \mathbb{R}^d$: Learnable persistent vectors. `nn.Parameter` tensors that encode task-level knowledge. Input-independent — the same vectors are used regardless of the prompt.
- $h_t \in \mathbb{R}^d$: Long-term memory read. Provides historical context.
- $S^{(t)} \in \mathbb{R}^d$: Current input. The immediate context.
- $\|$: Sequence concatenation along the token dimension.
- $\tilde{S}^{(t)}$: The combined sequence, with $N_p + 2$ tokens, each of dimension $d$.

**Dimensions** (with defaults $N_p = 4$, $d = 128$):
- Persistent vectors: $(4, 128)$
- Memory read: $(1, 128)$
- Current input: $(1, 128)$
- Concatenated: $\tilde{S}^{(t)} \in \mathbb{R}^{6 \times 128}$

**Code** — `titans/mac_layer.py:112-123`:
```python
s_t = self.encoder.encode(prompt)                        # S^(t) ∈ ℝ^d
h_t = self.memory.read(s_t)                              # h_t ∈ ℝ^d
p_vecs = self.persistent_vectors(batch_size=1)           # (1, N_p, d)
s_tilde = torch.cat([p_vecs, h_t_seq, s_t_seq], dim=1)  # (1, N_p+2, d)
```

### 7.2 Self-Attention (Eq. 23)

Multi-head self-attention over $\tilde{S}^{(t)}$:

**Step 1 — Pre-norm:**

$$\hat{S} = \text{LayerNorm}(\tilde{S}^{(t)}) \quad \in \mathbb{R}^{(N_p+2) \times d}$$

**Step 2 — QKV projections** (these are **attention** projections, separate from the memory $W_K, W_V, W_Q$):

$$Q^a = \hat{S} \cdot W_Q^a \quad \in \mathbb{R}^{(N_p+2) \times d}$$

$$K^a = \hat{S} \cdot W_K^a \quad \in \mathbb{R}^{(N_p+2) \times d}$$

$$V^a = \hat{S} \cdot W_V^a \quad \in \mathbb{R}^{(N_p+2) \times d}$$

**Variable roles:**
- $W_Q^a, W_K^a, W_V^a \in \mathbb{R}^{d \times d}$: Attention QKV projection matrices. The superscript $a$ distinguishes "attention" from "memory" projections.
- $Q^a$: How each token wants to query the sequence.
- $K^a$: What each token is discoverable by.
- $V^a$: What content each token contributes when attended to.

**Step 3 — Split into $H$ heads:**

For head $h = 1, \ldots, H$:

$$Q_h = Q^a_{:,\; (h-1)d_h\, :\, hd_h} \quad \in \mathbb{R}^{(N_p+2) \times d_h}$$

$$K_h, V_h \text{ split identically from } K^a, V^a$$

where $d_h = d / H$ is the per-head dimension.

**Step 4 — Scaled dot-product attention per head:**

$$A_h = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right) \quad \in \mathbb{R}^{(N_p+2) \times (N_p+2)}$$

**Variable roles:**
- $Q_h K_h^\top \in \mathbb{R}^{(N_p+2) \times (N_p+2)}$: Raw attention scores. Element $[i,j]$ measures how much token $i$ should attend to token $j$.
- $\sqrt{d_h}$: Scaling factor to prevent dot products from growing large with dimension, which would push softmax into extreme (near-0 or near-1) values.
- $A_h$: Attention weight matrix. Each row sums to 1. Row $i$ gives the distribution of attention from token $i$ over all tokens.

$$\text{head}_h = A_h \cdot V_h \quad \in \mathbb{R}^{(N_p+2) \times d_h}$$

**Step 5 — Concatenate heads and project:**

$$\text{MultiHead} = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \cdot W_{\text{out}} \quad \in \mathbb{R}^{(N_p+2) \times d}$$

- $W_{\text{out}} \in \mathbb{R}^{d \times d}$: Output projection matrix. Mixes the outputs of all heads.

**Step 6 — Residual connection:**

$$y = \tilde{S}^{(t)} + \text{Dropout}(\text{MultiHead}) \quad \in \mathbb{R}^{(N_p+2) \times d}$$

Extract the current-input position output:

$$y_t = y_{N_p + 2} \quad \in \mathbb{R}^d \qquad \text{(last position in the sequence)}$$

**What the attention mechanism decides:**
- Attending to $p_i$ positions → incorporates task knowledge
- Attending to $h_t$ position → incorporates historical memory
- Attending to $S^{(t)}$ position → prioritizes current input

**Code** — `titans/attention.py:63-103`

### 7.3 Memory Write (Eq. 24)

The **attention output** (not the raw input) is written into memory:

$$M_t = \text{TitansUpdate}(M_{t-1},\; y_t)$$

This means the memory stores an attention-refined representation that already incorporates context from all three branches.

**Code** — `titans/mac_layer.py:138`:
```python
surprise = self.memory.write(y_t.detach())
```

### 7.4 Output Gating (Eq. 25)

$$o_t = y_t \odot M_t^*(y_t) \quad \in \mathbb{R}^d$$

**Variable roles:**
- $y_t \in \mathbb{R}^d$: The attention output — the model's "thought" based on all three sources.
- $M_t^*(y_t) \in \mathbb{R}^d$: A fresh read from the **just-updated** memory, using $y_t$ as query.
- $\odot$: Element-wise multiplication (Hadamard product).
- $o_t \in \mathbb{R}^d$: The gated output. Dimensions where the memory "agrees" (high activation) pass through; dimensions where the memory is uncertain (low activation) get suppressed.

**Purpose:** The memory acts as a **relevance gate**. If the memory recognizes something in $y_t$ (because it was recently written), that signal is amplified. This creates a feedback loop: well-memorized information gets stronger representation.

**Code** — `titans/mac_layer.py:142-143`:
```python
mem_gate = self.memory.read(y_t.detach())  # M*_t(y_t)
o_t = y_t * mem_gate                       # element-wise gating
```

### 7.5 Complete MAC Per-Turn Algorithm

$$\boxed{ \begin{aligned} &\textbf{Input: } \text{prompt (text)},\; M_{t-1},\; P = \{p_1,\ldots,p_{N_p}\} \\ &\textbf{Output: } \text{response (text)},\; M_t \\ \\ &1.\; S^{(t)} = \text{Encoder}(\text{prompt}) \in \mathbb{R}^d \\ &2.\; q_t = W_Q S^{(t)} + b_Q,\;\; h_t = M_{t-1}^*(q_t) \\ &3.\; \tilde{S}^{(t)} = [p_1,\ldots,p_{N_p} \;\|\; h_t \;\|\; S^{(t)}] \in \mathbb{R}^{(N_p+2) \times d} \\ &4.\; y = \tilde{S}^{(t)} + W_{\text{out}} \cdot \text{MHA}\!\big(\text{LN}(\tilde{S}^{(t)})\big) \\ &5.\; y_t = y[-1] \in \mathbb{R}^d \\ &6.\; M_t = \text{TitansUpdate}(M_{t-1},\; y_t) \\ &7.\; o_t = y_t \odot M_t^*(y_t) \\ &8.\; \text{texts} = \text{NearestNeighbor}(o_t) \\ &9.\; \text{enriched} = [\text{persistent\_text}] \,+\, [\text{texts}] \,+\, [\text{prompt}] \\ &10.\; \text{response} = \text{LLM}(\text{enriched}) \end{aligned} }$$

---

## 8. Text Encoding Pipeline

The encoder transforms raw text into an L2-normalized $d$-dimensional vector:

$$\text{text} \xrightarrow{\text{Tokenizer}} \text{token\_ids} \in \mathbb{Z}^L$$

$$\xrightarrow{\text{BERT}} H \in \mathbb{R}^{L \times d_{\text{bert}}}$$

$$\xrightarrow{\text{Pool}} h \in \mathbb{R}^{d_{\text{bert}}}$$

$$\xrightarrow{W_{\text{proj}}} p = W_{\text{proj}} h + b_{\text{proj}} \in \mathbb{R}^d$$

$$\xrightarrow{\text{Normalize}} x = \frac{p}{\|p\|_2} \in \mathbb{R}^d$$

**Variable roles:**
- $\text{token\_ids} \in \mathbb{Z}^L$: Integer sequence of sub-word token IDs from the BERT tokenizer. $L \leq 512$.
- $H \in \mathbb{R}^{L \times d_{\text{bert}}}$: Contextual embeddings from BERT. Each of the $L$ tokens has a $d_{\text{bert}}$-dimensional representation that encodes its meaning in context.
- $h \in \mathbb{R}^{d_{\text{bert}}}$: Pooled single-vector representation of the entire text.
- $p \in \mathbb{R}^d$: Projected down to the memory dimension $d$.
- $x \in \mathbb{R}^d$: L2-normalized. All vectors live on the unit hypersphere $\|x\| = 1$.

**Pooling strategies:**

$$\text{Mean pooling:} \quad h = \frac{\sum_{i:\, \text{mask}_i = 1} H_i}{\sum_{i} \text{mask}_i}$$

$$\text{CLS pooling:} \quad h = H_0 \quad \text{(the [CLS] token embedding)}$$

$$\text{Max pooling:} \quad h_j = \max_{i:\, \text{mask}_i = 1} H_{i,j} \quad \text{(element-wise max over non-padding)}$$

**Variable roles:**
- $\text{mask}_i \in \{0, 1\}$: The attention mask from the tokenizer. $1$ for real tokens, $0$ for padding tokens.
- $H_i \in \mathbb{R}^{d_{\text{bert}}}$: The contextual embedding of the $i$-th token.

**Code** — `titans/text_encoder.py:54-101`

---

## 9. Text Retrieval — Nearest-Neighbor Decoder

The decoder stores $(\text{vector}, \text{text})$ pairs and retrieves the top-$K_{\text{top}}$ most similar texts.

**Cosine similarity** between query $q$ and stored vector $v_i$ (both L2-normalized):

$$\text{sim}(q, v_i) = \frac{q \cdot v_i}{\|q\| \cdot \|v_i\|} = q^\top v_i$$

The simplification to a dot product holds because the encoder always L2-normalizes its output ($\|q\| = \|v_i\| = 1$).

**Retrieval algorithm:**

$$\text{sims} = \begin{bmatrix} q^\top v_1 \\ q^\top v_2 \\ \vdots \\ q^\top v_n \end{bmatrix} \in \mathbb{R}^n$$

$$\text{indices} = \text{argsort}(-\text{sims})\big[1 : K_{\text{top}}\big]$$

$$\text{results} = \big\{ \text{text}_i \;\big|\; i \in \text{indices},\; \text{sims}_i > \tau \big\}$$

**Variable roles:**
- $q \in \mathbb{R}^d$: The query vector (current prompt's embedding).
- $v_i \in \mathbb{R}^d$: The $i$-th stored embedding from a previous conversation turn.
- $n$: Total number of stored entries.
- $\text{sims} \in \mathbb{R}^n$: Cosine similarities to all $n$ stored entries.
- $K_{\text{top}} = 3$: Maximum number of results to return.
- $\tau = 0.05$: Minimum similarity threshold. Filters out irrelevant matches.

**Code** — `titans/text_decoder.py:27-44`:
```python
mat = torch.stack(self._vectors)          # (n, d)
sims = torch.mv(mat, query)              # (n,) — dot products
values, indices = torch.topk(sims, k)    # top-k highest similarities
results = [text for idx, sim in zip(indices, values) if sim > threshold]
```

---

## 10. Persistent Memory (§3.3)

Two types exist in the system:

### 10a. Text Persistent Memory

Fixed text strings prepended to every LLM prompt:

$$\text{enriched} = \underbrace{[\langle\text{SOS}\rangle\; \text{token}_1 \;\langle\text{EOS}\rangle]}_{\text{persistent context}} \;\|\; \underbrace{[\text{Memory 1}, \ldots]}_{\text{retrieved memories}} \;\|\; \underbrace{[\text{User: prompt}]}_{\text{current input}}$$

**Variable roles:**
- $\langle\text{SOS}\rangle$, $\langle\text{EOS}\rangle$: Delimiter tokens marking persistent context boundaries.
- $\text{token}_i$: A human-written instruction string (e.g., "You are a helpful assistant").

These are **NOT learned** — they are manually configured strings.

### 10b. Vector Persistent Memory (MAC only)

Learnable parameter vectors that participate in attention:

$$P = \{p_1, p_2, \ldots, p_{N_p}\} \quad \in \mathbb{R}^{N_p \times d}$$

Initialized as:

$$p_i \sim \mathcal{N}(0,\; 0.02^2 \cdot I_d) \quad \forall\, i = 1, \ldots, N_p$$

**Variable roles:**
- $p_i \in \mathbb{R}^d$: The $i$-th persistent vector. A learnable `nn.Parameter` that encodes task-level knowledge in a distributed vector representation.
- $N_p = 4$: Number of persistent vectors. More vectors = more capacity for task knowledge, but increases the attention sequence length.
- Updated by the outer-loop optimizer during training. Fixed at inference time.
- **Input-independent** — the same vectors are used for every prompt.

**Key difference:** Text persistent memory affects the LLM through prompt text. Vector persistent memory affects the system through the attention computation (Section 7.2).

**Code** — `titans/attention.py:117-131`:
```python
class PersistentMemoryVectors(nn.Module):
    def __init__(self, num_vectors, d_model):
        self.vectors = nn.Parameter(torch.randn(num_vectors, d_model) * 0.02)

    def forward(self, batch_size):
        return self.vectors.unsqueeze(0).expand(batch_size, -1, -1)
```

---

## 11. Hyperparameter Summary

### 11.1 Inner Loop (Memory Update at Inference)

| Parameter | Symbol | Default | Shape | Effect |
|-----------|--------|---------|-------|--------|
| Inner-loop LR | $\theta$ | 0.01 | scalar | How fast MLP memorizes new associations. Higher = faster learning, more interference. |
| Momentum decay | $\eta$ | 0.9 | scalar | How long surprise persists. $\eta^{10} = 0.35$ → signal halves in ~7 steps. |
| Forgetting gate | $\alpha$ | 0.02 | scalar | Rate of old memory erasure. $(1-\alpha)^{50} = 0.36$ → memory halves in ~34 steps. |
| MLP layers | $N_l$ | 2 | integer | Memory depth. More layers = more expressive but harder to optimize. |
| MLP hidden dim | $d_h$ | 128 | integer | Width of MLP layers. Usually set equal to $d$. |

### 11.2 Outer Loop (Meta-Learning Training)

| Parameter | Symbol | Default | Shape | Effect |
|-----------|--------|---------|-------|--------|
| Outer-loop LR | $\lambda$ | $10^{-4}$ | scalar | Learning rate for $W_K, W_V, W_Q, W_{\text{proj}}$. |
| Epochs | $E$ | 10 | integer | Full passes over training data. |
| Episode length | $N$ | 8 | integer | Texts per episode. Must be $\leq$ |training texts|. |
| Eval ratio | $r_{\text{eval}}$ | 0.25 | scalar | Fraction reserved for eval. $N_w = N(1-r)$, $N_e = Nr$. |
| Gradient clip | $c$ | 1.0 | scalar | Max gradient L2 norm. 0 = no clipping. |
| Weight decay | $w$ | 0.0 | scalar | AdamW L2 regularization on outer params. |

### 11.3 Architecture

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Model dimension | $d$ | 128 | Size of all vectors, keys, values, queries. |
| Top-K retrieval | $K_{\text{top}}$ | 3 | Number of text snippets retrieved per query. |
| Similarity threshold | $\tau$ | 0.05 | Minimum cosine similarity for retrieval. |
| Attention heads | $H$ | 4 | MAC only. Parallel attention heads. |
| Persistent vectors | $N_p$ | 4 | MAC only. Learnable task-knowledge vectors. |
| Chunk size | $b$ | 16 | Parallel memory. Tokens per parallel chunk. |
| BERT model | — | bert-base-uncased | Pretrained encoder backbone ($d_{\text{bert}} = 768$). |
| Pooling strategy | 
— | mean | How BERT token embeddings are aggregated into one vector. |

---

## 12. Code-to-Math Reference Table

| Math Expression | Variable Roles | Code Location |
|----------------|----------------|---------------|
| $k_t = W_K x_t + b_K$ | $x_t$: input, $k_t$: key for writing | `neural_memory.py:98-100` — `project_key()` |
| $v_t = W_V x_t + b_V$ | $x_t$: input, $v_t$: value to store | `neural_memory.py:102-104` — `project_value()` |
| $q_t = W_Q x_t + b_Q$ | $x_t$: input, $q_t$: query for reading | `neural_memory.py:106-108` — `project_query()` |
| $M(x) = W_2 \sigma(W_1 x + b_1) + b_2$ | $W_i, b_i$: MLP weights = memory, $\sigma$: SiLU | `neural_memory.py:171-186` — `_mlp_forward()` |
| $\ell_t = \|M_{t-1}(k_t) - v_t\|^2$ | $\ell_t$: surprise scalar, $\hat{v}_t$: MLP prediction | `neural_memory.py:264-265` — `_write_training()` |
| $g_t = \nabla_{\theta_M} \ell_t$ | $g_t$: gradient collection, $\theta_M$: MLP params | `neural_memory.py:270-274` — `autograd.grad()` |
| $S_t = \eta S_{t-1} - \theta g_t$ | $S_t$: momentum, $\eta$: decay, $\theta$: inner LR | `neural_memory.py:281-282` |
| $M_t = (1-\alpha) M_{t-1} + S_t$ | $\alpha$: forget rate, $S_t$: momentum signal | `neural_memory.py:283` |
| $y_t = M_t^*(q_t)$ | $y_t$: read output, no weight update | `neural_memory.py:190-205` — `read()` |
| $\mathcal{L} = \sum \ell^{\text{write}} + \sum \ell^{\text{eval}}$ | $\mathcal{L}$: total outer loss | `trainer.py:165` |
| $\Phi \leftarrow \Phi - \lambda \nabla_\Phi \mathcal{L}$ | $\Phi$: outer params, $\lambda$: outer LR | `trainer.py:177` — `optimizer.step()` |
| $\beta_t = \prod_{j=1}^t (1-\alpha_j)$ | $\beta_t$: cumulative decay factor | `parallel_memory.py:128-141` |
| $S_t = a_t S_{t-1} + b_t$ | $a_t = \eta$: gate, $b_t = -\theta g_t$: input | `parallel_memory.py:37-125` |
| $\tilde{S} = [P \| h_t \| S^{(t)}]$ | $P$: persistent vecs, $h_t$: mem read, $S^{(t)}$: input | `mac_layer.py:123` |
| $A = \text{softmax}(Q K^\top / \sqrt{d_h}) V$ | $Q,K,V$: attention projections, $d_h$: head dim | `attention.py:63-103` |
| $o_t = y_t \odot M_t^*(y_t)$ | $y_t$: attn output, $\odot$: element-wise gate | `mac_layer.py:142-143` |
| $x = \frac{W_{\text{proj}} \cdot \text{Pool}(\text{BERT}(\text{text}))}{\| \cdot \|}$ | $x$: L2-normalized embedding | `text_encoder.py:54-101` |
| $\text{sim}(q, v_i) = q^\top v_i$ | $q$: query, $v_i$: stored vec (both unit norm) | `text_decoder.py:27-44` |

---

*Generated from the Titans codebase at `/Users/shashi.b/Documents/titan/`*

*Paper reference: Behrouz et al., "Titans: Learning to Memorize at Test Time" (arXiv:2501.00663)*
