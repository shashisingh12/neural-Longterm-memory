"""
Titans-Inspired External Memory Layer
======================================
Drop-in wrapper around ANY existing LLM.
Zero extra dependencies — only NumPy required.

Flow:
    prompt
      ↓
  MemoryLayer
      ↓  (retrieves relevant past context)
  enriched_prompt  =  [persistent_context]  +  [retrieved memories]  +  [prompt]
      ↓
  your_LLM(enriched_prompt)
      ↓
  response  →  written back into memory

Architecture (Behrouz et al. 2501.00663 §3):
  - Neural MLP whose weights ARE the memory
  - Surprise  = gradient magnitude of associative loss
  - Momentum  = running average of surprises (so post-surprise tokens are not lost)
  - Forgetting = weight decay gate alpha_t in [0, 1]
  - Persistent tokens  = task-level knowledge, always prepended
"""

import numpy as np
import json
import hashlib
import time
from typing import Callable, Optional


# ─────────────────────────────────────────────────────────
#  ACTIVATION  (SiLU, used in the paper)
# ─────────────────────────────────────────────────────────

def silu(x):
    return x / (1 + np.exp(-x))

def silu_grad(x):
    s = 1 / (1 + np.exp(-x))
    return s + x * s * (1 - s)


# ─────────────────────────────────────────────────────────
#  1. NEURAL MEMORY MLP
#     Weights of this network = compressed long-term memory
# ─────────────────────────────────────────────────────────

class NeuralMemoryMLP:
    """
    Paper Section 3.1 update rule:

        S_t = eta_t * S_{t-1}  -  theta_t * grad(loss)    (momentum over surprise)
        M_t = (1 - alpha_t) * M_{t-1}  +  S_t             (forgetting gate)
        loss = || M(k_t) - v_t ||^2                        (associative memory loss)

    READ  (Eq. 15): y_t = M*(q_t)   -- inference, no weight update
    WRITE:          one step of the above gradient update rule
    """

    def __init__(self, d_model=64, n_layers=2, seed=42):
        rng = np.random.default_rng(seed)
        self.d_model  = d_model
        self.n_layers = n_layers

        self.W = []
        self.b = []
        for _ in range(n_layers):
            scale = np.sqrt(2.0 / d_model)
            self.W.append(rng.normal(0, scale, (d_model, d_model)))
            self.b.append(np.zeros(d_model))

        # Momentum buffers S_t
        self.mW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]

        # Hyper-parameters (scalar; paper makes these data-dependent)
        self.lr             = 0.01   # theta_t : inner-loop learning rate
        self.momentum_decay = 0.90   # eta_t   : surprise momentum decay
        self.forget         = 0.02   # alpha_t : weight decay / forgetting gate

    def _forward(self, x):
        h = x.copy()
        cache = []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W.T + b
            cache.append((h.copy(), z.copy()))
            if i < self.n_layers - 1:
                h = silu(z)
            else:
                h = z          # last layer is linear
        return h, cache

    def _backward(self, cache, output, target):
        delta = 2 * (output - target)
        gW = [None] * self.n_layers
        gb = [None] * self.n_layers
        for i in reversed(range(self.n_layers)):
            h_in, z = cache[i]
            if i < self.n_layers - 1:
                delta = delta * silu_grad(z)
            gW[i] = np.outer(delta, h_in)
            gb[i] = delta.copy()
            delta  = delta @ self.W[i]
        return gW, gb

    def write(self, key, value):
        """Update memory weights given one (key, value) association."""
        out, cache = self._forward(key)
        gW, gb     = self._backward(cache, out, value)
        for i in range(self.n_layers):
            # Eq. 14: S_t = eta * S_{t-1} - theta * grad
            self.mW[i] = self.momentum_decay * self.mW[i] - self.lr * gW[i]
            self.mb[i] = self.momentum_decay * self.mb[i] - self.lr * gb[i]
            # Eq. 13: M_t = (1 - alpha) * M_{t-1} + S_t
            self.W[i]  = (1 - self.forget) * self.W[i] + self.mW[i]
            self.b[i]  = (1 - self.forget) * self.b[i] + self.mb[i]

    def read(self, query):
        """Retrieve without updating weights — paper Eq. 15: M*(q_t)"""
        out, _ = self._forward(query)
        return out

    def state_dict(self):
        return {
            "W":  [w.tolist() for w in self.W],
            "b":  [b.tolist() for b in self.b],
            "mW": [m.tolist() for m in self.mW],
            "mb": [m.tolist() for m in self.mb],
        }

    def load_state_dict(self, d):
        self.W  = [np.array(w) for w in d["W"]]
        self.b  = [np.array(b) for b in d["b"]]
        self.mW = [np.array(m) for m in d["mW"]]
        self.mb = [np.array(m) for m in d["mb"]]


# ─────────────────────────────────────────────────────────
#  2. TEXT <-> VECTOR BRIDGE
# ─────────────────────────────────────────────────────────

class TextEncoder:
    """
    Lightweight deterministic text -> d_model vector.
    Uses character n-gram hashing + random projection.

    To upgrade: replace .encode() with any sentence-transformer, e.g.
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        def encode(self, text): return model.encode(text)
    """

    def __init__(self, d_model=64, seed=0):
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.proj    = rng.normal(0, 1, (d_model, 256)) / np.sqrt(256)

    def encode(self, text):
        vec  = np.zeros(256)
        text = text.lower()
        for i in range(len(text)):
            for n in (1, 2, 3):
                gram = text[i:i+n]
                h = int(hashlib.md5(gram.encode()).hexdigest(), 16) % 256
                vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        out = self.proj @ vec
        out_norm = np.linalg.norm(out)
        if out_norm > 0:
            out /= out_norm
        return out


class TextDecoder:
    """Nearest-neighbour cosine lookup: vector -> past text snippets."""

    def __init__(self):
        self._vecs  = []
        self._texts = []

    def register(self, vec, text):
        self._vecs.append(vec.copy())
        self._texts.append(text)

    def decode(self, vec, top_k=3):
        if not self._vecs:
            return []
        mat  = np.stack(self._vecs)       # (N, d)
        sims = mat @ vec                  # cosine similarities (all normalised)
        idx  = np.argsort(-sims)[:top_k]
        return [self._texts[i] for i in idx if sims[i] > 0.05]

    def state_dict(self):
        return {"vecs":  [v.tolist() for v in self._vecs],
                "texts": self._texts}

    def load_state_dict(self, d):
        self._vecs  = [np.array(v) for v in d["vecs"]]
        self._texts = d["texts"]


# ─────────────────────────────────────────────────────────
#  3. PERSISTENT MEMORY  (paper Section 3.3)
# ─────────────────────────────────────────────────────────

class PersistentMemory:
    """
    Fixed text tokens prepended to every prompt.
    Encodes stable task knowledge: persona, rules, domain context.
    Paper: P = [p1, p2, ..., p_Np]  -- learnable but input-independent.
    """

    def __init__(self):
        self._tokens = []

    def add(self, text):
        self._tokens.append(text)

    def render(self):
        if not self._tokens:
            return ""
        return "\n".join(f"[PERSISTENT] {t}" for t in self._tokens)


# ─────────────────────────────────────────────────────────
#  4. TITANS MEMORY LAYER  -- the main wrapper
# ─────────────────────────────────────────────────────────

class TitansMemoryLayer:
    """
    Wraps ANY existing LLM with external neural long-term memory.

    Quick Start
    -----------
        from titans_memory import TitansMemoryLayer

        memory = TitansMemoryLayer()
        memory.persistent.add("You are a helpful assistant.")

        def my_llm(prompt: str) -> str:
            return your_model.generate(prompt)   # your existing call

        response = memory.run("What is Python?", my_llm)

    Per-Turn Flow
    -------------
        1.  encode(prompt)         -> query vector q
        2.  memory.read(q)         -> retrieved vector  (no weight update)
        3.  decoder.decode(vec)    -> top-k past text snippets
        4.  build enriched prompt:
                [persistent tokens]
                [retrieved memory snippets]
                [current user prompt]
        5.  call your LLM with enriched prompt
        6.  write (prompt + response) into memory
                k = encode("Q: ... A: ...")
                v = encode(response)
                memory.write(k, v)
    """

    def __init__(self, d_model=64, memory_layers=2, top_k=3, verbose=False):
        self.encoder    = TextEncoder(d_model)
        self.memory     = NeuralMemoryMLP(d_model, memory_layers)
        self.decoder    = TextDecoder()
        self.persistent = PersistentMemory()
        self.top_k      = top_k
        self.verbose    = verbose
        self._turn      = 0

    # ── MAIN ENTRY POINT ─────────────────────────────────────────────────
    def run(self, prompt, llm_fn):
        """
        Parameters
        ----------
        prompt  : str  -- raw user input
        llm_fn  : callable str -> str  -- your existing LLM

        Returns
        -------
        str  -- LLM response
        """
        self._turn += 1
        t0 = time.time()

        # 1. Encode prompt
        q_vec = self.encoder.encode(prompt)

        # 2. Read from neural memory (no weight update)
        retrieved_vec   = self.memory.read(q_vec)
        retrieved_texts = self.decoder.decode(retrieved_vec, self.top_k)

        if self.verbose:
            print(f"\n[Memory | Turn {self._turn}]")
            print(f"  Retrieved {len(retrieved_texts)} snippet(s)")
            for i, t in enumerate(retrieved_texts, 1):
                print(f"    [{i}] {t[:90]}")

        # 3. Build enriched prompt
        enriched = self._build_prompt(prompt, retrieved_texts)

        if self.verbose:
            print(f"  Enriched prompt: +{len(enriched) - len(prompt)} chars added")

        # 4. Call your LLM
        response = llm_fn(enriched)

        # 5. Write into memory
        combined = f"Q: {prompt}\nA: {response}"
        k_vec    = self.encoder.encode(combined)
        v_vec    = self.encoder.encode(response)
        self.memory.write(k_vec, v_vec)
        self.decoder.register(k_vec, combined)

        if self.verbose:
            print(f"  Written to memory | elapsed: {time.time()-t0:.3f}s")

        return response

    def _build_prompt(self, prompt, memories):
        parts = []

        p = self.persistent.render()
        if p:
            parts.append(p)

        if memories:
            parts.append("=== RELEVANT LONG-TERM MEMORY ===")
            for i, m in enumerate(memories, 1):
                parts.append(f"[Memory {i}]\n{m}")
            parts.append("=== END MEMORY ===")

        parts.append(f"User: {prompt}")
        return "\n\n".join(parts)

    # ── SAVE / LOAD ───────────────────────────────────────────────────────
    def save(self, path):
        state = {
            "memory":     self.memory.state_dict(),
            "decoder":    self.decoder.state_dict(),
            "persistent": self.persistent._tokens,
            "turn":       self._turn,
            "d_model":    self.memory.d_model,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"[Memory] Saved -> {path}")

    def load(self, path):
        with open(path) as f:
            state = json.load(f)
        self.memory.load_state_dict(state["memory"])
        self.decoder.load_state_dict(state["decoder"])
        self.persistent._tokens = state.get("persistent", [])
        self._turn              = state.get("turn", 0)
        print(f"[Memory] Loaded <- {path}  (turn {self._turn})")

    def stats(self):
        n_params = sum(np.prod(w.shape) for w in self.memory.W)
        n_params += sum(b.shape[0] for b in self.memory.b)
        return {
            "turns_seen":     self._turn,
            "memory_entries": len(self.decoder._texts),
            "memory_params":  int(n_params),
            "d_model":        self.memory.d_model,
            "mlp_layers":     self.memory.n_layers,
        }


# ─────────────────────────────────────────────────────────
#  DEMO  --  replace my_llm() with your real model
# ─────────────────────────────────────────────────────────

def demo():
    print("=" * 60)
    print("  Titans External Memory Layer  |  Demo")
    print("=" * 60)

    # ── REPLACE THIS with your real LLM ─────────────────────────
    def my_llm(prompt):
        """
        Plug your LLM in here. Examples:

        # OpenAI
        from openai import OpenAI
        client = OpenAI()
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"user","content": prompt}])
        return r.choices[0].message.content

        # Ollama (local)
        import requests
        r = requests.post("http://localhost:11434/api/generate",
            json={"model":"llama3","prompt":prompt,"stream":False})
        return r.json()["response"]

        # HuggingFace
        from transformers import pipeline
        gen = pipeline("text-generation", model="gpt2")
        return gen(prompt, max_new_tokens=200)[0]["generated_text"]
        """
        # Fake LLM for demo
        pl = prompt.lower()
        if "python" in pl:
            return "Python is a high-level, readable programming language, popular in AI/ML."
        if "titans" in pl:
            return "Titans is a neural architecture using a deep MLP as long-term memory, updated by surprise-driven gradients."
        if "memory" in pl and "titans" not in pl:
            return "Computer memory stores data; RAM is fast and volatile, disk is slow and persistent."
        if any(w in pl for w in ["earlier", "mentioned", "recall", "discussed", "past"]):
            return "Based on the context provided, we discussed Python and the Titans architecture."
        return f"I understand your question about: {prompt[:60]}"
    # ─────────────────────────────────────────────────────────────

    memory = TitansMemoryLayer(d_model=64, memory_layers=2, top_k=2, verbose=True)

    # Persistent task context (injected into every prompt)
    memory.persistent.add("You are a helpful technical assistant.")
    memory.persistent.add("Domain: AI, software engineering, computer science.")

    questions = [
        "What is Python?",
        "Explain the Titans architecture briefly.",
        "How does long-term memory work in computers?",
        "What programming language did we talk about earlier?",   # recall
        "Which architecture uses gradient-based memory updates?", # recall
    ]

    for q in questions:
        print(f"\n{'─'*55}")
        print(f"USER: {q}")
        r = memory.run(q, my_llm)
        print(f"LLM : {r}")

    print(f"\n{'='*55}")
    print("Memory Stats:")
    print(json.dumps(memory.stats(), indent=2))

    # Save and restore
    memory.save("titans_memory.json")
    print("\n--- Restoring from disk ---")
    m2 = TitansMemoryLayer(d_model=64, memory_layers=2, top_k=2)
    m2.load("/tmp/titans_memory.json")
    r  = m2.run("What topics have we covered so far?", my_llm)
    print(f"LLM: {r}")


if __name__ == "__main__":
    demo()
