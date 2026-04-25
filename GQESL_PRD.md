# Product Requirements Document (PRD)
## GeoQuantum Emergent Semantic Language (GQESL)
### Private Agent-to-Agent Communication Protocol

**Version:** 1.0  
**Date:** April 2026  
**Status:** Draft  
**Author:** Abdulaziz Alqarzaie  

---

## 1. Executive Summary

GQESL (GeoQuantum Emergent Semantic Language) is a novel agent-to-agent (A2A) communication protocol that enables two or more AI agents to exchange task intents and coordination signals through a private, eavesdrop-resistant semantic channel — without ever transmitting natural language or interpretable structured data on the wire.

The system replaces conventional JSON-RPC or natural-language message passing with a mathematically grounded pipeline: agent intents are expressed as raw tensors, transformed through session-private non-linear projections, quantized into indices of a secret codebook, and transmitted as compact 38-byte packets. The receiving agent reconstructs, denoises, and collapses the signal back into an actionable task using geospatially-inspired topological operators (RCC-8) for coordination logic.

The result is a communication protocol that is:
- **Semantically private** — wire payloads carry no interpretable meaning without the session secret
- **Coordination-aware** — topological relations drive task allocation, not just message routing
- **Language-free** — natural language is absent from the entire pipeline
- **Lightweight** — the full encode-transmit-collapse cycle runs in under 5ms on CPU
- **Novel** — no published system combines runtime tensor-based emergent communication with geospatial topology operators for LLM agent coordination

---

## 2. Problem Statement

### 2.1 The Gap in Current A2A Protocols

Modern multi-agent frameworks — including Google's Agent2Agent (A2A) protocol, IBM's Agent Communication Protocol (ACP), LangGraph message passing, and Semantic Kernel integrations — share a fundamental architectural weakness: **they communicate in plaintext or structured schema formats that are fully interpretable by any party with access to the transport layer**.

These systems rely on TLS for transport security, but TLS protects only the channel, not the semantic content. Any infrastructure provider, proxy, logging middleware, or compromised node with access to the decrypted traffic can read the exact task instructions, data references, and coordination logic being exchanged between agents.

This creates a critical vulnerability in:
- Multi-agent deployments on shared cloud infrastructure
- Autonomous agents operating in adversarial or untrusted environments
- Proprietary AI pipelines where task logic is a business secret
- Multi-organization agent collaboration where full trust cannot be assumed

### 2.2 Why Existing Solutions Are Insufficient

Standard encryption (TLS, E2E encryption) protects transmission but leaves the semantic layer exposed to:
- Infrastructure providers who terminate TLS
- Logging and observability systems that capture decrypted traffic
- Any agent node that is compromised mid-session
- Side-channel inference from message structure, size, and timing patterns

Steganographic approaches hide the existence of communication but not its content once detected. Homomorphic encryption is computationally prohibitive for real-time agent coordination. None of the existing approaches provide **semantic privacy** — the property that even a successful decryption yields no useful information about the intent being communicated.

### 2.3 The Core Insight

The solution is not stronger encryption — it is **removing interpretable content from the wire entirely**. If the transmitted payload is a discrete codebook index (an integer), and the codebook is a private artifact that never leaves the agents' memory, then there is nothing to decrypt. The privacy guarantee comes from the inaccessibility of the grounding artifact, not from the strength of a cipher.

---

## 3. Project Goals

### 3.1 Primary Goals

- Build a working Python implementation of the GQESL protocol as a LangGraph-based multi-agent system
- Demonstrate measurable semantic privacy: wire payloads must yield no meaningful information to a passive eavesdropper
- Demonstrate task coordination accuracy: RCC-8 relations must correctly drive task allocation in at least 80% of test cases
- Achieve end-to-end latency under 10ms for the encode-transmit-collapse cycle (excluding LLM execution time)

### 3.2 Secondary Goals

- Publish a technical report or paper describing the protocol, its security properties, and evaluation results
- Provide a Streamlit dashboard for live visualization of the semantic ledger, compression ratio, and drift scores
- Establish a baseline comparison between GQESL and standard LangGraph message passing on task accuracy, compression ratio, and semantic privacy metrics

### 3.3 Non-Goals

- This project does not aim to replace TLS or provide information-theoretic security guarantees (the security model is computational, not unconditional)
- This project does not involve actual quantum computing hardware or quantum circuits
- This project does not implement a full production-grade deployment pipeline (devops, containerization, monitoring)
- This project does not target real-time streaming data or sub-millisecond latency requirements

---

## 4. System Overview

### 4.1 What the System Does

GQESL enables two AI agents (Agent A and Agent B) to coordinate on tasks by exchanging messages that are:
1. Constructed from structured tensor representations of agent intent (no natural language)
2. Transformed into a private semantic space using session-specific secret matrices
3. Compressed into a single integer index via a secret codebook
4. Transmitted as a 38-byte signed packet
5. Reconstructed, verified, and collapsed back into a concrete action by the receiver

The topological relationship between the sender's intent and the receiver's existing knowledge is encoded as an RCC-8 spatial operator (EQ, PO, EC, DC), which is transmitted alongside the codebook index and serves as the coordination instruction — telling the receiver how to split, merge, or delegate the associated task.

### 4.2 Key Concepts

**Intent Tensor:** A fixed-dimension (384-element) float32 vector representing agent intent. It is constructed from a structured internal state (task type, entity type, output format, priority, source reference) and never from natural language.

**Private Projection:** A two-layer non-linear transformation (tanh activation) applied to the salted intent tensor using secret weight matrices W1 and W2, derived deterministically from the ECDH shared secret via HKDF. This transformation maps the tensor into a private semantic space that is inaccessible without the session secret.

**Secret Codebook:** A set of 4096 random 384-dimensional vectors, generated from the shared secret, held privately by both agents. The projected tensor is quantized to the nearest codebook entry, and only the index (0–4095) is transmitted.

**RCC-8 Relation:** One of four topological operators (EQ, PO, EC, DC) computed by comparing the sender's projected tensor to the closest entry in the shared semantic ledger. The relation encodes the spatial relationship between sender and receiver knowledge and drives task allocation strategy.

**Semantic Ledger:** A per-session LanceDB vector store holding the concept vectors both agents have registered. It enables concept lookup, drift detection, and RCC-8 computation.

**Shared Secret:** A 32-byte value computed independently by both agents via ECDH key exchange (X25519). Never transmitted. The root from which all session secrets are derived.

---

## 5. Architecture

### 5.1 High-Level Components

The system consists of seven major components:

**1. Key Exchange Module**  
Handles the one-time ECDH handshake between agents at session initialization. Produces the shared secret and derives all session keys (W1, W2, codebook seed, HMAC key, salt seed) via HKDF.

**2. Intent Tensor Builder**  
Constructs the 384-dimensional intent tensor from a structured Python dataclass (AgentIntent). Defines the concept basis vectors for all supported task types, entity types, output formats, priorities, and source references. No natural language input is accepted.

**3. Semantic Encoder**  
Applies the two-step pipeline: salt injection → non-linear projection (W1, tanh, W2) → codebook quantization. Produces the (idx, relation, counter, hmac) wire packet.

**4. RCC-8 Engine**  
Computes the topological relation between the current projected tensor and the semantic ledger. Returns one of four relations and the associated coordination strategy.

**5. Semantic Ledger**  
A LanceDB-backed vector store maintaining the session's registered concept vectors. Supports insert, nearest-neighbor search, drift scoring, and session-bounded queries.

**6. Semantic Decoder**  
On the receiver side: HMAC verification → codebook lookup → desalting → inverse projection → collapse → ledger decode → action dispatch.

**7. Drift Monitor**  
A background process that periodically samples recent message pairs, computes embedding variance per concept, and triggers re-negotiation when drift exceeds a configurable threshold.

### 5.2 LangGraph Integration

The system is implemented as a LangGraph StateGraph with the following nodes:

- **key_exchange_node:** Runs once at session start; populates shared session state with all derived secrets
- **intent_builder_node:** Accepts agent internal state; outputs intent tensor
- **encode_node:** Applies salt, projection, quantization; outputs wire packet
- **transmit_node:** Handles the actual message send (in-process queue or network socket)
- **receive_node:** Accepts wire packet; verifies HMAC and counter freshness
- **decode_node:** Codebook lookup, desalting, inverse projection, collapse
- **dispatch_node:** Reads RCC-8 strategy; routes task to appropriate agent tool
- **drift_monitor_node:** Runs as a conditional edge; triggers re-negotiation if drift detected

### 5.3 Data Flow Summary

```
[Agent A Internal State]
        |
        v
[Intent Tensor (384 floats)] — never leaves agent memory
        |
[Salt Injection] — per-message, derived from shared_secret + counter
        |
[Non-Linear Projection] — W1, tanh, W2 — session-private
        |
[RCC-8 Relation Compute] — against semantic ledger
        |
[Codebook Quantization] — 384 floats → 1 integer (0–4095)
        |
[HMAC Sign] — over (idx, relation, counter)
        |
[WIRE: 38 bytes] ——————————————————→ [Receive]
                                              |
                                     [HMAC Verify + Counter Check]
                                              |
                                     [Codebook Lookup] — integer → vector
                                              |
                                     [Desalt + Inverse Projection]
                                              |
                                     [Collapse] — vector → concept
                                     [RCC-8 Interpret] — strategy selection
                                              |
                                     [Ledger Decode → Action]
                                              |
                                     [Execute]
```

---

## 6. Security Model

### 6.1 Threat Model

The system is designed to resist the following adversaries:

**Passive Eavesdropper:** An attacker who can observe all wire traffic between agents but cannot modify it. This is the primary threat model. The system provides strong resistance: wire payloads are integers with no statistical correlation to intent content across sessions (due to per-message salting).

**Active Man-in-the-Middle:** An attacker who can intercept and modify wire packets. The HMAC over (idx, relation, counter) with a session-private key detects any modification. The monotonic counter prevents replay attacks.

**Known-Plaintext Attacker:** An attacker who observes wire packets and can correlate some of them with known agent behaviors. The non-linear projection (tanh) means the system of equations relating observed packets to the projection matrices is non-convex and cannot be solved by linear algebra. Key rotation every N messages limits the window of exposure.

**Infrastructure Provider:** A cloud host or proxy that terminates TLS and reads decrypted traffic. The system provides semantic privacy at this level: the host sees 38-byte packets containing integers and 2-character codes. Without the shared secret (which is never transmitted), these packets are meaningless.

### 6.2 Security Properties

| Property | Status | Mechanism |
|---|---|---|
| Semantic privacy | Strong | Codebook index carries no decodable meaning without secret codebook |
| Integrity | Strong | HMAC-SHA256 over full packet content |
| Replay resistance | Strong | Monotonic counter inside HMAC scope |
| Forward secrecy | Moderate | Session nonce rotation; past sessions unaffected by current key compromise |
| Known-plaintext resistance | Moderate | Non-linear projection; key rotation limits window |
| Side-channel resistance | Weak | Not addressed in v1; behavioral obfuscation is a future extension |

### 6.3 Explicit Non-Guarantees

- The system does not provide information-theoretic (unconditional) security
- A sufficiently powerful attacker with millions of known-plaintext pairs and knowledge of the encoder architecture may be able to recover W1/W2 via non-convex optimization
- The system does not hide the existence of communication between agents (traffic analysis can confirm agents are communicating)
- Compromise of either agent's process memory exposes the shared secret for that session

---

## 7. Functional Requirements

### 7.1 Key Exchange

- The system must implement X25519 ECDH key exchange between two agent instances
- The shared secret must be derived using HKDF-SHA256 with distinct salts for each derived key (W1, W2, codebook, HMAC key, salt seed)
- Key exchange must complete before any tensor messages are sent
- Session nonces must be generated per session and incorporated into all HKDF derivations for forward secrecy

### 7.2 Intent Tensor Construction

- The IntentTensor class must accept a structured AgentIntent dataclass as input (no string fields except optional debug labels)
- The AgentIntent dataclass must cover at minimum: task_type (enum), entity_type (enum), output_format (enum), priority (float), source_ref (bytes hash)
- The intent tensor must be constructed as a weighted sum of concept basis vectors from a predefined basis matrix
- The basis matrix must be session-private (derived from shared secret) in the hardened configuration, or shared public basis in the baseline configuration

### 7.3 Semantic Encoding

- The encoder must apply per-message salt injection before projection
- The salt must be deterministically derived from (shared_secret, message_counter) using a CSPRNG
- The projection must be non-linear: tanh(salted_tensor @ W1) @ W2
- Input normalization must be applied before tanh to prevent saturation (norm to 0.5 range)
- The encoder must quantize the projected tensor to the nearest entry in the secret codebook using cosine distance
- The encoder must compute the RCC-8 relation against the semantic ledger before quantization
- The wire packet must be exactly: {idx: int, r: str[2], c: int, h: bytes[32]}

### 7.4 RCC-8 Engine

- The engine must implement all four RCC-8 relations: EQ (sim > 0.95), PO (sim > 0.85), EC (sim > 0.60), DC (sim ≤ 0.60)
- Each relation must map to a distinct coordination strategy: EXACT_MATCH, SPLIT_EXECUTION, FULL_HANDOFF, NEGOTIATE_FIRST
- The engine must query the semantic ledger for the closest registered concept before computing the relation
- Threshold values must be configurable per session

### 7.5 Semantic Ledger

- The ledger must be backed by LanceDB with an in-memory or file-backed store
- The ledger must support: register(concept_vector, concept_id), search(query_vector, k=1), get_all(session_id), drift_score(concept_id)
- The ledger must be session-scoped: entries from different sessions must not be mixed
- Drift score must be computed as the cosine variance of the last K usage vectors for a given concept

### 7.6 Semantic Decoding

- The decoder must verify the HMAC before any other processing; failed verification must result in hard discard and alert
- The decoder must verify counter monotonicity; gaps or repeats must result in hard discard and alert
- The decoder must perform codebook lookup using the received index
- The decoder must apply desalting and approximate inverse projection
- The decoder must collapse the reconstructed tensor to the nearest concept in the semantic ledger
- The decoder must read the RCC-8 relation and select the corresponding coordination strategy
- The decoder must dispatch the action according to strategy and collapsed concept

### 7.7 Drift Monitor

- The drift monitor must run as a background process or conditional LangGraph node
- The monitor must compute drift scores for all concepts after every N messages (N configurable, default 20)
- If any concept's drift score exceeds the threshold (default 0.15), the monitor must trigger a re-negotiation round for that concept
- Re-negotiation must update the ledger entry with the new centroid vector

---

## 8. Non-Functional Requirements

### 8.1 Performance

- Encode-transmit-collapse latency (excluding LLM execution): under 10ms on CPU
- Codebook generation at session startup: under 500ms
- ECDH key exchange: under 100ms
- Ledger search (nearest neighbor over 1000 entries): under 5ms

### 8.2 Scalability

- The system must support sessions with up to 10,000 messages without ledger degradation
- The codebook must support up to 4096 entries (12-bit index space)
- The basis matrix must support up to 256 distinct concept dimensions

### 8.3 Reliability

- HMAC failures must be logged with full packet metadata (excluding secret material)
- Counter failures must trigger session re-initialization
- Codebook quantization error above 0.3 cosine distance must log a warning (potential concept not in codebook)
- All LanceDB operations must be wrapped in try/except with graceful degradation

### 8.4 Testability

- All cryptographic operations must be deterministically reproducible given a fixed seed (for testing only; production uses true randomness)
- The encode/decode pipeline must be roundtrip-testable: encode(intent) → decode(packet) must recover a vector within 0.05 cosine distance of the original
- The RCC-8 engine must have unit tests for all four relation boundaries with edge cases

---

## 9. Evaluation Plan

### 9.1 Metric 1 — Compression Ratio

**Definition:** (Token count of equivalent natural language message) / (Wire payload size in bytes)  
**Target:** > 20x compression after ledger warmup (after first 10 concepts registered)  
**Method:** Run 100 coordination tasks. For each task, record the natural language description length and the wire packet size. Report mean, median, and distribution.

### 9.2 Metric 2 — Task Allocation Accuracy

**Definition:** Fraction of tasks where the RCC-8 relation correctly predicted the optimal coordination strategy, as judged against a human-labeled ground truth  
**Target:** > 80% accuracy  
**Method:** Construct a 100-task benchmark with 25 tasks per RCC-8 class. Run GQESL. Compare predicted strategy to ground truth label. Report per-class and overall accuracy.

### 9.3 Metric 3 — Semantic Recovery Fidelity

**Definition:** Cosine similarity between original intent tensor and decoded tensor at Agent B  
**Target:** Mean cosine similarity > 0.92 across 1000 messages  
**Method:** Generate 1000 random intent tensors. Encode and decode each. Compute cosine similarity between original and recovered tensor. Report mean, min, and 5th percentile.

### 9.4 Metric 4 — Eavesdropper Information Gain

**Definition:** Mutual information between wire packet content and original intent, estimated via a trained classifier  
**Method:** Train a neural classifier on (wire packet → intent class) using 10,000 intercepted packets with known intents. Measure classification accuracy. Random (1/N) accuracy indicates zero information gain.  
**Target:** Classifier accuracy below 1.5x random baseline

### 9.5 Metric 5 — Semantic Drift Rate

**Definition:** Fraction of concepts that require re-negotiation over a 1000-message session  
**Target:** < 5% of concepts trigger re-negotiation  
**Method:** Run a 1000-message simulation. Count re-negotiation events. Report per-concept and overall drift rate.

---

## 10. Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| Agent Framework | LangGraph | Native support for stateful multi-agent graphs with shared checkpointer |
| LLM Backend | DeepSeek API | Low cost per call; sufficient capability for agent reasoning |
| Vector Store | LanceDB | Embedded (no server), fast ANN search, native numpy integration |
| Cryptography | cryptography (Python) | X25519 ECDH, HKDF, HMAC-SHA256 — all standard primitives |
| Numerical Ops | NumPy | All tensor operations, projections, codebook search |
| Encoder Model | sentence-transformers (all-MiniLM-L6-v2) | 384-dim embeddings; fast CPU inference; well-characterized |
| Dashboard | Streamlit | Rapid development; native Python; agent-friendly |
| Testing | pytest | Standard; supports parameterized and fixture-based tests |
| Environment | Python 3.11+, Google Colab or local Jupyter | Matches existing development environment |

---

## 11. Project Structure

```
gqesl/
├── core/
│   ├── key_exchange.py         # ECDH handshake, HKDF key derivation
│   ├── intent_tensor.py        # AgentIntent dataclass, tensor construction
│   ├── encoder.py              # Salt injection, projection, quantization
│   ├── decoder.py              # Codebook lookup, desalting, inverse projection, collapse
│   ├── rcc8.py                 # RCC-8 relation computation and strategy mapping
│   ├── codebook.py             # Codebook generation and nearest-neighbor search
│   └── ledger.py               # LanceDB-backed semantic ledger
│
├── agents/
│   ├── agent_a.py              # Sender agent LangGraph graph definition
│   ├── agent_b.py              # Receiver agent LangGraph graph definition
│   └── session.py              # Shared session state, counter management
│
├── monitor/
│   └── drift_monitor.py        # Drift detection and re-negotiation trigger
│
├── dashboard/
│   └── app.py                  # Streamlit dashboard for ledger and metrics
│
├── evaluation/
│   ├── benchmark.py            # Task benchmark construction and runner
│   ├── metrics.py              # All five evaluation metrics
│   └── baseline.py             # Comparison against standard LangGraph messaging
│
├── tests/
│   ├── test_key_exchange.py
│   ├── test_encoder.py
│   ├── test_decoder.py
│   ├── test_rcc8.py
│   ├── test_ledger.py
│   ├── test_roundtrip.py       # Full encode → decode roundtrip tests
│   └── test_security.py        # Eavesdropper classifier, known-plaintext tests
│
├── config.py                   # All thresholds, dimensions, defaults
├── requirements.txt
└── README.md
```

---

## 12. Development Phases

### Phase 1 — Cryptographic Foundation (Days 1–2)
Implement key_exchange.py and codebook.py. Establish the ECDH handshake, HKDF derivation of all session keys, and codebook generation. Write unit tests confirming that both agents independently produce identical W1, W2, codebook, and salt seed from the same shared secret.

### Phase 2 — Tensor Pipeline (Days 3–4)
Implement intent_tensor.py, encoder.py, and decoder.py. Build the AgentIntent dataclass, the concept basis matrix, and the full encode/decode pipeline. Write roundtrip tests confirming cosine recovery > 0.92.

### Phase 3 — RCC-8 and Ledger (Day 5)
Implement rcc8.py and ledger.py. Build the LanceDB-backed ledger with insert, search, and drift scoring. Implement the four RCC-8 relations and their strategy mappings. Unit test all four relation boundaries.

### Phase 4 — LangGraph Integration (Days 6–7)
Wire all components into LangGraph StateGraph nodes for Agent A and Agent B. Implement the session state, counter management, and the drift monitor as a conditional node. Run an end-to-end test with two agents completing a real task (e.g., entity extraction pipeline).

### Phase 5 — Evaluation (Days 8–10)
Implement all five evaluation metrics. Run the full benchmark. Generate comparison against the standard LangGraph messaging baseline. Produce plots and summary statistics.

### Phase 6 — Dashboard and Documentation (Days 11–14)
Build the Streamlit dashboard. Write the README and technical report. Polish the codebase for sharing.

---

## 13. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| tanh saturation causes high reconstruction error | Medium | High | Input normalization to [-0.5, 0.5] range before projection |
| Codebook size (4096) insufficient for session diversity | Low | Medium | Support two-index encoding for fine-grained quantization |
| LanceDB performance degrades at 10K+ entries | Low | Medium | Partition ledger by session; prune stale entries |
| RCC-8 thresholds poorly calibrated for task domain | Medium | Medium | Empirically calibrate thresholds on 100-task pilot before full evaluation |
| Eavesdropper classifier beats 1.5x random baseline | Medium | High | If this occurs, increase codebook size or add additional noise injection layer |
| DeepSeek API rate limits disrupt evaluation runs | Low | Low | Cache LLM calls; use local model (Ollama) as fallback |

---

## 14. Success Criteria

The project is considered successful when all of the following are achieved:

1. A working end-to-end implementation where two LangGraph agents complete a multi-step task using exclusively GQESL messaging (no natural language on the wire)
2. Compression ratio > 20x after ledger warmup
3. Task allocation accuracy > 80% on the 100-task benchmark
4. Semantic recovery fidelity (cosine similarity) > 0.92 mean
5. Eavesdropper classifier accuracy below 1.5x random baseline
6. Semantic drift rate < 5% over 1000-message sessions
7. A Streamlit dashboard displaying live metrics during agent execution
8. A written technical report suitable for submission to an AI systems venue

---

## 15. Glossary

| Term | Definition |
|---|---|
| A2A | Agent-to-Agent — communication between two autonomous AI agents |
| ECDH | Elliptic Curve Diffie-Hellman — key exchange protocol that produces a shared secret without transmitting it |
| ESL | Emergent Semantic Language — a communication protocol that emerges from agent interaction rather than being predefined |
| GQESL | GeoQuantum Emergent Semantic Language — the name of this protocol |
| HKDF | HMAC-based Key Derivation Function — expands a shared secret into multiple derived keys |
| HMAC | Hash-based Message Authentication Code — cryptographic integrity check |
| Intent Tensor | A 384-dimensional float vector representing an agent's task intent, constructed without natural language |
| RCC-8 | Region Connection Calculus (8 relations) — a formal spatial topology system used here as a coordination operator |
| Semantic Ledger | A per-session vector store mapping concept vectors to registered concept identifiers |
| Shared Secret | A 32-byte value known only to both agents, computed via ECDH, never transmitted |
| Superposition | A weighted sum of multiple concept vectors representing ambiguous or multi-faceted intent |
| W1, W2 | Secret 384×384 projection matrices derived from the shared secret; define the private semantic space |
| Wire Packet | The 38-byte message transmitted between agents: {idx, r, c, h} |
| X25519 | A modern, efficient elliptic curve used for ECDH key exchange |

