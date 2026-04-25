# gqesl_a2a/scenarios/wheel_of_time.py
#
# GQESL scenario: person-name extraction over a WoT-inspired passage.
# Natural language never appears on the wire — only SemanticMessage bytes.
#
# Run: python -m gqesl_a2a.scenarios.wheel_of_time
# Report: python -m gqesl_a2a.scenarios.wheel_of_time --markdown [--skip-llm]

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from gqesl_a2a import config
from gqesl_a2a.agents.agent_a import AgentA
from gqesl_a2a.agents.agent_b import AgentB
from gqesl_a2a.agents.session import InProcessBus, bootstrap_session
from gqesl_a2a.core.crypto import compute_salt, get_session_keys
from gqesl_a2a.core.ledger import SemanticLedger, get_ledger, set_ledger
from gqesl_a2a.core.semantic_state import (
    cosine_similarity,
    decode_pipeline_vector_stages,
    encode_pipeline_vector_stages,
)
from gqesl_a2a.core.tensor_builder import (
    AgentIntent,
    EntityType,
    OutputFormat,
    TaskType,
    build_basis_matrix,
    build_intent_tensor,
    collapse_to_intent,
)

logger = logging.getLogger(__name__)


class RunRecorder:
    """Captures stdout lines plus a short comment for markdown export."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []

    def emit(self, text: str, comment: str = "") -> None:
        # Windows cp1252 consoles cannot print U+2212 / U+2192 from pipeline traces
        print(_console_safe(text))
        self.entries.append((text, comment))

    def emit_workflow(self, title: str, steps: list[dict[str, Any]] | None) -> None:
        self.emit("", "blank line before subsection")
        self.emit("  " + ("-" * 58), "visual separator")
        self.emit(f"  {title}", "workflow subsection title")
        self.emit("  " + ("-" * 58), "visual separator")
        if not steps:
            self.emit("  (no steps)", "no LangGraph-style trace returned")
            return
        for i, step in enumerate(steps, start=1):
            name = step.get("step", "?")
            detail = step.get("detail", "")
            self.emit(f"  {i:2}. {name}", f"pipeline step name: {name}")
            if detail:
                for line in detail.split("\n"):
                    self.emit(f"      {line}", "extra detail for this step")


def _format_full_vector_json(arr: np.ndarray) -> str:
    """One JSON array per line is huge; use compact single-line JSON."""
    flat = np.asarray(arr, dtype=np.float64).ravel()
    return json.dumps([float(x) for x in flat], separators=(",", ":"))


# Order and copy match ``encode_pipeline_vector_stages`` in ``semantic_state``.
_STEP3_ENCODE_STAGE_EXPLAIN: list[tuple[str, str, str]] = [
    (
        "01_raw_intent",
        "Raw intent (384-d)",
        "This is the **superposition** over the session HKDF **concept basis**: task, entity type, output format, priority, and the passage **source_ref** (SHA-256 of the document). It is the full semantic \"command\" before any nonlinearity. **Natural language text is not placed on the wire**; the intent is this vector and the 41-byte **SemanticMessage** built later.",
    ),
    (
        "02_normalised_half_radius",
        "Scale to a bounded ball (half-radius 0.5)",
        "The tensor is L2-**unitised** then scaled to **half-radius 0.5** so all components live in a **stable region** before `tanh`. That limits activation magnitude, reduces numerical blow-up, and makes later **codebook** distances comparable. An observer without **W1/W2, basis, and salt** cannot turn this into readable prose.",
    ),
    (
        "03_plus_salt",
        "Add HKDF per-message salt (counter `c`)",
        "The **same** salt Agent A and B derive for counter `c` (from the session **salt seed**) is added **element-wise**. So each packet is **bound to one counter value**; combined with the **HMAC** and the strict counter check, old packets cannot be replayed as new work. The salt is **unpredictable** to anyone who does not hold the session material.",
    ),
    (
        "04_tanh_W1_hidden",
        "Nonlinear mix: `tanh(salted @ W1)`",
        "`W1` is **session-private** (from HKDF). The **tanh** squashes the mixed signal into a bounded hidden state. This is **not a reversible linear map** of the raw intent: it is part of a **defense in depth** so that a passive listener only sees a short index and a MAC, not a linear projection of the command.",
    ),
    (
        "05_W2_projected",
        "Linear map to quantisation space: `hidden @ W2`",
        "`W2` is the second orthogonal map in the design (decode uses the paired transposes). The output lies where **Euclidean** distance to **codebook rows** is the right metric (random high-dimensional vectors are nearly orthogonal, so cosine is a weak quantiser).",
    ),
    (
        "06_codebook_row_idx",
        "Quantise: the codebook row chosen as index `idx` on the wire",
        "The real encode path picks the **nearest** row in L2. The **wire carries `idx` (2 bytes)**, not this float block. **That is the core \"bullet\" property**: the receiver checks **HMAC** over (`v`, `idx`, `r`, `c`). Tampering with `idx` (or the relation code `r`) **invalidates the MAC** unless the attacker has the **session HMAC key**.",
    ),
]

# Order matches ``decode_pipeline_vector_stages`` in ``semantic_state``.
_STEP3_DECODE_STAGE_EXPLAIN: list[tuple[str, str, str]] = [
    (
        "01_codebook_row",
        "Codebook row from authenticated `idx`",
        "Agent B has already passed **HMAC** and **counter** checks. The wire delivers **`idx` only**; B loads the corresponding **384-d** row from the **session codebook** (not transmitted as floats). This snapshot is the **start of the inverse pipeline**; it is *not* what traveled on the network as a raw vector.",
    ),
    (
        "02_reverse_W2",
        "Inverse of the second map: `codebook_vec @ W2.T`",
        "Undoes the encoder map through `W2`. Matrices are **QR-orthogonal** in this design, so inversion uses **`.T`**, not a separate pseudo-inverse. The result lives in the **pre-arctanh** space before inverse tanh.",
    ),
    (
        "03_arctanh",
        "Inverse tanh (with clip)",
        "Values are **clipped** to a safe range (see `ARCTANH_CLIP` in config) then **`arctanh`** is applied to approximate the **inverse of `tanh`** from the send side. Quantisation and float noise mean this is an **approximate** reconstruction, not a bit-perfect round trip.",
    ),
    (
        "04_post_arctanh_norm",
        "L2 normalise to half-radius 0.5",
        "Matches the **stability** step on the encode path: keep magnitudes in a **bounded** region before the next map so `reverse_W1` does not blow up numerically.",
    ),
    (
        "05_reverse_W1_salted",
        "Inverse of the first map: `pre_tanh @ W1.T`",
        "Recovers the **salted** tensor in the same domain the sender had **before** removing salt. Again **`W1.T`** is the exact inverse in the QR-orthogonal model.",
    ),
    (
        "06_desalted",
        "Subtract the same HKDF salt (counter `c`)",
        "B derives **the same** per-message `salt` as A for this **counter** from the **salt seed**. If `idx` or `c` were wrong, the MAC would have failed; if salt were wrong, the later **unit intent** would not align with the **basis** for collapse.",
    ),
    (
        "07_unit_intent",
        "Unit vector for **collapse** (reconstructed intent direction)",
        "L2-**unit** normalisation yields the **intent direction** that **Agent B** feeds to **cosine** against the **concept basis** to produce **`concept_id`**, and (with `collapse_to_intent`) to recover coarse task/entity locally. This vector **never** goes back on the wire as 384 floats; execution (e.g. NER) stays **local**.",
    ),
]


def _append_full_vector_stages(
    out: list[str],
    st: dict[str, np.ndarray],
    ordered_keys: list[str],
    expl_map: dict[str, tuple[str, str]],
) -> None:
    for key in ordered_keys:
        vec = st.get(key)
        if vec is None:
            continue
        title, expl_body = expl_map.get(key, (key, ""))
        v = np.asarray(vec, dtype=np.float64).ravel()
        norm = float(np.linalg.norm(v))
        out.append(f"### `{key}` — {title}")
        out.append("")
        out.append(expl_body)
        out.append("")
        out.append(f"**L2 norm (snapshot)** : {norm:.8f}")
        out.append("")
        out.append("```json")
        out.append(_format_full_vector_json(v))
        out.append("```")
        out.append("")


def _workflow_reasoning_lines(
    workflow: list[dict[str, Any]] | None,
    *,
    start_step: str | None = None,
) -> list[str]:
    """Convert workflow step dicts into short markdown bullets."""
    if not workflow:
        return ["- *(no workflow available in this run)*"]
    start_idx = 0
    if start_step is not None:
        for i, step in enumerate(workflow):
            if str(step.get("step", "")) == start_step:
                start_idx = i
                break
    lines: list[str] = []
    for step in workflow[start_idx:]:
        name = str(step.get("step", "?"))
        detail = str(step.get("detail", "")).strip()
        if detail:
            lines.append(f"- `{name}`: {detail}")
        else:
            lines.append(f"- `{name}`")
    return lines


def build_poc_scenario_markdown(
    *,
    encode_stages: OrderedDict[str, np.ndarray] | dict[str, np.ndarray],
    decode_stages: OrderedDict[str, np.ndarray] | dict[str, np.ndarray],
    session_id: str,
    source_ref_prefix: str,
    task_line: str,
    wire_len: int,
    wire_hex: str,
    codebook_idx: int,
    concept_id: str,
    decode_similarity: float,
    relation: str,
    strategy: str,
    doc_ref_prefix: str,
    doc_chars: int,
    doc_match_score: float,
    command_task: str,
    command_entity: str,
    command_format: str,
    command_priority: float,
    collapsed_task: str,
    collapsed_entity: str,
    collapsed_format: str,
    collapsed_priority: float,
    sender_workflow: list[dict[str, Any]] | None,
    receiver_workflow: list[dict[str, Any]] | None,
    skip_llm: bool,
    ner_note_markdown: str,
) -> str:
    """PoC-style markdown: goals, this run, security notes, full encode + decode 384-d traces."""
    iso = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    enc: OrderedDict[str, np.ndarray] = (
        encode_stages
        if isinstance(encode_stages, OrderedDict)
        else OrderedDict(encode_stages)
    )
    dec: OrderedDict[str, np.ndarray] = (
        decode_stages
        if isinstance(decode_stages, OrderedDict)
        else OrderedDict(decode_stages)
    )
    enc_map = {k: (t, b) for k, t, b in _STEP3_ENCODE_STAGE_EXPLAIN}
    dec_map = {k: (t, b) for k, t, b in _STEP3_DECODE_STAGE_EXPLAIN}
    enc_order = [k for k, _, _ in _STEP3_ENCODE_STAGE_EXPLAIN]
    dec_order = [k for k, _, _ in _STEP3_DECODE_STAGE_EXPLAIN]
    sender_reasoning = "\n".join(_workflow_reasoning_lines(sender_workflow))
    receiver_reasoning_after_unit = "\n".join(
        _workflow_reasoning_lines(receiver_workflow, start_step="decode.unit_rescale")
    )
    validation = [
        "1. **No natural language on the wire** — the PoC bus carries only **41 bytes** per `SemanticMessage` (version, `idx`, relation code, counter, HMAC).",
        "2. **Session material** (basis, `W1`, `W2`, codebook, HMAC and salt keying) is established out of band of this document via **ECDH + HKDF**; third parties cannot reproduce the vector traces without it.",
        "3. **Integrity** — the HMAC binds `(v, idx, r, c)`; the decode appendix below is **valid only for the same authenticated `idx`** and counter-derived salt that passed verification in this run.",
        "4. **Replay resistance** — monotonic counter and per-message salt tie each packet to a **single** send instant.",
        "5. **Quantisation** — the wire agreement is on **`idx`**, not on a 1500-byte float blob; the appendices show **reconstructible** 384-d states for **audit and reproducibility**, not an alternate wire encoding.",
    ]
    val_lines = "\n".join(validation)
    out: list[str] = [
        "# Proof of concept: GQESL semantic wire (Wheel of Time scenario)",
        "",
        f"**Capture time:** {iso}  ",
        "**Artifact:** one automated scenario run; vectors are **numerical ground truth** for the tensor geometry in that run.",
        "",
        "---",
        "",
        "## 1. What this PoC is meant to show",
        "",
        "We demonstrate **agent-to-agent** coordination where **no natural-language instruction** is serialized on the network. **Agent A** compresses a structured intent (task, entity, format, document digest) into a **384-dimensional** space, then through a **keyed, nonlinear pipeline** to a **codebook index** and short metadata. **Agent B** receives a **41-byte** packet, **verifies** it cryptographically, then runs the **inverse** pipeline to recover an approximate **unit intent** for **collapse** and **local** execution. This document is the **evidence pack**: it states the **business outcome** of the run, then appends the **full float trace** for both directions so reviewers can see **how every transform changes the vector**.",
        "",
        "---",
        "",
        "## 2. Configuration captured in this run",
        "",
        "| Item | Value |",
        "|------|--------|",
        f"| Semantic task | {task_line} |",
        f"| `source_ref` (first 12 bytes of SHA-256, hex) | `{source_ref_prefix}...` |",
        f"| `session_id` (prefix) | `{session_id[:16]}...` |",
        f"| `idx` in packet (codebook) | **{codebook_idx}** |",
        f"| Optional local NER | {'skipped (`--skip-llm`)' if skip_llm else 'executed'} |",
        "",
        "---",
        "",
        "## 3. Data path (conceptual)",
        "",
        "```text",
        "Agent A:  build_intent(384) -> encode (norm, salt, W1, tanh, W2, L2->codebook) -> sign HMAC",
        "          -> SemanticMessage 41 B",
        "    |",
        "    v  (in-process bus in this PoC; real deploy: same 41 B constraint)",
        "Agent B:  verify HMAC, check counter, derive salt",
        "          -> decode (codebook[idx], W2.T, arctanh, W1.T, -salt, unit norm)",
        "          -> collapse vs basis  ->  relation/strategy, document pick, (optional) LLM",
        "```",
        "",
        f"The **only** non-local payload in production terms is the **41-byte** record; the appendices show **intermediate 384-d states** for **transparency**, not a second on-wire format.",
        "",
        "---",
        "",
        "## 4. Wire observation (this run)",
        "",
        f"- **Packet length:** {wire_len} bytes  ",
        f"- **Full hex:** `{wire_hex}`  ",
        "",
        "**Note:** The appendices do not stream these floats on the link; they document the **internal** state after each transform for **reproducibility and third-party review**.",
        "",
        "---",
        "",
        "## 5. Agent B: decode outcome and downstream outputs",
        "",
        "After `verify_and_decode`, **Agent B** holds a **reconstructed 384-d** direction (see Appendix B, stage `07_unit_intent`) and **collapses** it against the session **basis** to name a **concept** and a **cosine** score. The RCC-8 **relation** on the packet is mapped to a **coordination strategy**. Passage text is **resolved locally** from registered document digests, not from the wire.",
        "",
        f"| Field | Observed in this run |",
        f"|--------|------------------------|",
        f"| `concept` | **{concept_id}** (cosine vs basis = **{decode_similarity:.4f}**) |",
        f"| `r` (RCC-8 on wire) | **{relation}** |",
        f"| Strategy | **{strategy}** |",
        f"| Document resolve (local) | best registered SHA-256 match **score = {doc_match_score:.4f}**; **{doc_chars}** chars; ref prefix `{doc_ref_prefix}...` |",
        "",
        "### Agent A command (explicit)",
        "",
        "This is the command Agent A formed before wire packing:",
        "",
        f"- **task_type**: `{command_task}`",
        f"- **entity_type**: `{command_entity}`",
        f"- **output_format**: `{command_format}`",
        f"- **priority**: `{command_priority:.6f}`",
        f"- **source_ref prefix**: `{source_ref_prefix}...`",
        "",
        "### Unit-vector collapse output (post `07_unit_intent`)",
        "",
        f"- **task_type**: `{collapsed_task}`",
        f"- **entity_type**: `{collapsed_entity}`",
        f"- **output_format**: `{collapsed_format}`",
        f"- **priority** (reconstructed): `{collapsed_priority:.6f}`",
        "",
        "### Agent B action after unit-vector collapse",
        "",
        "Given the collapsed unit-vector intent, Agent B takes these actions in order:",
        "",
        f"- maps wire relation `{relation}` -> strategy `{strategy}`",
        f"- resolves local document candidate by semantic match (score `{doc_match_score:.4f}`)",
        f"- prepares execute phase locally (NER block if enabled)",
        f"- advances anti-replay counter to lock packet freshness",
        "",
        "### Each agent reasoning (this run)",
        "",
        "**Agent A reasoning (encode -> sign -> pack)**",
        "",
        sender_reasoning,
        "",
        "**Agent B reasoning after unit-vector collapse**",
        "",
        "These are the steps from `decode.unit_rescale` onward, which produce final output behavior.",
        "",
        receiver_reasoning_after_unit,
        "",
        "### Local execute (not on the wire)",
        "",
        ner_note_markdown,
        "",
        "---",
        "",
        "## 6. PoC validation (security and integrity in scope)",
        "",
        val_lines,
        "",
        "---",
        "",
        "## 7. Appendix A — Agent A: encode pipeline (full 384 floats per stage)",
        "",
        "Each block is a **JSON array of 384** `float` values, one line, after the named transform. Order matches `encode_pipeline_vector_stages` in the codebase.",
        "",
    ]
    _append_full_vector_stages(out, enc, enc_order, enc_map)
    out += [
        "---",
        "",
        "## 8. Appendix B — Agent B: decode / reconstruction pipeline (full 384 floats per stage)",
        "",
        "These snapshots mirror **`decode_tensor`** and `decode_pipeline_vector_stages`. **On the real wire, Agent B only sees the 41-byte packet**; the continuous vectors below are the **internal** state after each inverse step, given the **same** `idx`, `c`, and session keys as in this run. Stage **`07_unit_intent`** is the vector used for **collapse** into `concept_id` and for downstream intent resolution.",
        "",
    ]
    _append_full_vector_stages(out, dec, dec_order, dec_map)
    out += [
        "---",
        "",
        "## 9. How to reproduce this file",
        "",
        "```bash",
        "python -m gqesl_a2a.scenarios.wheel_of_time --step3-encode-md gqesl_a2a/scenarios/wheel_of_time_step3_encode.md --skip-llm",
        "```",
        "",
        "Omit `--skip-llm` to include a real **name list** in section 5. Add `--markdown path.md` to also write the full transcript export.",
    ]
    return "\n".join(out)


def build_markdown_report(
    *,
    recorder: RunRecorder,
    session_id: str,
    encode_stages: dict[str, np.ndarray],
    decode_stages: dict[str, np.ndarray],
    codebook_idx: int,
    wire_hex: str,
    wire_len: int,
    skip_llm: bool,
) -> str:
    """Assemble markdown: annotated transcript + full 384-d vectors per stage."""
    iso = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    parts: list[str] = [
        "# GQESL Wheel of Time scenario — real run export",
        "",
        f"Generated: **{iso}**",
        "",
        "This file combines the live console transcript (with per-line comments), "
        "then **full 384-float** vector snapshots for each encode/decode transform.",
        "",
        "---",
        "",
        "## 1. Annotated console transcript",
        "",
        "| # | Line | Comment |",
        "|---|------|---------|",
    ]
    for i, (line, com) in enumerate(recorder.entries, start=1):
        esc_line = _console_safe(line.replace("|", "\\|").replace("\n", " "))
        esc_com = _console_safe((com or "").replace("|", "\\|"))
        parts.append(f"| {i} | `{esc_line}` | {esc_com} |")
    parts += [
        "",
        "---",
        "",
        "## 2. Session / wire facts",
        "",
        f"- **session_id**: `{session_id}`",
        f"- **wire packet**: `{wire_len}` bytes",
        f"- **wire hex**: `{wire_hex}`",
        f"- **codebook idx** (encode quantisation): `{codebook_idx}`",
        f"- **LLM NER block**: {'skipped (`--skip-llm`)' if skip_llm else 'executed'}",
        "",
        "---",
        "",
        "## 3. Full vectors — sender (encode pipeline)",
        "",
        "Each block is **384 floats** (JSON array). Values change after every transform.",
        "",
    ]
    for name, vec in encode_stages.items():
        v = np.asarray(vec, dtype=np.float64).ravel()
        norm = float(np.linalg.norm(v))
        parts.append(f"### `{name}`")
        parts.append("")
        parts.append(f"<!-- L2 norm ≈ {norm:.8f} -->")
        parts.append("")
        parts.append("```json")
        parts.append(_format_full_vector_json(v))
        parts.append("```")
        parts.append("")
    parts += [
        "---",
        "",
        "## 4. Full vectors — receiver (decode pipeline)",
        "",
    ]
    for name, vec in decode_stages.items():
        v = np.asarray(vec, dtype=np.float64).ravel()
        norm = float(np.linalg.norm(v))
        parts.append(f"### `{name}`")
        parts.append("")
        parts.append(f"<!-- L2 norm ≈ {norm:.8f} -->")
        parts.append("")
        parts.append("```json")
        parts.append(_format_full_vector_json(v))
        parts.append("```")
        parts.append("")
    return "\n".join(parts)

# ─────────────────────────────────────────────────────────────────
# Local document registry (session-local store; not on wire)
# ─────────────────────────────────────────────────────────────────

_DOCUMENTS: dict[bytes, str] = {}


def register_document(text: str) -> bytes:
    """SHA-256 digest becomes the 32-byte ``source_ref`` in ``AgentIntent``."""
    ref = hashlib.sha256(text.encode("utf-8")).digest()
    _DOCUMENTS[ref] = text
    return ref


def get_document(ref: bytes) -> str:
    return _DOCUMENTS[ref]


def _document_refs() -> list[bytes]:
    return list(_DOCUMENTS.keys())


def resolve_document_for_decoded_tensor(
    decoded: np.ndarray,
    basis: np.ndarray,
) -> tuple[bytes, str, float]:
    """Pick the registered document whose intent tensor best matches ``decoded``.

    Quantisation + salt make exact ``source_ref`` recovery from the tensor alone
    unreliable; with one (or few) registered passages this resolves the right text.

    ``collapse_to_intent`` recovers coarse task/entity/format; priority is fuzzy, so
    we score each document with both the collapsed priority and ``1.0``.
    """
    collapsed = collapse_to_intent(decoded, basis)
    p_collapsed = float(np.clip(collapsed["priority"], 0.0, 1.0))
    priority_candidates = {p_collapsed, 1.0}

    best_ref: bytes | None = None
    best_sim = -1.0
    for ref in _document_refs():
        for prio in priority_candidates:
            intent = AgentIntent(
                collapsed["task_type"],
                collapsed["entity_type"],
                collapsed["output_format"],
                prio,
                ref,
            )
            cand = build_intent_tensor(intent, basis)
            sim = cosine_similarity(decoded.astype(np.float32), cand.astype(np.float32))
            if sim > best_sim:
                best_sim = sim
                best_ref = ref
    if best_ref is None:
        raise RuntimeError("No documents registered — call register_document first")
    return best_ref, get_document(best_ref), best_sim


# ─────────────────────────────────────────────────────────────────
# Passage (original writing inspired by Book One — not a direct quote)
# ─────────────────────────────────────────────────────────────────

DOCUMENT = """
The village of Emond's Field sat quietly in the Two Rivers,
far from the concerns of kings and thrones. Rand al'Thor,
a tall young farmer with reddish hair, walked alongside his
father Tam al'Thor toward the village on Bel Tine eve.

In the village inn, the Winespring Inn owned by Brandelwyn
al'Vere, called Bran by all who knew him, a gleeman had
arrived. His name was Thom Merrilin, and he wore a cloak
patched with every color imaginable.

Rand's closest friends were Matrim Cauthon, known as Mat,
a boy with a love of games and mischief, and Perrin Aybara,
a broad-shouldered blacksmith's apprentice who rarely spoke
but thought deeply when he did. The village Wisdom, Nynaeve
al'Meara, watched the festivities with her characteristic
frown, her long dark braid pulled over one shoulder.

Egwene al'Vere, daughter of the innkeeper, joined the group
near the old oak. She had grown up alongside Rand and the
others, though lately she seemed older somehow. A strange
woman had arrived in the village that same evening — Moiraine
Damodred, accompanied by her Warder Lan Mandragoran, a man
who moved like a hunting wolf and spoke even less than Perrin.

The Dark One's reach had stretched even to Emond's Field that
night, and nothing would be the same for Rand, Mat, Perrin,
Egwene, and Nynaeve after the Trollocs came.
"""

EXPECTED_NAMES = [
    "Rand al'Thor",
    "Tam al'Thor",
    "Brandelwyn al'Vere",
    "Thom Merrilin",
    "Matrim Cauthon",
    "Perrin Aybara",
    "Nynaeve al'Meara",
    "Egwene al'Vere",
    "Moiraine Damodred",
    "Lan Mandragoran",
]


def _console_safe(s: str) -> str:
    return (
        s.replace("\u2212", "-")
        .replace("\u2192", "->")
        .replace("\u2265", ">=")
        .replace("\u2264", "<=")
        .replace("\u00d7", "x")
        .replace("\u2713", "+")
        .replace("\u00b1", "+/-")
    )


def _say(recorder: RunRecorder | None, text: str, comment: str = "") -> None:
    if recorder is not None:
        recorder.emit(text, comment)
    else:
        print(_console_safe(text))


def _task_summary_from_intent(intent: AgentIntent, *, suffix: str = "") -> str:
    """Build a display task summary from the runtime intent."""
    base = (
        f"{intent.task_type.name} + {intent.entity_type.name} -> {intent.output_format.name}"
    )
    return f"{base}{suffix}"


def _receiver_graph_lane(recorder: RunRecorder | None) -> None:
    """Maps this demo onto the LangGraph receiver topology in ``graph/graph.py``."""
    _say(recorder, "", "blank line")
    _say(
        recorder,
        "  LangGraph receiver graph (same logical lane as this demo):",
        "maps this script to LangGraph nodes",
    )
    _say(
        recorder,
        "    verify -> decode -> rcc8_route -> [exact_split|handoff|negotiate]",
        "receiver graph order",
    )
    _say(
        recorder,
        "    -> collapse -> [drift_monitor] -> execute -> respond",
        "continuation after routing",
    )
    _say(
        recorder,
        "  Here: Agent B ``verify_and_decode`` = verify + decode + collapse + strategy",
        "what Agent B class bundles in one call",
    )
    _say(
        recorder,
        "        NER block below = ``execute_node``-style work (LLM local only).",
        "LLM never serialized to wire",
    )


def _print_workflow_steps_console(title: str, steps: list[dict[str, Any]] | None) -> None:
    print(f"\n  {'-' * 58}")
    print(f"  {title}")
    print(f"  {'-' * 58}")
    if not steps:
        print("  (no steps)")
        return
    for i, step in enumerate(steps, start=1):
        name = step.get("step", "?")
        detail = step.get("detail", "")
        print(_console_safe(f"  {i:2}. {name}"))
        if detail:
            for line in detail.split("\n"):
                print(_console_safe(f"      {line}"))


async def run_scenario(
    *,
    recorder: RunRecorder | None = None,
    skip_llm: bool = False,
    markdown_path: Path | None = None,
    step3_encode_markdown_path: Path | None = None,
) -> None:
    if (markdown_path is not None or step3_encode_markdown_path is not None) and recorder is None:
        recorder = RunRecorder()
    source_ref = register_document(DOCUMENT.strip())
    intent = AgentIntent(
        task_type=TaskType.EXTRACT,
        entity_type=EntityType.PERSON,
        output_format=OutputFormat.JSON,
        priority=1.0,
        source_ref=source_ref,
    )
    task_summary_line = _task_summary_from_intent(
        intent, suffix=" over a WoT-inspired passage"
    )

    _say(recorder, "", "blank line before title banner")
    _say(recorder, "=" * 62, "banner top rule")
    _say(recorder, "  GQESL private A2A - Wheel of Time scenario", "scenario title")
    _say(recorder, f"  Task: {task_summary_line}", "task summary")
    _say(recorder, "  Wire: SemanticMessage only (41 B per packet) - no NL on wire", "wire constraint")
    _say(recorder, "=" * 62, "banner bottom rule")

    _say(recorder, "", "blank line")
    _say(recorder, "  End-to-end phases (this script)", "numbered high-level plan")
    _say(
        recorder,
        "    1. Bootstrap session + ledger (ECDH, HKDF, basis, warm vocabulary)",
        "cryptographic + ledger setup",
    )
    _say(
        recorder,
        "    2. Register passage locally -> source_ref = SHA-256(document)",
        "document digest becomes tensor ingredient",
    )
    _say(
        recorder,
        "    3. Register warm_wot_book1_passage for RCC-8 on this intent shape",
        "adds vocabulary row for relation scoring",
    )
    _say(
        recorder,
        "    4. Agent A: build intent tensor -> encode -> sign -> 41 B wire",
        "sender pipeline",
    )
    _say(recorder, "    5. InProcessBus: A -> B (async queues)", "in-process transport")
    _say(
        recorder,
        "    6. Agent B: HMAC verify -> decode -> collapse -> strategy",
        "receiver pipeline",
    )
    _say(
        recorder,
        "    7. Resolve passage from decoded tensor (candidate source_refs)",
        "local document pick (not on wire)",
    )
    _say(
        recorder,
        "    8. Agent B: NER via LLM (local only; never on wire)",
        "optional named-entity extraction",
    )

    _say(recorder, "", "blank line")
    _say(recorder, "[ BOOTSTRAP ]", "section: session start")
    set_ledger(SemanticLedger())
    info_a, info_b = bootstrap_session()
    session_id = info_a.session_id
    keys = get_session_keys(session_id)
    basis = build_basis_matrix(keys.basis_seed)

    t0 = build_intent_tensor(intent, basis).astype(np.float64)
    nrm = np.linalg.norm(t0)
    template = (t0 / nrm * 0.5).astype(np.float32) if nrm > 0 else t0.astype(np.float32)
    get_ledger().register(template, "warm_wot_book1_passage", session_id)

    ledger = get_ledger()
    _say(recorder, f"  session_id     : {session_id[:16]}...", "truncated UUID for display")
    _say(recorder, f"  ledger rows    : {ledger.concept_count(session_id)}", "ledger row count")
    _say(recorder, f"  document chars : {len(DOCUMENT.strip())}", "UTF-8 character count of passage")
    _say(
        recorder,
        f"  source_ref     : {source_ref.hex()[:24]}... (SHA-256 of UTF-8 text)",
        "first 12 bytes of digest shown",
    )

    bus_a, bus_b = InProcessBus.create_pair()
    agent_a = AgentA(info_a)
    agent_b = AgentB(info_b)

    _say(recorder, "", "blank line")
    _say(recorder, "[ AGENT A - encode + sign ]", "section: sender")
    result_a = agent_a.encode_and_sign(intent, with_workflow=True)
    wire_bytes = result_a["wire_bytes"]
    if recorder is not None:
        recorder.emit_workflow("Sender pipeline (tensor -> wire)", result_a.get("workflow"))
    else:
        _print_workflow_steps_console("Sender pipeline (tensor -> wire)", result_a.get("workflow"))

    _say(
        recorder,
        f"\n  Outcome: idx={result_a['packet']['idx']}  r={result_a['relation']}  "
        f"wire={len(wire_bytes)} B",
        "codebook index, RCC-8 relation code, packet size",
    )
    _say(recorder, f"  Hex    : {wire_bytes.hex()}", "full wire payload as hex")

    _say(recorder, "", "blank line")
    _say(recorder, "[ WIRE ]", "section: transport")
    loop = asyncio.get_running_loop()
    t_wire0 = loop.time()
    await bus_a.send(wire_bytes)
    wire_received = await bus_b.receive()
    t_wire1 = loop.time()
    _say(recorder, f"  latency ~{(t_wire1 - t_wire0) * 1000:.3f} ms", "queue send+recv timing")
    assert wire_received == wire_bytes, "bus corruption"
    _say(recorder, "  integrity: OK (bytes match)", "A/B queues preserved bytes")

    total_wire = len(wire_bytes)

    _say(recorder, "", "blank line")
    _say(recorder, "[ AGENT B - verify + decode ]", "section: receiver")
    packet = {
        "v": result_a["packet"]["v"],
        "idx": result_a["packet"]["idx"],
        "r": result_a["packet"]["r"],
        "c": result_a["packet"]["c"],
        "h": result_a["packet"]["h"],
    }
    result_b = agent_b.verify_and_decode(packet, with_workflow=True)
    if result_b is None:
        _say(recorder, "  ERROR: verify/decode failed", "fatal: drop packet")
        return

    if recorder is not None:
        recorder.emit_workflow("Receiver pipeline (wire -> tensor)", result_b.get("workflow"))
    else:
        _print_workflow_steps_console("Receiver pipeline (wire -> tensor)", result_b.get("workflow"))

    _say(
        recorder,
        f"\n  relation : {result_b['relation']}  strategy: {result_b['strategy']}",
        "from wire + RCC-8 strategy map",
    )

    decoded_vec = np.array(result_b["decoded_tensor"], dtype=np.float32)
    collapsed_from_unit = collapse_to_intent(decoded_vec, basis)
    ref_bin, doc_text, doc_sim = resolve_document_for_decoded_tensor(decoded_vec, basis)
    _say(recorder, "", "blank line")
    _say(recorder, "[ DOCUMENT RESOLVE ] (local, not on wire)", "section: pick passage")
    _say(
        recorder,
        f"  best SHA-256 match vs collapsed intent: score={doc_sim:.4f}",
        "cosine best among registered docs",
    )
    _say(recorder, f"  ref      : {ref_bin.hex()[:24]}...", "winning digest prefix")
    _say(recorder, f"  chars    : {len(doc_text)}", "resolved passage length")

    _receiver_graph_lane(recorder)

    ner_step3_note = ""
    if skip_llm:
        _say(recorder, "", "blank line")
        _say(recorder, "[ AGENT B - NER / execute (LLM local only) ]", "section skipped")
        _say(recorder, "  (skipped: `--skip-llm` - no API call)", "placeholder when exporting without LLM")
        ner_step3_note = (
            "NER was **not** run (`--skip-llm`). **Verify, decode, collapse, strategy, and document resolve** "
            "still completed; only the **local JSON name list** from the model is missing in this run."
        )
    else:
        _say(recorder, "", "blank line")
        _say(recorder, "[ AGENT B - NER / execute (LLM local only) ]", "section: LLM")
        llm = ChatOpenAI(
            model=config.DEEPSEEK_MODEL,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            temperature=0.0,
        )
        prompt = (
            "You are a named entity recognition system.\n"
            "Extract every character's full name from the passage below.\n"
            "Return ONLY a valid JSON array of strings.\n"
            "Use full names as in the text. No markdown. No explanation.\n\n"
            f"Passage:\n{doc_text}"
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.lower().startswith("json"):
                raw = raw[4:]
        names = json.loads(raw)

        _say(recorder, f"  model   : {config.DEEPSEEK_MODEL}", "LLM id")
        _say(recorder, f"  names   : {len(names)}", "entity count returned")
        for name in names:
            ok = any(
                name.lower() in e.lower() or e.lower() in name.lower()
                for e in EXPECTED_NAMES
            )
            mark = "+" if ok else "?"
            _say(recorder, f"    {mark}  {name}", "one extracted name vs ground truth")
        lines = [f"  - `{n}`" for n in names[:20]]
        if len(names) > 20:
            lines.append(f"  - ... and {len(names) - 20} more")
        ner_step3_note = (
            f"**Execute (local LLM)** used `{config.DEEPSEEK_MODEL}`. **{len(names)}** names returned.\n\n"
            + "\n".join(lines)
        )

    _say(recorder, "", "blank line")
    _say(recorder, "[ SUMMARY ]", "section: totals")
    _say(
        recorder,
        f"  Total wire bytes this run : {total_wire} (one SemanticMessage)",
        "bytes on bus",
    )
    _say(recorder, "  NL bytes on wire          : 0", "no natural language in packet")
    _say(recorder, f"  Expected names (ground)   : {len(EXPECTED_NAMES)}", "evaluation list size")

    export_full = markdown_path is not None
    export_step3 = step3_encode_markdown_path is not None
    if (export_full or export_step3) and recorder is not None:
        tensor = build_intent_tensor(intent, basis)
        salt = compute_salt(keys.salt_seed, 0)
        enc_stages, idx_snap = encode_pipeline_vector_stages(
            tensor, salt, keys.W1, keys.W2, keys.codebook
        )
        if idx_snap != result_a["packet"]["idx"]:
            logger.warning(
                "Snapshot idx %s != wire idx %s",
                idx_snap,
                result_a["packet"]["idx"],
            )
        idx_use = int(result_a["packet"]["idx"])
        dec_stages = decode_pipeline_vector_stages(
            idx_use, keys.codebook, salt, keys.W1, keys.W2
        )
        if export_full:
            md = build_markdown_report(
                recorder=recorder,
                session_id=session_id,
                encode_stages=enc_stages,
                decode_stages=dec_stages,
                codebook_idx=idx_use,
                wire_hex=wire_bytes.hex(),
                wire_len=len(wire_bytes),
                skip_llm=skip_llm,
            )
            assert markdown_path is not None
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(md, encoding="utf-8")
            _say(recorder, "", "blank line")
            _say(recorder, f"  Wrote markdown report -> {markdown_path}", "export path")
        if export_step3 and result_b is not None:
            assert step3_encode_markdown_path is not None
            step3_body = build_poc_scenario_markdown(
                encode_stages=enc_stages,
                decode_stages=dec_stages,
                session_id=session_id,
                source_ref_prefix=source_ref.hex()[:24],
                task_line=task_summary_line,
                wire_len=len(wire_bytes),
                wire_hex=wire_bytes.hex(),
                codebook_idx=idx_use,
                concept_id=result_b["concept_id"],
                decode_similarity=float(result_b["similarity"]),
                relation=str(result_b["relation"]),
                strategy=str(result_b["strategy"]),
                doc_ref_prefix=ref_bin.hex()[:24],
                doc_chars=len(doc_text),
                doc_match_score=float(doc_sim),
                command_task=intent.task_type.name,
                command_entity=intent.entity_type.name,
                command_format=intent.output_format.name,
                command_priority=float(intent.priority),
                collapsed_task=collapsed_from_unit["task_type"].name,
                collapsed_entity=collapsed_from_unit["entity_type"].name,
                collapsed_format=collapsed_from_unit["output_format"].name,
                collapsed_priority=float(collapsed_from_unit["priority"]),
                sender_workflow=result_a.get("workflow"),
                receiver_workflow=result_b.get("workflow"),
                skip_llm=skip_llm,
                ner_note_markdown=ner_step3_note,
            )
            step3_encode_markdown_path.parent.mkdir(parents=True, exist_ok=True)
            step3_encode_markdown_path.write_text(step3_body, encoding="utf-8")
            _say(
                recorder,
                f"  Wrote PoC / wire-trace report -> {step3_encode_markdown_path}",
                "PoC encode+decode doc path",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Wheel of Time GQESL scenario")
    parser.add_argument(
        "--markdown",
        nargs="?",
        const=Path(__file__).resolve().parent / "wheel_of_time_real_run.md",
        type=Path,
        help="Write annotated transcript + full 384-d vectors to this path",
    )
    parser.add_argument(
        "--step3-encode-md",
        type=Path,
        default=None,
        help=(
            "Write PoC report: full 384-d encode + decode traces, per-stage notes, "
            "and Agent B outcomes (can combine with --markdown)"
        ),
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip DeepSeek NER (faster; note appears in report)",
    )
    args = parser.parse_args()

    if args.markdown is not None or args.step3_encode_md is not None:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    asyncio.run(
        run_scenario(
            markdown_path=args.markdown,
            step3_encode_markdown_path=args.step3_encode_md,
            skip_llm=args.skip_llm,
        )
    )


if __name__ == "__main__":
    main()
