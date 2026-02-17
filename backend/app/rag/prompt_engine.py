"""
Enterprise Prompt Engineering Layer
====================================
Production-grade prompt construction with:
- Citation-forcing templates
- Chain-of-thought prompting
- Hallucination reduction guardrails
- Query-type adaptive templates
- Context compression for token efficiency
- Conflict resolution instructions
- Multi-turn conversation handling
"""

from __future__ import annotations

import re
from typing import Any

# ── System Prompts ────────────────────────────────────────────────────────

_SYSTEM_PROMPT_BASE = """\
You are OptionQuant AI, an expert assistant for quantitative finance and \
option pricing. Your knowledge covers Black-Scholes, Monte Carlo simulation, \
option Greeks, volatility modeling, deep learning for finance, variance \
reduction techniques, American options, hedging strategies, stochastic \
volatility models, and risk management.

STRICT RULES:
1. ONLY use the provided CONTEXT passages to form your answer. Do NOT use \
any external knowledge or make assumptions beyond the context.
2. If the context is insufficient, say so honestly — NEVER fabricate information.
3. CITE sources explicitly using [Source N] notation when referencing facts.
4. Keep answers concise, well-structured, and educational.
5. Use bullet points or numbered lists for clarity when appropriate.
6. Include relevant formulas in LaTeX notation when helpful.
7. If sources CONFLICT, acknowledge the discrepancy and present both views.
8. End with a brief summary sentence for complex answers.
9. Do NOT hallucinate facts, formulas, or citations not in the context."""

_CHAIN_OF_THOUGHT_SUFFIX = """

REASONING APPROACH:
- First, identify which context passages are most relevant
- Then, synthesize information across passages
- Finally, formulate a clear, well-cited answer
- Show your reasoning step by step when the question requires analysis"""

_ANTI_HALLUCINATION_SUFFIX = """

CRITICAL — ANTI-HALLUCINATION PROTOCOL:
- Every factual claim MUST have a [Source N] citation
- If you cannot find supporting evidence in the context, state: \
"The provided context does not contain sufficient information to answer this aspect."
- Do NOT complete partial information with assumptions
- When citing formulas, verify they appear in the context
- Prefer direct quotes over paraphrasing for key definitions"""

# ── Query-Type Specific Instructions ──────────────────────────────────────

_TYPE_INSTRUCTIONS: dict[str, str] = {
    "factual": (
        "INSTRUCTION: Provide a precise, definition-focused answer. "
        "Include the exact formula or definition from the context. "
        "Cite the specific source for each fact."
    ),
    "analytical": (
        "INSTRUCTION: Provide a detailed analytical explanation with reasoning. "
        "Break down the concept step by step. "
        "Use formulas where appropriate and explain each term. "
        "Cite sources for each analytical claim."
    ),
    "comparative": (
        "INSTRUCTION: Provide a structured comparison. "
        "Use a clear format: similarities, differences, pros/cons, or a table. "
        "Ensure each comparison point cites the relevant source. "
        "End with a summary recommendation if applicable."
    ),
    "procedural": (
        "INSTRUCTION: Provide clear step-by-step instructions. "
        "Number each step. Include formulas and parameter descriptions. "
        "Note any prerequisites or assumptions for each step."
    ),
    "general": (
        "INSTRUCTION: Provide a clear, well-grounded answer. "
        "Structure your response logically. "
        "Cite sources for any factual claims."
    ),
}

# ── Context Compression ──────────────────────────────────────────────────


def _compress_evidence(
    evidence: list[str],
    max_total_chars: int = 4000,
) -> list[str]:
    """
    Compress evidence passages to fit within token budget.
    Prioritizes keeping complete sentences and removing redundancy.
    """
    if not evidence:
        return []

    total = sum(len(e) for e in evidence)
    if total <= max_total_chars:
        return evidence

    # Calculate per-passage budget (proportional to original length)
    compressed: list[str] = []
    budget_per = max_total_chars // len(evidence)

    for passage in evidence:
        if len(passage) <= budget_per:
            compressed.append(passage)
        else:
            # Keep complete sentences up to budget
            sentences = re.split(r"(?<=[.!?])\s+", passage)
            kept: list[str] = []
            current_len = 0
            for sent in sentences:
                if current_len + len(sent) + 1 <= budget_per:
                    kept.append(sent)
                    current_len += len(sent) + 1
                else:
                    break
            compressed.append(" ".join(kept) if kept else passage[:budget_per])

    return compressed


def _remove_redundant_evidence(evidence: list[str]) -> list[str]:
    """Remove evidence passages that are subsets of other passages."""
    if len(evidence) <= 1:
        return evidence

    result: list[str] = []
    for i, passage in enumerate(evidence):
        is_redundant = False
        p_lower = passage.lower()[:100]
        for j, other in enumerate(evidence):
            if i != j and p_lower in other.lower():
                is_redundant = True
                break
        if not is_redundant:
            result.append(passage)

    return result if result else evidence


# ── Conflict Detection ────────────────────────────────────────────────────


def _detect_conflicts(evidence: list[str]) -> bool:
    """
    Basic conflict detection: check if evidence contains
    contradictory numerical values or opposing claims.
    """
    # Extract numerical claims
    numbers: dict[str, set[str]] = {}
    for passage in evidence:
        # Find patterns like "X = Y" or "X is Y"
        for match in re.finditer(
            r"(\b\w+\b)\s*(?:=|is|equals?)\s*([0-9]+\.?[0-9]*)",
            passage, re.IGNORECASE,
        ):
            var, val = match.group(1).lower(), match.group(2)
            if var not in numbers:
                numbers[var] = set()
            numbers[var].add(val)

    # Check for conflicting values
    for var, vals in numbers.items():
        if len(vals) > 1:
            return True

    return False


# ── Prompt Builder ────────────────────────────────────────────────────────


def build_system_prompt(
    query_type: str = "general",
    enable_cot: bool = True,
    enable_anti_hallucination: bool = True,
) -> str:
    """Build the system prompt with optional reasoning enhancements."""
    prompt = _SYSTEM_PROMPT_BASE

    if enable_cot and query_type in ("analytical", "comparative", "procedural"):
        prompt += _CHAIN_OF_THOUGHT_SUFFIX

    if enable_anti_hallucination:
        prompt += _ANTI_HALLUCINATION_SUFFIX

    return prompt


def build_user_prompt(
    question: str,
    evidence: list[str],
    sources: list[str],
    query_type: str = "general",
    chat_history: list[dict] | None = None,
    max_context_chars: int = 4000,
    confidence_label: str = "medium",
) -> str:
    """
    Build the user message with retrieved context for the LLM.

    Features:
    - Token-efficient context formatting
    - Source attribution
    - Conversation history (condensed)
    - Query-type specific instructions
    - Conflict awareness
    - Confidence-aware framing
    """
    # Compress and deduplicate evidence
    clean_evidence = _remove_redundant_evidence(evidence)
    compressed = _compress_evidence(clean_evidence, max_context_chars)

    # Format context block with source indexing
    context_block = "\n\n".join(
        f"[Source {i + 1}]: {e}" for i, e in enumerate(compressed)
    )

    source_list = ", ".join(sources[:4]) if sources else "knowledge base"

    # Conversation history (condensed)
    history_block = ""
    if chat_history:
        recent = chat_history[-6:]
        history_lines = []
        for msg in recent:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")[:200]
            history_lines.append(f"{role}: {content}")
        if history_lines:
            history_block = (
                "\nCONVERSATION HISTORY (for context continuity):\n"
                + "\n".join(history_lines) + "\n"
            )

    # Query-type instruction
    type_instruction = _TYPE_INSTRUCTIONS.get(
        query_type, _TYPE_INSTRUCTIONS["general"]
    )

    # Conflict detection
    has_conflicts = _detect_conflicts(compressed)
    conflict_note = ""
    if has_conflicts:
        conflict_note = (
            "\nNOTE: The retrieved sources may contain conflicting information. "
            "Please acknowledge and address any discrepancies.\n"
        )

    # Confidence-aware framing
    confidence_note = ""
    if confidence_label == "low":
        confidence_note = (
            "\nCAUTION: Retrieval confidence is LOW. Be especially careful to "
            "only state what is directly supported by the context. Clearly "
            "indicate any uncertainty.\n"
        )

    return (
        f"CONTEXT (retrieved from: {source_list}):\n"
        f"{context_block}\n"
        f"{history_block}"
        f"{conflict_note}"
        f"{confidence_note}\n"
        f"QUESTION: {question}\n\n"
        f"{type_instruction} Use ONLY the context above. "
        f"Cite [Source N] for every factual claim."
    )


# ── Response Post-Processing ─────────────────────────────────────────────


def validate_response(
    response: str,
    evidence: list[str],
    sources: list[str],
) -> dict[str, Any]:
    """
    Validate the LLM response for quality signals.

    Returns dict with:
    - has_citations: bool
    - citation_count: int
    - potential_hallucination: bool
    - response_quality: str ("high", "medium", "low")
    """
    # Check for citations
    citation_pattern = re.compile(r"\[Source\s*\d+\]", re.IGNORECASE)
    citations = citation_pattern.findall(response)
    has_citations = len(citations) > 0

    # Check for hedging language (sign of uncertain info)
    hedging_patterns = [
        r"\bprobably\b", r"\bmight\b", r"\bperhaps\b",
        r"\bI think\b", r"\bI believe\b", r"\bgenerally\b",
        r"\btypically\b", r"\busually\b",
    ]
    hedging_count = sum(
        1 for p in hedging_patterns
        if re.search(p, response, re.IGNORECASE)
    )

    # Check for potential hallucination signals
    # (claims about things not in evidence)
    evidence_text = " ".join(evidence).lower()

    # Look for specific numbers in response that aren't in evidence
    response_numbers = set(re.findall(r"\b\d+\.?\d*\b", response))
    evidence_numbers = set(re.findall(r"\b\d+\.?\d*\b", evidence_text))
    novel_numbers = response_numbers - evidence_numbers - {"1", "2", "3", "4", "5"}
    potential_hallucination = len(novel_numbers) > 3

    # Quality assessment
    if has_citations and not potential_hallucination and hedging_count < 3:
        quality = "high"
    elif has_citations or (not potential_hallucination):
        quality = "medium"
    else:
        quality = "low"

    return {
        "has_citations": has_citations,
        "citation_count": len(citations),
        "hedging_signals": hedging_count,
        "potential_hallucination": potential_hallucination,
        "novel_numbers": len(novel_numbers),
        "response_quality": quality,
    }


def post_process_response(response: str) -> str:
    """
    Clean and improve the LLM response.
    - Normalize citation format
    - Remove any self-referential statements
    - Clean up formatting
    """
    # Normalize citation format: [source 1] -> [Source 1]
    response = re.sub(
        r"\[source\s*(\d+)\]",
        r"[Source \1]",
        response,
        flags=re.IGNORECASE,
    )

    # Remove "As an AI" type self-references
    response = re.sub(
        r"(?:As an AI|I am an AI|I'm an AI)[^.]*\.\s*",
        "",
        response,
        flags=re.IGNORECASE,
    )

    # Clean up excessive newlines
    response = re.sub(r"\n{3,}", "\n\n", response)

    return response.strip()
