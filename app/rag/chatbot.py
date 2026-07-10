"""RAG chatbot "Ask the Financials" (arch §2.5; PRD Module 5; data design §6.4).

Flow: PHI input scan -> retrieve -> threshold gate -> LLM synthesis with
citations -> PHI output scan -> session/audit log.
"""
from __future__ import annotations

import re
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.config import settings
from app.llm.client import get_llm
from app.rag.index import SIM_THRESHOLD, TOP_K, get_index
from app.security.audit import log_phi_detection, write_audit
from app.security.phi import scan_input, scan_output

SYSTEM_PROMPT = """You are a financial analyst assistant for a medical billing and collections organization.
You answer questions about facility collections, attorney performance, visit billing, and
settlement pipelines using ONLY the provided context documents.

Rules:
1. ONLY use facts present in the provided context. Do not infer or extrapolate.
2. ALWAYS cite your sources using the format: [Source: {source_doc_id}, {date}].
3. If the context does not contain enough information to answer, say:
   "I don't have enough data to answer this reliably. Please check with the DA team."
4. NEVER include patient names, dates of birth, SSNs, or any patient-identifying information.
5. When presenting numbers, always specify the time period they cover.
6. Be concise and direct — the user is a collections professional, not a data scientist.
"""

INSUFFICIENT = "I don't have enough information to answer this reliably. Please check with the DA team."

DOMAIN_TERMS = {
    "aging", "alert", "anomaly", "attorney", "balance", "billing", "case",
    "cash", "collection", "collections", "collect", "collected", "facility",
    "forecast", "insurance", "lop", "outstanding", "payment", "payments",
    "performance", "revenue", "settlement", "visit", "visits",
}


@dataclass
class ChatResponse:
    answer: str
    citations: list[dict] = field(default_factory=list)
    insufficient: bool = False
    blocked: bool = False
    phi_redacted: bool = False
    latency_ms: int = 0
    session_id: str = ""
    retrieval: dict = field(default_factory=dict)


def _norm_entity(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _extract_entity(query: str, docs: Iterable) -> str | None:
    q = _norm_entity(query)
    matches: list[tuple[int, str]] = []
    for doc in docs:
        entity = getattr(doc, "entity_id", "")
        if entity and _norm_entity(entity) in q:
            matches.append((len(_norm_entity(entity)), entity))
    if matches:
        matches.sort(reverse=True)
        return matches[0][1]
    return None


def _looks_domain_relevant(query: str) -> bool:
    q = _norm_entity(query)
    terms = set(q.split())
    return bool(terms & DOMAIN_TERMS)


def _format_context(hits) -> tuple[str, list[dict]]:
    blocks, cites = [], []
    for hit in hits:
        blocks.append(f"[{hit.source_doc_id} | {hit.date}] {hit.text}")
        cites.append({
            "source_doc_id": hit.source_doc_id,
            "date": hit.date,
            "entity_id": hit.entity_id,
            "entity_type": hit.entity_type,
            "score": round(hit.score, 3),
            "metadata": hit.metadata,
        })
    return "\n".join(blocks), cites


def ask(query: str, *, user_id: str = "anon", user_role: str = "collections",
        session_id: str | None = None) -> ChatResponse:
    start = time.time()
    session_id = session_id or str(uuid.uuid4())
    retrieval = {
        "vector_store": settings.vector_store,
        "embedding_model": settings.openai_embed_model if settings.llm_enabled else "stub-hash",
        "llm_model": settings.openai_model if settings.llm_enabled else "stub",
        "top_k": TOP_K,
        "similarity_threshold": SIM_THRESHOLD,
        "entity_filter": None,
        "top_score": 0.0,
        "retrieved_count": 0,
        "status": "not_started",
        "reason": "",
    }

    # 1. PHI input scan (AC-5.5).
    sin = scan_input(query)
    if sin.blocked:
        log_phi_detection("chatbot", user_id, user_role, sin.matches, "INPUT")
        retrieval.update({"status": "skipped", "reason": "input_phi_blocked"})
        return ChatResponse(
            answer="This request appears to ask for individual patient information, "
                   "which I can't provide. Please ask about aggregated financial metrics.",
            blocked=True, latency_ms=int((time.time() - start) * 1000),
            session_id=session_id, retrieval=retrieval)

    # 2. Retrieve.
    idx = get_index()
    entity_id = _extract_entity(query, idx.docs)
    retrieval["entity_filter"] = entity_id
    if entity_id is None and not _looks_domain_relevant(query):
        retrieval.update({"status": "insufficient", "reason": "outside_financial_domain"})
        write_audit(user_id=user_id, user_role=user_role, service="chatbot",
                    action="insufficient", query_text=query, input_phi_scan="clean",
                    retrieved_sources=[],
                    response_latency_ms=int((time.time() - start) * 1000), session_id=session_id)
        return ChatResponse(answer=INSUFFICIENT, insufficient=True,
                            latency_ms=int((time.time() - start) * 1000),
                            session_id=session_id, retrieval=retrieval)

    hits = idx.search(query, entity_id=entity_id)
    top_score = hits[0].score if hits else 0.0
    retrieval.update({"top_score": round(top_score, 3), "retrieved_count": len(hits)})

    # 3. Threshold gate (AC-5.3).
    if not hits or (top_score < SIM_THRESHOLD and entity_id is None):
        retrieval.update({
            "status": "insufficient",
            "reason": "below_similarity_threshold" if hits else "no_hits",
        })
        write_audit(user_id=user_id, user_role=user_role, service="chatbot",
                    action="insufficient", query_text=query, input_phi_scan="clean",
                    retrieved_sources=[h.source_doc_id for h in hits],
                    response_latency_ms=int((time.time() - start) * 1000), session_id=session_id)
        return ChatResponse(answer=INSUFFICIENT, insufficient=True,
                            latency_ms=int((time.time() - start) * 1000),
                            session_id=session_id, retrieval=retrieval)

    context, cites = _format_context(hits[:4])  # top-4 after retrieval (§6.3)
    retrieval.update({
        "status": "grounded",
        "reason": "entity_filter_match" if entity_id else "threshold_passed",
        "retrieved_count": len(cites),
    })

    # 4. LLM synthesis.
    llm = get_llm()
    user_prompt = f"Context documents:\n{context}\n\nQuestion: {query}\n\nAnswer with citations."
    raw = llm.complete(SYSTEM_PROMPT, user_prompt)
    if not llm.enabled:
        # Stub: produce a grounded answer from the top doc so offline output is useful.
        raw = (f"{hits[0].text} "
               f"[Source: {cites[0]['source_doc_id']}, {cites[0]['date']}]")

    # 5. PHI output scan (AC-5.6).
    sout = scan_output(raw)
    answer = sout.redacted_text if sout.phi_detected else raw
    if sout.phi_detected:
        log_phi_detection("chatbot", user_id, user_role, sout.matches, "OUTPUT")

    latency = int((time.time() - start) * 1000)
    write_audit(user_id=user_id, user_role=user_role, service="chatbot",
                action="answer", query_text=query, input_phi_scan="clean",
                output_phi_scan="redacted" if sout.phi_detected else "clean",
                llm_model=llm.model_name, retrieved_sources=[c["source_doc_id"] for c in cites],
                response_latency_ms=latency, session_id=session_id)

    return ChatResponse(answer=answer, citations=cites, phi_redacted=sout.phi_detected,
                        latency_ms=latency, session_id=session_id, retrieval=retrieval)


if __name__ == "__main__":
    from app.db import init_db
    init_db()
    r = ask("Why did round_rock collections drop recently?")
    print(r.answer)
    print("citations:", r.citations)
