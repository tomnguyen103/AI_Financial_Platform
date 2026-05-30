"""Small RAG evaluation harness for the portfolio demo."""
from __future__ import annotations

import json
import time
from pathlib import Path

from app.db import init_db
from app.rag.chatbot import ask

EVAL_PATH = Path("eval/rag_questions.jsonl")


def load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in EVAL_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    init_db()
    cases = load_cases()
    passed = 0
    total_latency = 0
    rows = []

    for case in cases:
        start = time.time()
        response = ask(case["query"], user_id="eval", user_role="da_analyst")
        latency_ms = int((time.time() - start) * 1000)
        total_latency += latency_ms

        blocked_ok = response.blocked == case["expect_blocked"]
        citation_ok = (len(response.citations) > 0) == case["expect_citation"]
        ok = blocked_ok and citation_ok
        passed += 1 if ok else 0
        rows.append({
            "id": case["id"],
            "ok": ok,
            "blocked_ok": blocked_ok,
            "citation_ok": citation_ok,
            "latency_ms": latency_ms,
            "citations": [c["source_doc_id"] for c in response.citations],
        })

    print(json.dumps({
        "passed": passed,
        "total": len(cases),
        "pass_rate": round(passed / max(len(cases), 1), 3),
        "avg_latency_ms": round(total_latency / max(len(cases), 1), 1),
        "cases": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
