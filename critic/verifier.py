
from __future__ import annotations

import json

VERDICTS = {"SUPPORTED", "UNSUPPORTED", "CONTRADICTED"}


def critic_prompt(question, chunks, answer, claims):
    evidence = [
        {"chunk_id": c["chunk_id"], "text": c["text"]}
        for c in chunks
    ]

    decision_rules = """
You are a strict evidence-grounded claim evaluator.

Judge the claim using ONLY the supplied retrieved evidence.
Do not use outside knowledge, assumptions, or plausibility.

CLASSIFICATION RULES — FOLLOW IN ORDER:

1. CONTRADICTED
Choose CONTRADICTED if the evidence explicitly states a fact that
conflicts with a material factual part of the claim.

Example:
Evidence: "The company was founded in 2010."
Claim: "The company was founded in 1998."
Verdict: CONTRADICTED.

2. SUPPORTED
Choose SUPPORTED only if the evidence directly supports every
material factual part of the claim, or supports a reasonable inference.

3. UNSUPPORTED
Choose UNSUPPORTED if the evidence neither establishes the claim
nor explicitly conflicts with it, or supports only part of the claim.

IMPORTANT:
- Missing information means UNSUPPORTED, not CONTRADICTED.
- An explicit factual conflict means CONTRADICTED, not UNSUPPORTED.
- Do not mark claims SUPPORTED merely because they sound plausible.
- Do not treat the draft answer or its citations as proof.
- Compare the claim directly against the evidence text.
- Select only evidence chunk IDs relevant to your verdict.
- For SUPPORTED, select chunks that support the claim.
- For CONTRADICTED, select chunks that establish the conflicting fact.
- For UNSUPPORTED, return an empty evidence_chunk_ids list unless
  a supplied chunk provides relevant evidence for a partial assessment.
- If uncertain, choose UNSUPPORTED and explain the uncertainty.
"""

    if len(claims) == 1:
        payload = {
            "question": question,
            "evidence": evidence,
            "draft_answer": answer,
            "claim": claims[0]["text"],
        }

        instruction = decision_rules + """
Evaluate this single claim only.

Return exactly one JSON object with these fields:
{
  "verdict": "SUPPORTED",
  "evidence_chunk_ids": [],
  "rationale": "Brief evidence-based explanation",
  "correction": ""
}

Use exactly one of these verdicts:
SUPPORTED, UNSUPPORTED, CONTRADICTED.

Do not return a claim ID or additional claims.
If the claim is UNSUPPORTED, explain what the evidence fails to establish.
If the claim is CONTRADICTED, explain the conflict.
The correction must be grounded in the supplied evidence, or be empty.
"""

    else:
        payload = {
            "question": question,
            "evidence": evidence,
            "draft_answer": answer,
            "claims": claims,
        }

        instruction = decision_rules + """
Evaluate EVERY provided claim independently.

Return exactly one JSON object with a "claims" array.
Each result must contain:
- claim_id: copy the input claim_id exactly
- verdict: SUPPORTED, UNSUPPORTED, or CONTRADICTED
- evidence_chunk_ids: list of relevant supplied chunk IDs
- rationale: brief evidence-based explanation
- correction: evidence-grounded correction, or empty string

Do not omit claims, duplicate claims, merge claims, or invent IDs.
Return exactly one result for each input claim.
"""

    return [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]


def parse_verdicts(
    raw: str,
    claims: list[dict],
    valid_chunk_ids: set[str],
) -> list[dict]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Critic returned invalid JSON: {e}") from e

    if isinstance(payload, dict) and "verdict" in payload and len(claims) == 1:
        items = [{**payload, "claim_id": claims[0]["claim_id"]}]
    else:
        items = payload.get("claims") if isinstance(payload, dict) else None

    if not isinstance(items, list):
        raise ValueError(
            "Critic response must contain a verdict object or claims array"
        )

    expected = {c["claim_id"] for c in claims}
    seen = set()
    out = []

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each critic claim must be an object")

        cid = item.get("claim_id")
        verdict = item.get("verdict")

        if cid not in expected or cid in seen:
            raise ValueError(f"Unknown or duplicate claim_id: {cid}")

        if verdict not in VERDICTS:
            raise ValueError(f"Invalid verdict for {cid}: {verdict}")

        evidence_ids = item.get("evidence_chunk_ids", [])

        if (
            not isinstance(evidence_ids, list)
            or any(
                not isinstance(chunk_id, str)
                or chunk_id not in valid_chunk_ids
                for chunk_id in evidence_ids
            )
        ):
            raise ValueError(f"Invalid evidence IDs for {cid}")

        seen.add(cid)

        out.append({
            "claim_id": cid,
            "verdict": verdict,
            "evidence_chunk_ids": evidence_ids,
            "rationale": str(item.get("rationale", "")),
            "correction": str(item.get("correction", "")),
        })

    if seen != expected:
        raise ValueError(
            f"Missing critic verdicts: {sorted(expected - seen)}"
        )

    return out


def review_claims(
    client,
    question: str,
    evidence: list[dict],
    answer: str,
    claims: list[dict],
) -> dict:
    """Review each claim independently and retain raw responses."""
    valid_ids = {e["chunk_id"] for e in evidence}
    verdicts = []
    raw = []
    latency = 0.0

    for claim in claims:
        response = client.complete(
            critic_prompt(question, evidence, answer, [claim]),
            temperature=0,
            json_mode=True,
        )

        raw.append(response["text"])
        latency += response["latency_seconds"]

        verdicts.extend(
            parse_verdicts(response["text"], [claim], valid_ids)
        )

    return {
        "verdicts": verdicts,
        "raw_responses": raw,
        "latency_seconds": latency,
    }
