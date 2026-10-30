# from __future__ import annotations

# def citation_metrics(claims: list[dict], verdict_by_id: dict[str,str]) -> dict:
#     cited=[c for c in claims if c.get("citations")]
#     cited_supported=sum(verdict_by_id.get(c["claim_id"])=="SUPPORTED" for c in cited)
#     needs=[c for c in claims if c.get("text","").strip()]
#     appropriate=sum(bool(c.get("citations")) and verdict_by_id.get(c["claim_id"])=="SUPPORTED" for c in needs)
#     return {"cited_claims":len(cited),"citation_faithfulness":cited_supported/len(cited) if cited else None,"claims_requiring_citation":len(needs),"citation_completeness":appropriate/len(needs) if needs else None}

# def citation_faithfulness_from_records(claims: list[dict]) -> dict:
#     """Compute critic-assisted citation support from per-claim supporting chunk IDs.

#     A citation counts as faithful when the claim verdict is SUPPORTED and at least one
#     cited chunk is among the chunks the critic identified as supporting that claim.
#     This is an automated proxy; validate it against human judgments before publication.
#     """
#     cited=[c for c in claims if c.get("citations")]
#     faithful=0
#     for claim in cited:
#         citations=set(claim.get("citations",[]))
#         support=set(claim.get("supporting_evidence_ids",[]))
#         if claim.get("verdict")=="SUPPORTED" and citations.intersection(support): faithful+=1
#     return {"cited_claims":len(cited),"faithful_cited_claims":faithful,"citation_faithfulness":faithful/len(cited) if cited else None}




from __future__ import annotations


def citation_metrics(
    claims: list[dict],
    verdict_by_id: dict[str, str],
) -> dict:
    """Calculate basic citation faithfulness and citation completeness.

    Citation faithfulness:
        Percentage of cited claims judged SUPPORTED.

    Citation completeness:
        Percentage of extracted claims that are cited and judged SUPPORTED.

    Note:
        These metrics do not verify whether the cited chunks actually support
        the claims. Use citation_faithfulness_from_records for that check.
    """
    cited = [
        claim for claim in claims
        if claim.get("citations")
    ]

    cited_supported = sum(
        verdict_by_id.get(claim["claim_id"]) == "SUPPORTED"
        for claim in cited
    )

    needs = [
        claim for claim in claims
        if claim.get("text", "").strip()
    ]

    appropriate = sum(
        bool(claim.get("citations"))
        and verdict_by_id.get(claim["claim_id"]) == "SUPPORTED"
        for claim in needs
    )

    return {
        "cited_claims": len(cited),
        "citation_faithfulness": (
            cited_supported / len(cited) if cited else None
        ),
        "claims_requiring_citation": len(needs),
        "citation_completeness": (
            appropriate / len(needs) if needs else None
        ),
    }


def citation_faithfulness_from_records(claims: list[dict]) -> dict:
    """Measure whether cited chunk IDs match evaluator-identified support.

    A claim counts as citation-faithful only when:
    1. Its verdict is SUPPORTED.
    2. At least one cited chunk ID matches a supporting evidence ID.

    This is an automated proxy and requires human validation.
    """

    cited = [
        claim for claim in claims
        if claim.get("citations")
    ]

    faithful = 0

    for claim in cited:
        citations = {
            str(chunk_id).strip()
            for chunk_id in claim.get("citations", [])
            if chunk_id is not None
        }

        supporting_evidence = {
            str(chunk_id).strip()
            for chunk_id in claim.get("supporting_evidence_ids", [])
            if chunk_id is not None
        }

        if (
            claim.get("verdict") == "SUPPORTED"
            and citations.intersection(supporting_evidence)
        ):
            faithful += 1

    return {
        "cited_claims": len(cited),
        "faithful_cited_claims": faithful,
        "citation_faithfulness": (
            faithful / len(cited) if cited else None
        ),
    }
