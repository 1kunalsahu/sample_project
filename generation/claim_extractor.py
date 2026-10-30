
from __future__ import annotations

import re

# Matches citations such as [65404] or [chunk_12].
CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")

# Matches evidence labels and captures their citation markers.
EVIDENCE_LABEL_PATTERN = re.compile(
    r"\(?\s*Evidence\s*:\s*((?:\[[^\]]+\]\s*)+)\)?",
    flags=re.IGNORECASE,
)

# Removes numbered and bulleted list prefixes.
LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:(?:\d+[.)])|[-*•])\s+")

# Matches a sentence followed by citations immediately after its punctuation.
SENTENCE_PATTERN = re.compile(
    r".+?(?:[.!?](?:\s*\[[A-Za-z0-9_.:-]+\])*(?=\s|$)|$)"
)


def extract_claims(answer: str) -> list[dict]:
    """Extract sentence-level claims and nearby citations.

    Heuristic extraction only; it does not establish whether a citation
    actually supports the extracted claim.
    """
    claims: list[dict] = []
    order = 0

    for line in answer.strip().splitlines():
        line = LIST_PREFIX_PATTERN.sub("", line.strip()).strip()
        if not line:
            continue

        # Convert "(Evidence: [65404])" into "[65404]".
        line = EVIDENCE_LABEL_PATTERN.sub(r"\1", line)

        for match in SENTENCE_PATTERN.finditer(line):
            fragment = match.group().strip()
            if not fragment:
                continue

            # Capture citations before removing them from the claim text.
            citations = list(
                dict.fromkeys(CITATION_PATTERN.findall(fragment))
            )

            cleaned = CITATION_PATTERN.sub("", fragment)

            # Remove introductory phrases left incomplete by inline citations.
            cleaned = re.sub(
                r"\b(?:according to|as stated by|as reported by)\s*,?\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

            # Remove filler phrases that become awkward after citation removal.
            cleaned = re.sub(
                r"\b(?:additionally,\s*)?"
                r"(?:I would like to point out that|it is worth noting that)\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

            # Repair the specific malformed phrase caused by citation removal.
            cleaned = re.sub(
                r"\bthat\s+states that\b",
                "states that",
                cleaned,
                flags=re.IGNORECASE,
            )

            # Normalize punctuation and whitespace.
            cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

            if not cleaned or re.fullmatch(r"\d+[.)]?", cleaned):
                continue

            claims.append(
                {
                    "claim_id": f"claim_{order + 1:04d}",
                    "text": cleaned,
                    "order": order,
                    "citations": citations,
                    "extraction_status": "sentence_heuristic",
                }
            )
            order += 1

    return claims
