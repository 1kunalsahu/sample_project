
from generation.claim_extractor import extract_claims


# Test 1: Citations after sentences.
answer_1 = (
    "The company was founded in 2010. [chunk_1] "
    "Its revenue increased in 2022. [chunk_2]"
)

claims_1 = extract_claims(answer_1)

assert len(claims_1) == 2
assert claims_1[0]["citations"] == ["chunk_1"]
assert claims_1[1]["citations"] == ["chunk_2"]

print("PASS: citations after sentences.")


# Test 2: Evidence labels used by baseline answers.
answer_2 = (
    "The associate signs the cheque. (Evidence: [65404])\n"
    "The business opens an account. (Evidence: [309023])"
)

claims_2 = extract_claims(answer_2)

assert len(claims_2) == 2
assert claims_2[0]["citations"] == ["65404"]
assert claims_2[1]["citations"] == ["309023"]

print("PASS: evidence-label citations.")


# Test 3: Inline citations should not leave citation gaps in claim text.
answer_3 = (
    "According to [325273], you can fill in your business name "
    "and address in the From section of a money order. "
    "[325273]"
)

claims_3 = extract_claims(answer_3)

assert len(claims_3) == 1
assert claims_3[0]["citations"] == ["325273"]
assert "According to ," not in claims_3[0]["text"]

print("PASS: inline citations are extracted without citation gaps.")
print("\nAll claim extractor tests passed.")
