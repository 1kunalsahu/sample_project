
from generation.generator import create_chat_model
from critic.verifier import review_claims

client = create_chat_model(
    "ollama",
    "llama3:latest",
    "http://localhost:11434",
)

tests = [
    {
        "name": "Supported claim",
        "question": "What services does the company provide?",
        "evidence": [
            {
                "chunk_id": "chunk_1",
                "text": "The company provides cloud storage services to businesses.",
            }
        ],
        "claim": "The company provides cloud storage services to businesses.",
        "expected": "SUPPORTED",
    },
    {
        "name": "Unsupported claim",
        "question": "When was the company founded?",
        "evidence": [
            {
                "chunk_id": "chunk_1",
                "text": "The company provides cloud storage services to businesses.",
            }
        ],
        "claim": "The company was founded in 1998.",
        "expected": "UNSUPPORTED",
    },
    {
        "name": "Contradicted claim",
        "question": "When was the company founded?",
        "evidence": [
            {
                "chunk_id": "chunk_1",
                "text": "The company was founded in 2010.",
            }
        ],
        "claim": "The company was founded in 1998.",
        "expected": "CONTRADICTED",
    },
]

for test in tests:
    claims = [
        {
            "claim_id": "claim_0001",
            "text": test["claim"],
        }
    ]

    result = review_claims(
        client=client,
        question=test["question"],
        evidence=test["evidence"],
        answer=test["claim"],
        claims=claims,
    )

    actual = result["verdicts"][0]["verdict"]
    rationale = result["verdicts"][0]["rationale"]

    print(f"\n--- {test['name']} ---")
    print(f"Expected: {test['expected']}")
    print(f"Actual:   {actual}")
    print(f"Result:   {'PASS' if actual == test['expected'] else 'CHECK'}")
    print(f"Rationale: {rationale}")
