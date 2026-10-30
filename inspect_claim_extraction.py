
import json
from pathlib import Path

from generation.claim_extractor import extract_claims

files = [
    Path("results/raw/baseline_pilot_v2.jsonl"),
    Path("results/raw/critic_pilot_v2.jsonl"),
]

for file_path in files:
    if not file_path.exists():
        print(f"\nFile not found: {file_path}")
        continue

    print(f"\n{'=' * 70}")
    print(f"FILE: {file_path}")
    print("=" * 70)

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line_number > 2:
                break

            record = json.loads(line)
            answer = record.get("answer", "")

            print(f"\nRecord {line_number}")
            print("Question:", record.get("question", "N/A"))
            print("Answer:", answer)

            print("\nExtracted claims:")
            for claim in extract_claims(answer):
                print(
                    f'{claim["claim_id"]}: {claim["text"]} '
                    f'| citations={claim["citations"]}'
                )
