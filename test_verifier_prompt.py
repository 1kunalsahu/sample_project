
from critic.verifier import critic_prompt

question = "When was the company founded?"

evidence = [
    {
        "chunk_id": "chunk_1",
        "text": "The company provides cloud storage services to businesses.",
    }
]

answer = "The company was founded in 1998."

claims = [
    {
        "claim_id": "claim_0001",
        "text": "The company was founded in 1998.",
    }
]

messages = critic_prompt(question, evidence, answer, claims)

print(messages[0]["content"])
print(messages[1]["content"])
