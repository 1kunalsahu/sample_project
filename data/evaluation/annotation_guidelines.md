# FiQA adaptation and annotation notes

The initial benchmark is BEIR FiQA. Its test qrels provide query to relevant corpus passage judgments, which are used for retrieval evaluation. The qrels are not verified answer text, gold facts, answerability judgments, or claim-level support annotations.

For claim faithfulness evaluation, annotate each generated atomic claim independently against the exact retrieved evidence shown to the model. Record `SUPPORTED`, `UNSUPPORTED`, or `CONTRADICTED`; lack of mention is not contradiction. Record supporting passage IDs and whether the attached citation supports the claim. Use two annotators and adjudicate disagreements before reporting agreement or critic quality.

Do not infer human labels from critic output. Do not treat an empty answer as a correct answer. Keep answer correctness separate from evidence faithfulness.
