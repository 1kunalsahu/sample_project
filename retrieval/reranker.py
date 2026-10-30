from __future__ import annotations


class CrossEncoderReranker:
    """Rerank the RRF candidate set with a sentence-transformers cross encoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        try:
            from sentence_transformers import CrossEncoder
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "Install criticrag[retrieval] to enable the cross-encoder reranker"
            ) from exc

        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Reranker] Using device: {self.device}")

        self.model_name = model_name

        try:
            local_model = snapshot_download(
                repo_id=model_name,
                local_files_only=True,
            )

            self.model = CrossEncoder(
                local_model,
                local_files_only=True,
                device=self.device,
            )

        except Exception:
            # If it is not cached, permit the normal Hugging Face download path.
            self.model = CrossEncoder(
                model_name,
                device=self.device,
            )

    def rerank(
        self,
        question: str,
        candidates: list[dict],
    ) -> list[dict]:

        if not candidates:
            return []

        pairs = [
            (question, item["text"])
            for item in candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        ranked = []

        for item, score in zip(candidates, scores):
            ranked.append(
                {
                    **item,
                    "retrieval_score": float(score),
                    "rrf_score": item["retrieval_score"],
                }
            )

        return sorted(
            ranked,
            key=lambda item: (
                -item["retrieval_score"],
                item["chunk_id"],
            ),
        )