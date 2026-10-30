# CriticRAG

Research code for a controlled comparison of standard RAG and RAG with an inference-time critic. The project documentation defines the study; this repository implements core retrieval, prompting, claim review, and evaluation components. It does not claim critic-based RAG is a new architecture.

## Dataset

The initial benchmark is the [BEIR FiQA dataset](https://huggingface.co/datasets/BeIR/fiqa), originating from the [FiQA 2018 challenge](https://sites.google.com/view/fiqa/home). It supplies finance-domain questions, a corpus of community text passages, and relevance judgments. This is suitable for retrieval benchmarking. It does **not** supply the verified answer strings, claim labels, unanswerable questions, or multi-hop/category labels required for all planned analyses. Those results stay unavailable until separately annotated; no labels are fabricated.

### 1. Run hybrid retrieval and Retrieval Recall@k

Measure hybrid retrieval (BM25 + dense embeddings, fused with RRF and reranked) against the official test qrels:

```powershell
python -m experiments.run_baseline --download --split test --k 5
```

The retrieval report is written to `results/metrics/fiqa_retrieval.json` and distinguishes evidence-set recall from query-level evidence hit rate. To get enough retrieved passages to calculate Recall@3, @5, and @10 from one run, retrieve 10:

```powershell
python -m experiments.run_baseline --split test --k 10 --limit 10 --output results/metrics/fiqa_hybrid_pilot.json
python -m experiments.calculate_metrics --input results/metrics/fiqa_hybrid_pilot.json --k 3 5 10
```

`--limit N` runs a pilot subset. Omit it to use the full selected split. If the saved retrieval output contains fewer than k passages, the calculator marks Recall@k as undefined for that k.

The benchmark uses relevance judged passages, not strict evidence-set gold for every answer. Recall figures should therefore be described as retrieval against BEIR qrels. Review the dataset's original terms before redistributing downloaded data.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Install dense retrieval and cross-encoder reranker dependencies:

```powershell
python -m pip install -e ".[retrieval]"
```

Hybrid retrieval requires downloading the configured sentence-transformers embedding and reranker models plus a compatible PyTorch runtime. The default config is CPU-compatible, though a GPU can speed up indexing. Passage vectors are cached under `.cache/criticrag/` and reused when the corpus and embedding model are unchanged. Set `reranker_model: null` in `configs/retrieval.yaml` or pass `--reranker-model none` to skip reranking while retaining BM25 + dense + RRF.

## Components

- `ingestion/fiqa.py`: download and normalize BEIR FiQA questions, corpus, and qrels.
- `ingestion/chunker.py`: deterministic text chunks with source offsets.
- `retrieval/`: BM25, dense embeddings, RRF fusion, and cross-encoder reranking; both experiment runners use this hybrid path.
- `pipeline/retrieval_cache.py`: retrieval configuration fingerprinting for paired conditions.
- `generation/`: OpenAI-compatible generation adapter and sentence-level claim candidates.
- `critic/`: evidence-bounded critic schema validation, citation ID checks, and conservative unsupported-claim filtering.
- `evaluation/`: retrieval, claim, citation, and paired binary metrics.
- `experiments/`: entry points and condition registry.

### 2. Run generation and critic review

Inference defaults to your local Ollama server. Install/pull models with Ollama, then set the roles in `configs/models.yaml`:

```yaml
backend: ollama
base_url: http://localhost:11434
generator: llama3.2:latest
critic: llama3.2:latest
evaluator: llama3.2:latest
```

Check local models with `ollama list`. If a model is absent, fetch it with `ollama pull llama3.2`. Ollama serves its local chat API at `http://localhost:11434`; the project uses `/api/chat` with streaming disabled. No API key or paid cloud inference is needed. You can set `OLLAMA_HOST` if your server uses a different address. `CRITICRAG_MODEL`, `CRITICRAG_CRITIC_MODEL`, and `CRITICRAG_EVALUATOR_MODEL` can override the YAML names in the current shell. `.env.example` is an example only; this project does not load it automatically.

The OpenAI-compatible backend remains available by setting `backend: openai_compatible` in the YAML and supplying `CRITICRAG_API_BASE`, `CRITICRAG_API_KEY`, and `CRITICRAG_MODEL` environment variables.

Run a small baseline pilot and a critic pilot on the same first five test questions. Both use the same BM25 + dense + RRF + reranker retrieval configuration:

```powershell
python -m experiments.run_rag --condition baseline --limit 5 --k 10
python -m experiments.run_rag --condition critic --limit 5 --k 10
```

Each command prints a new JSONL output path under `results/raw/`. The baseline condition keeps the generated answer; it still runs a claim reviewer to obtain comparable metric labels. Set `CRITICRAG_EVALUATOR_MODEL` to the same evaluator for both runs. If unset, the generator model is also used as evaluator, which is a weaker self-evaluation setup. Use `--limit 0` to process the full 648-question split; this makes two model calls per question and can incur substantial latency and API cost.

### 3. Calculate the requested metrics

Use the exact JSONL path printed by `run_rag`:

```powershell
python -m experiments.calculate_metrics --input results/raw/critic_fiqa_test_YYYYMMDDTHHMMSSZ.jsonl --k 3 5 10
```

The summary is saved to `results/metrics/aggregate_metrics.json` (choose a different `--output` path to preserve multiple runs). Hallucination rate is `(UNSUPPORTED + CONTRADICTED claims) / evaluable claims`. Citation faithfulness is the fraction of **cited** claims whose verdict is `SUPPORTED` and whose cited chunk appears in the evaluator's `supporting_evidence_ids`. Retrieval Recall@k is computed against FiQA's relevant passage IDs. If claims or citations are absent, the corresponding metric is `null`/undefined; they are never silently counted as zero.

These claim and citation metrics are automated evaluator estimates, not human-validated findings. For research claims, annotate a sample independently and report agreement. FiQA retrieval qrels do not provide answer correctness or human claim verdicts.

The callable interfaces are also available in `pipeline.baseline.run_baseline` and `pipeline.critic_rag.run_critic_rag`.

## Status and limitations

Implemented: reusable retrieval and model interfaces, FiQA qrel normalization, claim-level metric functions, validated critic JSON parsing, and configuration-fingerprinted in-memory retrieval cache.

Not completed: full paired generation runner, persistent cache, cross-model orchestration, human annotation and adjudication workflow, reference-answer correctness scoring, statistical analysis beyond paired binary outcomes, and full-dataset model experiments. Experiment status is tracked in `experiments/experiment_registry.yaml`. No model outputs, annotations, or research findings are included.

The default retrieval configuration uses hybrid BM25/dense retrieval, RRF fusion, and the `BAAI/bge-reranker-base` cross-encoder. Override the model or disable the reranker in `configs/retrieval.yaml`. Retrieval metadata and candidate scores/ranks are recorded in the result outputs.

## Useful CLI help

```powershell
python -m experiments.run_baseline --help
python -m experiments.run_critic --help
python -m experiments.model_comparison --help
python -m experiments.ablations --help
```

See `data/evaluation/annotation_guidelines.md` before collecting claim labels. Keep raw model outputs and human annotations separate from derived metric summaries.

#

# Run the baseline for 100 questions

python -m experiments.run_rag --condition baseline --limit 100 --k 5

# Calculate baseline metrics

python -m experiments.calculate_metrics --input "results/raw/baseline_fiqa_test_20261019T231151Z.jsonl" --k 3 5 10 --output results/metrics/baseline_100_metrics.json

# Run the critic for the same 100 questions

python -m experiments.run_rag --condition critic --limit 100 --k 5

# Calculate critic metrics

python -m experiments.calculate_metrics --input "results/raw/PASTE_THE_REAL_CRITIC_FILENAME.jsonl" --k 3 5 10 --output results/metrics/critic_100_metrics.json
#   c r i t i c R A G  
 #   c r i t i c R A G  
 