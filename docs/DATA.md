# Data

Everything needed to train and evaluate MA-Chunk is in [`data/`](../data). Two large files are stored as parts;
rebuild them once after cloning:

```bash
python scripts/data/split_join.py join
```

## Files

| File | Rows | Content |
|---|---|---|
| `crag/questions.parquet` | 250 | CRAG questions (50 per domain): `interaction_id`, `domain`, `question_type`, `static_or_dynamic`, `query`, `answer` (original gold answer), `clean_answer` (short gold answer used as reference by the judge and RAGAS) |
| `crag/crag_sample.parquet` (parts) | 250 | Same questions with `page_results_text`, the text of their search-result pages (input of `ma_chunk.candidates`) |
| `crag/candidates.parquet` | 7,492 | Candidate pool: union of the five agents' top-8 chunks per question. `chunk_id`, `text`, `tokens` (`cl100k_base`), `s_<agent>` (score), `rank_<agent>` (rank in the whole page, 0 = best), `n_page_chunks`, `emb` (MiniLM embedding, used for redundancy) |
| `crag/labels.parquet` | 7,492 | Usefulness labels from the LLM judge (`useful` ∈ {0, 1}: 1,475 useful; `label_model`) |
| `crag/attr_labels.parquet` | 7,492 | RAGAS-aligned attribution labels: `n_stmt` (reference sentences), `attr` (0/1 per sentence), `attr_mask`, `attr_any`, `failed` |
| `hotpotqa/candidates.parquet` | 9,953 | 1,000 FlashRAG dev questions × their 10 provided paragraphs; same columns plus `query`, `answers`, `useful` (gold supporting paragraph) and `n_support` |
| `musique/candidates.parquet` (parts) | 20,000 | 1,000 FlashRAG dev questions × top-20 BM25 paragraphs from a closed corpus of 16,301 supporting paragraphs (gold not injected) |
| `{hotpotqa,musique}/attr_labels.parquet` | 9,953 / 20,000 | RAGAS-aligned attribution labels |
| `{hotpotqa,musique}/ragas_sample_ids.txt` | 300 | Fixed random sample (seed 0) used for the multi-hop RAGAS evaluation |

The five agent scores are: `bm25` (BM25, page-normalized), `spacy` (cosine of static word vectors), `minilm` and `e5` (dense cosine), `ce` (cross-encoder, sigmoid of the logit).

## Rebuilding the data from scratch

```bash
python -m ma_chunk.candidates --top-m 8 --out data/crag/candidates.parquet            # GPU, ~45 min
python -m ma_chunk.labels --candidates data/crag/candidates.parquet --out data/crag/labels.parquet   # LLM
python -m ma_chunk.multihop_data --dataset hotpotqa --n 1000 --out data/hotpotqa/candidates.parquet
python -m ma_chunk.multihop_data --dataset musique  --n 1000 --out data/musique/candidates.parquet
python -m ma_chunk.attr_labels --dataset crag --out data/crag/attr_labels.parquet      # LLM (optional)
```

Labelling calls go through the local LLM cache, so repeated runs are free. With a different judge model the labels (and results) may differ slightly; the inter-model agreement of the CRAG usefulness labels is κ = 0.64.

## Sources and licenses

The files are derived from public datasets and keep their original licenses; check them before redistribution or commercial use:
- **CRAG** (Comprehensive RAG Benchmark, NeurIPS 2024 Datasets and Benchmarks): questions, answers and search-result pages.
- **HotpotQA** and **MuSiQue**: development sets as distributed by the FlashRAG collection (`RUC-NLPIR/FlashRAG_datasets` on Hugging Face).

Labels and scores produced by this project are released under the repository's MIT license.
