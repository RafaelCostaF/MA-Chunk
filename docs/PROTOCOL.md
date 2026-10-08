# Validation protocol

The hypotheses, metrics, tests and decision criteria below were written down **before** the corresponding experiments were run. Deviations are listed at the end of each part, with their reason. All results are reported, including null and unfavourable ones.

Scope: CRAG (250 questions), HotpotQA and MuSiQue (1,000 each); 5-fold cross-validation (stratified by domain on CRAG); no LLM calls during training.

## Part A. Multi-agent validation

### A.1 Hypotheses

| ID | Question | Experiment | Main metric | Criterion (α = 0.05, Holm within each family) |
|---|---|---|---|---|
| H1 | Is the marginal-contribution reward better than the alternatives? | Same environment with rewards `marginal`, `terminal`, `individual` (selfish) | Team reward `R = U − λ·tokens/1000` on test folds | marginal > terminal and > individual |
| H2 | Do selfish objectives cause a social dilemma? | individual vs cooperative | Tokens sent, efficiency loss, redundancy | selfish sends more tokens and has lower R |
| H3 | Is marginal credit faithful to each agent's contribution? | Exact Shapley values per question (2⁵ coalitions) vs marginal credit | Spearman ρ per agent × question | ρ ≥ 0.8 |
| H4 | Does orchestration matter? | Turn order: round-robin, random, confidence | R | Report difference and significance (no direction) |
| H5 | Do explicit messages help, and are they robust? | MA vs MA+msg; Gaussian noise on messages at test time, σ ∈ {0, 0.05, 0.1, 0.2, 0.5} | R vs σ | Graceful degradation; at high σ, MA+msg ≥ MA − 0.02 |
| H6 | Is the team robust to failures and noisy agents? | (a) remove each agent at test time; (b) add a random-score agent during training | R drop per removed agent; share of the random agent's messages | random agent < 5% of messages; R(6 agents) ≥ R(5) − 0.02 |
| H7 | Why RL? Is a supervised selector with the same features and objective enough? | Logistic regression / gradient boosting predicting usefulness + greedy rule sending while expected gain ≥ cost | Utility–token frontier | Reported honestly; RL must match the supervised selector to justify itself |
| H8 | Does the policy transfer across datasets? | Train on one dataset, test zero-shot on the others | R relative to in-domain | Report relative loss |
| H9 | Sensitivity | top-M ∈ {4, 8, 12}; extra seeds | R, between-seed spread | spread < 0.01 utility |
| H10 | Is the training proxy aligned with the end task? | P(correct \| U) from the end-task evaluations | Point-biserial correlation | Report (motivates the abstention gate) |

### A.2 Multi-agent metrics (every run)

Message share per agent and specialization entropy; redundancy (pairs of sent chunks with MiniLM cosine > 0.9); credit fidelity (Spearman with Shapley); utility per 100 tokens; turns to the first useful chunk; learning curves.

### A.3 Statistics

- Runs with the same dataset and seed share folds, so the unit of analysis is the **cluster** (dataset, seed): paired differences are averaged over λ ∈ {0.3, 1} within each cluster, and clusters are compared with an exact Wilcoxon signed-rank test (9 clusters for ablations, 15 for main comparisons), 95% cluster-bootstrap intervals and Holm correction within each family. With fewer than 5 clusters, the min–max range replaces the bootstrap interval.
- End-task comparisons: paired Wilcoxon per question (scores averaged over seeds).

### A.4 Deviations

- **Hypervolume → frontier AUC over the common token window.** Hypervolume with a fixed reference point rewards methods whose points span more of the token axis (static top-k reaches ≈1,200 tokens; the team ≈330 on HotpotQA), which measures range rather than quality. We report the mean of the staircase frontier over the token window covered by every method of a dataset; hypervolume remains in the CSV.
- **H7 extended.** After seeing that the supervised selector with richer features was competitive, we added (a) a feature-parity control and (b) agents with the same rich observation; the criterion for (b) was written down before running it.
- **Pseudo-replication.** The first analysis treated every (dataset, λ, seed) as independent; the final analysis uses clusters (A.3).

### A.5 Outcome (paper, Section 6)

Decomposition beats the centralized agent (15/15 clusters); marginal > terminal (9/9); selfish rewards collapse on CRAG (3–7× more tokens) and coincide with the cooperative reward on multi-hop QA; random agent ≤ 0.1% of messages; supervised selector loses with parity features (15/15) and ties with rich features; failure-aware training reduces the worst failure loss by 61–95%.

## Part B. RAGAS study (precision-aware reward and RAGAS-aligned labels)

### B.1 Measurement

- RAGAS 0.2.15 (faithfulness, answer relevancy, answer correctness, context precision, context recall) with `gpt-5-nano` as judge and the same reader for every method; RAGAS-5 = per-question mean of the five metrics (faithfulness is undefined for empty answers and excluded).
- Every method is evaluated with contexts of at most 500 characters. Two baselines (Search-R1, BM25) deliver one ≈5,000-character block; scored as a single context, their context precision is 0.54 and 0.52, re-split it is 0.29 and 0.38, so the common granularity is required.
- CRAG: all 250 questions; HotpotQA and MuSiQue: fixed random samples of 300 (`data/*/ragas_sample_ids.txt`).
- Unit: question (paired across methods, scores averaged over seeds); Wilcoxon signed-rank, Holm across datasets.

### B.2 Variants

| ID | Change w.r.t. MA+msg | Training? |
|---|---|---|
| R0 | none (reference) | — |
| R1 | same chunks, re-ordered by the highest normalized agent score before reading | no |
| R2 | confidence turn order | yes |
| R3 | reward + `β·ΔAP` (RAGAS context precision over the channel order), β = 1 | yes |
| R4 = **MA-Chunk-P** | R2 + R3 | yes |
| R5 (exploratory) | R4 with utility saturating at two useful chunks on CRAG | yes |
| R6 | R4 with utility = ½ usefulness + ½ RAGAS-attribution coverage | yes |
| R7 | R4 with utility = RAGAS-attribution coverage | yes |

### B.3 Decision rule

Accept a variant as a RAGAS improvement if RAGAS-5 is higher than R0 with p_Holm < 0.05 on at least 2 of the 3 datasets **and** the mean number of chunks does not grow by more than 10%. Budget matching: when a variant sends more chunks, its token price λ is re-chosen from chunk counts alone (training metrics, no LLM) before any evaluation.

### B.4 Amendments (written before the results they concern)

1. R5 added after the reference RAGAS measurement showed that a one-chunk oracle has context recall 0.456.
2. Budget-matched re-run of R4 on CRAG (λ ∈ {0.4, 0.5} and {1.25, 1.5}; smallest λ within +10% of R0's chunks).
3. RAGAS-aligned labels (R6, R7), preceded by a feasibility pilot: proceed only if the new labels predict the measured context recall better than the usefulness labels on at least one dataset (they did: Spearman 0.74 vs 0.55 on CRAG, 0.68 vs 0.34 on HotpotQA).

### B.5 Outcome

- No variant meets the decision rule. At a matched budget, R4 (MA-Chunk-P) improves RAGAS-5 on HotpotQA (+0.008, p_Holm = 0.008) but not on CRAG or MuSiQue; context precision rises on all three (+0.007, +0.029, +0.014; significant on HotpotQA); on CRAG it reaches R0's RAGAS-5 with 20% fewer chunks (exploratory). R6/R7 leave RAGAS-5 unchanged.
- Ceiling: an oracle sending exactly the useful evidence is only 0.010 (CRAG), 0.027 (HotpotQA) and 0.016 (MuSiQue) above R0; with a frozen reader and one to three chunks the reader and judge, not the selector, bound RAGAS-5.
- On CRAG, MA-Chunk-P (2.8 chunks) is statistically tied with ColBERTv2, Search-R1 and DeepRetrieval+FAISS (8.5–10 chunks) and ahead of FAISS, DeepRetrieval+BM25 and the prior single-agent RL policies.
