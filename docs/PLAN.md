# WaferSense Plan

Remaining work after weeks 1–2, in 31 working days (day 0 to day 30). Days are work units, not calendar dates.

## Where we stand

| Done | Result (val) |
|---|---|
| EDA, lot-based split, 64×64 two-channel maps | Ten findings, no lot leakage |
| Baseline CNN (`cnn_cosine`) | F1 macro 0.895, 3.2% missed, 1.4% false alarms |
| Autoencoder and density rule | 62–68% of defects missed |
| SimCLR, no labels | 23% missed; weak on Scratch |
| Fair checks | CNN beats SimCLR on 1% / 10% labels and on unseen Scratch |
| SimCLR fine-tune | No gain over a random start (F1 0.894) |

**Decision (day 1):** `cnn_cosine` is frozen as the classifier and the embedding. Its file hashes and val results are in `results/metrics/frozen_model.json`.

## Rules for the whole plan

- Test set stays closed until day 29 and is used once.
- Model choice: highest val F1 macro among models that miss at most 4% of defects.
- Seed noise is about 0.003 in F1 macro; smaller gaps do not count.
- Every model run reports a confusion matrix and precision, recall and F1.
- New-type results come only from models that never saw the held-out type.
- Root-cause output is a literature-based hypothesis, not ground truth.
- A day is done only when its tests pass (see Testing).

## Testing

| Level | What it checks | When it runs |
|---|---|---|
| **Unit tests** | One function, on small synthetic data | Every push (CI) |
| **Integration tests** | Several parts together, such as data → model → metrics | Every push (CI) |
| **Smoke tests** | A real script runs 1–2 epochs on a few wafers | Before each Colab batch |
| **Data checks** | Real processed data: no lot leakage, valid values, expected counts | After data changes |
| **Model checks** | Frozen model still reaches its recorded val metrics | After any model or index change |

- Unit and integration tests use `pytest` and need **no GPU, raw data or network**.
- Tests that need real data or a checkpoint are marked `@pytest.mark.data` and are skipped in CI.
- External services (Bedrock, AWS) are mocked in unit tests.
- A bug fix starts with a test that fails because of the bug.
- Tests target behavior and important logic; there is no coverage percentage goal.

## Day 0: Test foundation

| Work | Tests | Done when |
|---|---|---|
| `pytest` setup, shared fixtures, `data` marker | Small synthetic wafers and labels in `tests/conftest.py` | `pytest` runs from the repo root |
| Data | `resize_nearest`, `encode`, `augment`, lot split with `check_no_leakage` | Values, shapes and rotations verified; a leaking split fails |
| Sampling and weights | `class_weights`, `sample_labels` | Absent class gets weight 0; every class is kept |
| Models | `WaferCNN`, `SimCLR`, `random_view`, `nt_xent`, `load_encoder` | Shapes correct; matching pairs give lower loss; weights load exactly |
| Metrics | `evaluate`, `defect_vs_ok`, `operating_point` | Match hand-computed examples |
| Embedding tests | `nearest`, `similar_cases`, `anomaly`, `new_type` | Match a small example with a known answer |
| Integration | One training step on synthetic data | Loss is finite and weights change |
| CI | GitHub Actions runs `pytest` on CPU | Green check on every push |

## Week 3: Similar cases and new defect types

| Day | Work | Output | Done when |
|---|---|---|---|
| 1 | Freeze the model: name, checkpoint, embedding version in `src/config.py` | Frozen train and val embeddings | Reloaded embeddings match the saved ones |
| 2 | FAISS index over labeled train wafers | `src/features/retrieval.py` | precision@5 matches the current 0.866 |
| 3 | Retrieval review: precision at k = 1, 5, 10, 20; how often neighbors share a lot; each neighbor shown with its lot | Query + neighbors figures with lot names, failure notes | Weak types (Loc, Scratch) explained; k and a per-lot neighbor limit chosen |
| 4 | Leave-one-type-out CNNs for the other 7 defect types (two Colab sessions) | 8 held-out models with `cnn_noscratch` | All runs on Drive |
| 5 | Compare new-type scores: max softmax, energy, kNN distance, Mahalanobis | `src/openset/scores.py`, AUROC per type | One score and threshold chosen on val |

**Tests:** FAISS matches brute-force search on random vectors; a wafer finds itself first; the index never returns val or test wafers; the per-lot limit is respected and still returns k neighbors; each score ranks a known outlier highest; held-out runs contain no wafer of the held-out type. **Model check:** frozen checkpoint reproduces F1 0.895.

## Week 4: Active learning and model review

The 638K unlabeled wafers have no true labels, so active learning is **simulated**: labeled train wafers start hidden and are revealed when selected.

| Day | Work | Output | Done when |
|---|---|---|---|
| 6 | Simulation loop: start from 1% of labels, fixed val | `src/active/simulate.py` | Random selection reproduces the 1% baseline |
| 7 | Strategies: random, entropy, margin, diversity (k-center) | `src/active/strategies.py` | All run on a small subset |
| 8 | Full simulation, 3 seeds | F1 vs labels-used curves | Labels needed to reach F1 0.85 per strategy |
| 9 | Label check: wafers whose neighbors and model disagree with their label | Ranked list and figures of likely wrong labels | Share of suspect labels estimated |
| 10 | **Model review gate** (see below) | Written go / no-go | Decision recorded here |

**Tests:** a wafer is never selected twice; labels stay hidden until selected; the budget is respected; same seed gives the same selection; entropy and margin pick the known most uncertain sample; k-center picks spread-out points; val wafers never enter the pool.

## Week 5: Model v2 (only if the gate says go) and explainability

| Day | Work | Output | Done when |
|---|---|---|---|
| 11 | Add a metric-learning loss (supervised contrastive or ArcFace) to the CNN | Training option in `train_cnn` | Smoke test passes |
| 12 | Train on Colab, 2 seeds; run the 4 embedding tests | Metrics and confusion matrices | Compared with `cnn_cosine` |
| 13 | Keep or reject v2. If kept: re-freeze, rebuild index, redo thresholds | Updated frozen model | Week 3–4 checks still pass |
| 14 | Grad-CAM for the frozen model | `src/features/explain.py` | Heatmaps for every defect type |
| 15 | Check the heatmaps: remove the highlighted dies and measure the score drop | Faithfulness result | Heatmaps beat random removal |

If the gate says no-go, days 11–13 move the rest of the plan forward.

**Tests:** the new loss is lower for well-separated classes and its gradients are finite; the old training path gives the same result when the option is off; Grad-CAM output matches the wafer size, stays in 0–1 and changes when the target class changes. **Model check (if v2 is kept):** rerun all embedding tests and rebuild the index.

## Week 6: Lot-level detection and yield

| Day | Work | Output | Done when |
|---|---|---|---|
| 16 | Normal per-lot defect rates by type | Baseline statistics | Baseline built from train lots only |
| 17 | Excursion detector: binomial test and CUSUM across lots | `src/features/excursion.py` | Runs on all val lots |
| 18 | Evaluate with **injected** excursions, since real ones are not labeled | Detection rate vs false alerts | Operating point chosen |
| 19 | Yield impact: dies lost per wafer and per lot | `src/features/yield_impact.py` | Totals match the die counts |
| 20 | One `analyze_wafer()` report combining all features, plus **similar-lot search**: each lot summarized as one vector (mean embedding and defect-type mix), nearest past lots returned | Structured JSON per wafer; `similar_lots()` | Works end to end on val wafers; a val lot's nearest past lots share its main defect type |

**Tests:** normal synthetic lots raise no alert at the expected rate; an injected excursion is detected; empty and one-wafer lots do not crash; dies lost equal a hand-counted map; `analyze_wafer()` output follows a fixed schema; a lot vector does not depend on wafer order; similar-lot search returns only train lots and never the query lot. **Integration:** wafer in, complete report out.

## Week 7: LLM agent

| Day | Work | Output | Done when |
|---|---|---|---|
| 21 | Root-cause knowledge base from literature, with sources | Pattern-to-cause table | Every entry has a citation |
| 22 | Agent tools: classify, similar cases, new-type check, lot history, yield | `src/agent/tools.py` | Each tool tested alone |
| 23 | Agent on Claude via AWS Bedrock | `src/agent/agent.py` | Full report for one wafer |
| 24 | Evaluate on a fixed case set: grounding, citations, correct tool use | Scored results | Unsupported claims measured |
| 25 | Guardrails: uncertainty wording, cost and latency limits | Final agent | Says "unknown" on unseen patterns |

**Tests:** each tool validates its input and output; a mocked LLM drives the tool loop; tool errors and Bedrock timeouts are handled; the step limit stops endless loops; every cause in a report has a source from the knowledge base. **Evaluation:** a small fixed case set runs with the real model before each agent change.

## Week 8: Product and final evaluation

| Day | Work | Output | Done when |
|---|---|---|---|
| 26 | FastAPI service, MLflow run tracking | `src/api/` | `/analyze` returns a report |
| 27 | Streamlit interface for engineers | Demo app | Full flow from upload to report |
| 28 | Docker and AWS (ECR, ECS Fargate, S3) | Deployed service | Public health check passes |
| 29 | Recheck model limits on val, then **one test-set evaluation** | Final metrics | Reported as measured, with no retuning |
| 30 | README, deck page numbers, demo video | Portfolio-ready project | A new reader can follow and run it |

**Tests:** API success and error responses, including invalid wafer maps; the Docker image builds and `/health` responds in CI; a post-deployment smoke test calls the live service. **Before day 29:** all tests and model checks must pass, because the test set is opened only once.

## Model review gate (day 10)

Improve the original model **once**, after the features show what is actually weak.

**Go if at least one is true:**
- Retrieval or new-type detection fails on a type because of the embedding.
- Active learning or the label check shows a clear, fixable weakness.

**Candidates, in order:**
1. Fix or remove wrong labels found on day 9, then retrain.
2. Metric-learning loss, which directly targets the embedding.
3. Augmentation aimed at thin patterns such as Scratch.

**v2 replaces `cnn_cosine` only if**, over 2 seeds, it misses at most 4% of defects and either:
- raises val F1 macro by at least 0.01, or
- raises new-type AUROC by at least 0.03 with no F1 loss beyond noise.

**Limit:** 3 days. If v2 does not pass, `cnn_cosine` stays.
