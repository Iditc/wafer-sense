# WaferSense

## Goal
End-to-end wafer map anomaly detection system for semiconductor manufacturing:
detect defect patterns, discover new (unseen) defect types, flag tool excursions
at lot level, and let an LLM agent propose root-cause hypotheses for engineers.
Portfolio project aimed at companies like Applied Materials.

## Data
- WM-811K (`LSWMD.pkl`, real fab wafer maps). Fields used: waferMap, failureType,
  lotName, waferIndex, dieSize. Only ~20% labeled (9 patterns incl. "none").
- Optional later: MixedWM38 (mixed-type defects).
- Raw data in `data/raw/`, processed parquet in `data/processed/` (never committed).

## Key Design Decisions
- Split train/val/test by `lotName` (never random) to avoid leakage.
- Primary metric: F1 macro. Always show confusion matrix + precision/recall/F1.
- Open-set: hold out defect classes to simulate unseen defects; report AUROC.
- Root-cause mapping is a literature-based hypothesis, not ground truth.

## Tech Stack
PyTorch, scikit-learn, FAISS, MLflow, FastAPI, Streamlit, Claude via AWS Bedrock,
Docker, AWS (ECS Fargate, S3, ECR). Training on Colab GPU; code lives in `src/`.

## Coding Conventions
- All comments, docstrings and docs in English.
- Each source file < 300 lines; small single-purpose functions with docstrings.
- Use pathlib; all paths from `src/config.py`.
- Do not reload the raw pickle if processed parquet already exists.
- Notebooks only launch training on Colab; no logic in notebooks.

## Git Workflow Rules
- After every working step, tell the user to commit with exact file names.
- Never use `git add .` or `git add -A`.
- Always remind to push.
