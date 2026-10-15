# ci-reliability-ml

Classifies CI failure causes from log text and estimates flaky-test risk from historical test signals.

## Data

Deterministic synthetic data is generated locally in `pipeline.py`. No private data or external download is required.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Evaluation results

With seed 42, a template-grouped holdout for CI failure causes reached 0.8333 accuracy and 0.8667 macro F1 across 72 logs. A stratified holdout for flaky-test risk reached 0.9120 accuracy, 0.9461 F1, and 0.9874 ROC AUC across 250 tests. These are measurements on synthetic data, not production CI runs.

## Limitations

The examples and generation rules are simplified. Real use would require appropriately licensed CI history and validation on data from separate time periods or projects.
