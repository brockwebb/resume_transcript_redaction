To run a full evaluation test, including recall scores and other metrics, follow these steps:

### **Option 1: Using `evaluation.evaluate` for a single test case**
If you want to evaluate a single test case, use:
```bash
python -m evaluation.evaluate detect --test <TEST_ID>
```
This runs the `StageOneEvaluator` from `evaluate.py`, which provides:
- Detection analysis (by entity type and detector)
- Mapping conflicts
- Missed entities
- Confidence distribution

However, this approach does not aggregate results across multiple tests.

---

### **Option 2: Using `test_runner.py` for a full test suite evaluation**
To run a comprehensive evaluation across multiple test cases and generate recall scores:
1. Ensure you are in the project root directory.
2. Run:
   ```bash
   python -m evaluation.test_runner
   ```
   This script:
   - Loads all test cases from `data/test_suite/test_manifest.yaml`
   - Compares detected entities against ground truth annotations
   - Computes precision, recall, and F1-score using `evaluation.metrics.entity_metrics`
   - Saves detailed results in `data/test_suite/results/run_<TIMESTAMP>/`

To compare different test runs, you can use:
```bash
python -m evaluation.test_runner compare <RUN_ID1> <RUN_ID2>
```
This provides differences in recall, precision, and F1-score.

