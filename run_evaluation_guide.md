# Resume Redaction System Evaluation Guide

## Running Evaluation Tests

There are several ways to run evaluation tests depending on your needs:

### 1. Focused Entity Testing
For testing specific entity types or parameters, use the focused test runner:

```bash
python -m evaluation.focused_test_runner [options]
```

Available options:
- `--entity-type` or `-e`: Test specific entity type (e.g., EMAIL_ADDRESS, PERSON)
- `--test-ids` or `-t`: Test specific test cases
- `--confidence` or `-c`: Set confidence threshold
- `--format` or `-f`: Output format (summary or detailed)

Examples:
```bash
# Test email detection
python -m evaluation.focused_test_runner --entity-type EMAIL_ADDRESS

# Test specific cases
python -m evaluation.focused_test_runner --test-ids alexis_rivera john_smith

# Test person detection with confidence threshold
python -m evaluation.focused_test_runner --entity-type PERSON --confidence 0.8

# Get detailed output for location detection
python -m evaluation.focused_test_runner --entity-type LOCATION --format detailed
```

### 2. Single Test Case Evaluation
For evaluating a single test case with full metrics:

```bash
python -m evaluation.evaluate detect --test <TEST_ID>
```

This provides:
- Detection analysis (by entity type and detector)
- Mapping conflicts
- Missed entities
- Confidence distribution

### 3. Legacy Full Test Suite
The original test suite runner is still available but recommended only for backwards compatibility:

```bash
python -m evaluation.test_runner
```

## Output and Results

### Summary Format
The summary output provides key metrics for quick assessment:
- Precision
- Recall
- F1 Score
- Entity counts

### Detailed Format
The detailed output includes:
- Complete metrics breakdown
- False positive/negative examples
- Match details
- Confidence scores for detected entities

## Results Storage
Test results are stored in:
```
data/test_suite/results/run_<TIMESTAMP>/
```

Each run contains:
- Individual test case results
- Summary metrics
- Confidence distribution analysis
- Entity type performance breakdown

## Best Practices

1. Start with focused testing:
   - Test one entity type at a time
   - Use summary format for quick checks
   - Switch to detailed format for debugging

2. Tune confidence thresholds:
   - Start with default thresholds
   - Use --confidence flag to experiment
   - Monitor precision/recall trade-offs

3. Debug specific test cases:
   - Use --test-ids for problematic cases
   - Check detailed output
   - Compare against ground truth

4. Monitor performance:
   - Track metrics over time
   - Pay attention to false positives/negatives
   - Review confidence distributions

## Tips for Common Tasks

### Finding Missed Entities
```bash
python -m evaluation.focused_test_runner --entity-type PERSON --format detailed
```
Look for entities in the "false_negatives" section.

### Tuning Detection
```bash
# Test different confidence thresholds
python -m evaluation.focused_test_runner --entity-type EMAIL_ADDRESS --confidence 0.7
python -m evaluation.focused_test_runner --entity-type EMAIL_ADDRESS --confidence 0.8
python -m evaluation.focused_test_runner --entity-type EMAIL_ADDRESS --confidence 0.9
```

### Debugging Specific Cases
```bash
# Get detailed output for specific test
python -m evaluation.focused_test_runner --test-ids problem_case --format detailed
```
