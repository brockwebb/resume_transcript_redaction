# Focused Entity Testing

## Overview
The focused test runner allows you to evaluate specific entity types or test cases with minimal output. This is particularly useful for debugging entity detection issues or tuning confidence thresholds.

## Basic Usage

Test a specific entity type:
```bash
python -m evaluation.focused_test_runner --entity-type EMAIL_ADDRESS
```

The output will show basic metrics for each test case:
```
Test ID: alexis_rivera
Precision: 0.857
Recall: 0.750
F1 Score: 0.800
```

## Command Options

- `--entity-type` or `-e`: Test specific entity type (e.g., EMAIL_ADDRESS, PERSON)
- `--test-ids` or `-t`: Test specific test cases
- `--confidence` or `-c`: Set confidence threshold
- `--format` or `-f`: Output format (summary or detailed)
- `--quiet` or `-q`: Minimize output

## Common Testing Scenarios

### Testing Specific Cases
Test one or more specific test cases:
```bash
python -m evaluation.focused_test_runner --test-ids alexis_rivera john_smith
```

### Confidence Threshold Testing
Test with a specific confidence threshold:
```bash
python -m evaluation.focused_test_runner --entity-type PERSON --confidence 0.8
```

### Detailed Analysis
Get detailed output including false positives/negatives:
```bash
python -m evaluation.focused_test_runner --entity-type LOCATION --format detailed
```

Example detailed output:
```
Test Case: alexis_rivera
Metrics:
  precision: 0.857
  recall: 0.750
  f1_score: 0.800

Matches:
  true_positives: 6
  false_positives: 1
    - "123 Main St" (0.82)
  false_negatives: 2
    - "456 Oak Ave" (0.00)
```

### Minimal Output
For scripting or CI/CD pipelines:
```bash
python -m evaluation.focused_test_runner --entity-type EMAIL_ADDRESS --quiet
```

## Tips for Effective Testing

1. Start with Individual Types
   - Test one entity type at a time
   - Use summary format for quick checks
   - Switch to detailed format for debugging

2. Tune Confidence Thresholds
   - Start with default thresholds
   - Use --confidence flag to experiment
   - Monitor precision/recall trade-offs

3. Debug Specific Cases
   - Use --test-ids for problematic cases
   - Check detailed output
   - Look for patterns in false positives/negatives

4. Batch Testing
   - Use --quiet for automated testing
   - Combine with other tools using shell scripting
   - Process output programmatically
