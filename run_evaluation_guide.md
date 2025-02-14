# Entity Detection Evaluation Guide

## Overview
The evaluation system provides tools for analyzing entity detection accuracy, validation effectiveness, and systematic patterns. This guide covers how to use the evaluation commands for different analysis needs.

## Basic Commands

### Single Test Analysis
Test a specific file for a specific entity type:
```bash
# Basic detection test
python -m evaluation.evaluate detect --test alexis_rivera --entity PERSON

# Detailed analysis mode
python -m evaluation.evaluate detect --test alexis_rivera --entity PERSON --analyze

# Analysis with pattern matching
python -m evaluation.evaluate detect --test alexis_rivera --entity PERSON --analyze --pattern-report
```

### Batch Testing
Test multiple files or entire test suite:
```bash
# Test all files for single entity type
python -m evaluation.evaluate batch --test $(ls data/test_suite/originals/*.pdf | sed 's/.*\///' | sed 's/\.pdf//') --entity PERSON

# Batch analysis with patterns
python -m evaluation.evaluate batch --test $(ls data/test_suite/originals/*.pdf | sed 's/.*\///' | sed 's/\.pdf//') --entity PERSON --analyze
```

## Analysis Options

### Output Formats
```bash
# Default text output
python -m evaluation.evaluate detect --test alexis_rivera --entity PERSON

# JSON output
python -m evaluation.evaluate detect --test alexis_rivera --entity PERSON --format json

# Detailed analysis output
python -m evaluation.evaluate detect --test alexis_rivera --entity PERSON --format detailed
```

### Analysis Modes
```bash
# Detection confidence analysis
python -m evaluation.evaluate detect --test alexis_rivera --entity PERSON --analyze confidence

# Context pattern analysis
python -m evaluation.evaluate detect --test alexis_rivera --entity PERSON --analyze patterns

# Validation rule analysis
python -m evaluation.evaluate detect --test alexis_rivera --entity PERSON --analyze validation
```

### Comparison
```bash
# Compare with previous run
python -m evaluation.evaluate compare --test alexis_rivera --compare-with previous_run --entity PERSON

# Compare with baseline
python -m evaluation.evaluate compare --test alexis_rivera --compare-with baseline --entity PERSON --analyze
```

## Common Analysis Scenarios

### Debug False Positives
```bash
# Get detailed analysis of false positives
python -m evaluation.evaluate detect --test alexis_rivera --entity PERSON --analyze --focus false-positives

# Example output:
False Positive Analysis:
- Entity: "Tableau"
  Confidence: 0.60
  Context: "...using Tableau for data visualization..."
  Likely Reason: Software/tool context detected
```

### Validation Rule Effectiveness
```bash
# Analyze validation rule impact
python -m evaluation.evaluate detect --test alexis_rivera --entity PERSON --analyze validation

# Example output:
Validation Rule Analysis:
- Rule: "title_markers"
  Applications: 12
  Average Impact: +0.2
  False Positive Prevention: 3 cases
```

### Pattern Analysis
```bash
# Analyze detection patterns
python -m evaluation.evaluate batch --test $(ls data/test_suite/originals/*.pdf | sed 's/.*\///' | sed 's/\.pdf//') --entity PERSON --analyze patterns

# Example output:
Pattern Analysis:
- Common False Positives:
  • Software Names: 8 instances
  • Company Names: 5 instances
- Context Indicators:
  • Technology Terms: 15 occurrences
  • Title Markers: 23 occurrences
```

## Tips for Effective Analysis

1. Start with Basic Detection
   - Use `detect` command first
   - Check basic metrics before detailed analysis
   - Note any obvious issues

2. Drill Down with Analysis
   - Use `--analyze` for deeper insights
   - Focus on specific issues with analysis modes
   - Compare results across multiple files

3. Pattern Investigation
   - Use batch analysis to find patterns
   - Look for systematic issues
   - Track validation rule effectiveness

4. Iterative Improvement
   - Compare results after changes
   - Track improvements with comparison tools
   - Document patterns and solutions
