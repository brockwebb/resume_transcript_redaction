# Redaction System Debug Tools

## Overview
This suite of debugging tools is designed to help diagnose and troubleshoot entity detection issues in the resume redaction system. The tools provide granular visibility into each stage of the detection pipeline, from initial configuration to final output.

## Tools

### 1. Configuration Debugger (`tools/config_debug.py`)
Verifies and validates system configuration files.

**Purpose:**
- Ensures all required configuration files exist
- Validates configuration file structure
- Checks cross-references between configs
- Verifies detector configurations

**Usage:**
```bash
python tools/config_debug.py
```

### 2. Entity Type Debugger (`tools/entity_debug.py`)
Focuses on debugging detection for a specific entity type across the entire pipeline.

**Purpose:**
- Compares ground truth vs detected entities
- Shows detection results from each detector
- Displays confidence scores and positions
- Helps identify pattern matching issues

**Usage:**
```bash
python tools/entity_debug.py <entity_type> <test_id>
# Example:
python tools/entity_debug.py EMAIL_ADDRESS alexis_rivera
```

### 3. Presidio Detector Debug (`tools/presidio_debug.py`)
Specifically examines Presidio's entity detection behavior.

**Purpose:**
- Tests both full document and line-by-line detection
- Shows context around detections
- Displays confidence scores
- Helps identify text extraction issues

**Usage:**
```bash
python tools/presidio_debug.py <test_id>
# Example:
python tools/presidio_debug.py alexis_rivera
```

### 4. Pipeline Debugger (`tools/pipeline_debug.py`)
Traces an entity through each stage of the detection pipeline.

**Purpose:**
- Shows initial detection results
- Displays validation outcomes
- Verifies routing decisions
- Compares final vs initial results

**Usage:**
```bash
python tools/pipeline_debug.py <test_id> <entity_type>
# Example:
python tools/pipeline_debug.py alexis_rivera EMAIL_ADDRESS
```

## When to Use Each Tool

1. **Start with `config_debug.py`** when:
   - Setting up a new environment
   - After configuration changes
   - When multiple detections are failing

2. **Use `entity_debug.py`** when:
   - Investigating a specific entity type
   - Comparing against ground truth
   - Checking detection accuracy

3. **Use `presidio_debug.py`** when:
   - Presidio detections are unexpected
   - Investigating context effects
   - Text extraction seems problematic

4. **Use `pipeline_debug.py`** when:
   - Initial detection works but final output fails
   - Validation issues are suspected
   - Investigating confidence scoring

## Adding New Tests

When adding new test cases:
1. Add PDF to `data/test_suite/originals/`
2. Add YAML annotation to `data/test_suite/annotations/`
3. Update `test_manifest.yaml`

## Tips for Effective Debugging

1. **Isolate the Issue:**
   - Start with one entity type
   - Use a simple test case
   - Compare against known good results

2. **Check Each Stage:**
   - Configuration
   - Initial detection
   - Validation
   - Final output

3. **Common Issues to Look For:**
   - Threshold misconfigurations
   - Text extraction problems
   - Validation rule conflicts
   - Position mismatches

## Maintenance and Updates

When updating the system:
1. Run all debug tools against test suite
2. Check for configuration changes
3. Verify threshold settings
4. Update test cases as needed

## Contributing

When adding new debugging capabilities:
1. Follow the existing pattern separation
2. Add clear output formatting
3. Include error handling
4. Update this README

## Relationship to Testing Framework

These debug tools complement the testing framework by:
- Providing granular visibility
- Helping diagnose test failures
- Supporting test case development
- Facilitating configuration validation

The tools are designed to work alongside the regular test suite rather than replace it.