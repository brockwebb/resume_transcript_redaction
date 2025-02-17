Core Implementation Details:
1. Data Models
   - Entity class is frozen/immutable - must use replace() for modifications
   - Confidence values must be preserved through all transformations
   - Entity validation_rules list tracks confidence adjustments

2. Detection Flow
   - Two paths: direct evaluation vs test runner with wrapper
   - Entity filtering should happen as early as possible
   - Detector sources (spacy, presidio, ensemble) must be preserved

3. Configuration & Logging  
   - Config loaded from YAML/JSON at initialization
   - Logger debug mode affects validation tracing
   - Entity type routing defined in entity_routing.yaml
   - Validation rules in validation_params.json

4. Testing Framework
   - Test data includes ground truth annotations
   - Test runner needs config_loader for metrics
   - Results saved with unique run IDs