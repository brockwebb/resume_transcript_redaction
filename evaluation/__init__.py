# evaluation/__init__.py
from pathlib import Path

# Verify we're in the correct directory structure
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Validation function used by all evaluation modules
def validate_project_root():
    """Verify the script is being run from the correct location"""
    required_dirs = ['evaluation', 'redactor', 'data']
    missing_dirs = [d for d in required_dirs if not (PROJECT_ROOT / d).is_dir()]
    
    if missing_dirs:
        raise RuntimeError(
            f"Script must be run from project root. Missing directories: {missing_dirs}\n"
            f"Current working directory: {Path.cwd()}\n"
            f"Expected project root: {PROJECT_ROOT}\n"
            f"Run script using: python -m evaluation.evaluate [args]"
        )