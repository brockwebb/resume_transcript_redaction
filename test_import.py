# test_import.py
import sys
import os

print("Current working directory:", os.getcwd())
print("\nPython path:")
for path in sys.path:
    print(path)

print("\nTrying to import RedactionProcessor...")

from redactor.redactor_logic import RedactionProcessor
print("Import successful!")
