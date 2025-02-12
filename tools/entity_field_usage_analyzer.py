import ast
from typing import Dict, List, Set, FrozenSet
from pathlib import Path
import sys
import os

class EntityFieldAnalyzer:
    """Analyzes Entity field usage and initialization patterns."""
    
    def __init__(self):
        self.field_usages = {}
        self.initialization_patterns = {}
        self.detector_specific_fields = {}
        
    def analyze_files(self, base_path: Path):
        """Analyze Entity usage across multiple files."""
        files = [
            base_path / 'evaluation' / 'models.py',
            base_path / 'redactor' / 'detectors' / 'presidio_detector.py',
            base_path / 'redactor' / 'detectors' / 'spacy_detector.py',
            base_path / 'redactor' / 'detectors' / 'ensemble_coordinator.py'
        ]
        
        for file_path in files:
            if not file_path.exists():
                print(f"Warning: File not found: {file_path}")
                continue
                
            print(f"Analyzing {file_path}")
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
                
            self.field_usages[file_path.name] = self._analyze_field_usage(tree)
            self.initialization_patterns[file_path.name] = self._analyze_initializations(tree)
            
    def _analyze_field_usage(self, tree: ast.AST) -> Dict:
        """Analyze how Entity fields are accessed."""
        field_accesses = {}
        
        class FieldVisitor(ast.NodeVisitor):
            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name):
                    if 'entity' in node.value.id.lower():
                        field_name = node.attr
                        if field_name not in field_accesses:
                            field_accesses[field_name] = 0
                        field_accesses[field_name] += 1
                self.generic_visit(node)
                
        FieldVisitor().visit(tree)
        return field_accesses

    def _analyze_initializations(self, tree: ast.AST) -> List[Dict]:
        """Analyze Entity initialization patterns."""
        initializations = []
        
        class InitVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == 'Entity':
                    init_args = {}
                    
                    # Get positional args
                    for i, arg in enumerate(node.args):
                        if isinstance(arg, ast.Constant):
                            init_args[f'pos_arg_{i}'] = arg.value
                            
                    # Get keyword args
                    for kw in node.keywords:
                        init_args[kw.arg] = True
                        
                    initializations.append(init_args)
                self.generic_visit(node)
                
        InitVisitor().visit(tree)
        return initializations
        
    def _get_pattern_key(self, pattern: Dict) -> FrozenSet[str]:
        """Convert a pattern dict to a hashable key."""
        return frozenset(pattern.keys())
        
    def find_inconsistencies(self) -> Dict:
        """Find inconsistencies in Entity usage."""
        inconsistencies = {
            'field_mismatches': [],
            'initialization_patterns': [],
            'detector_specific': {}
        }
        
        # Compare field usage across files
        all_fields = set()
        for file_fields in self.field_usages.values():
            all_fields.update(file_fields.keys())
            
        for field in all_fields:
            files_using = []
            for filename, fields in self.field_usages.items():
                if field in fields:
                    files_using.append(filename)
            
            if len(files_using) != len(self.field_usages):
                inconsistencies['field_mismatches'].append({
                    'field': field,
                    'used_in': files_using,
                    'missing_from': [f for f in self.field_usages.keys() 
                                   if f not in files_using]
                })
                
        # Compare initialization patterns
        init_patterns = {}
        for filename, patterns in self.initialization_patterns.items():
            if not patterns:  # Skip files with no initializations
                continue
                
            # Create a hashable key from the pattern keys
            pattern_keys = frozenset(key for pattern in patterns for key in pattern.keys())
            
            if pattern_keys not in init_patterns:
                init_patterns[pattern_keys] = []
            init_patterns[pattern_keys].append(filename)
            
        if len(init_patterns) > 1:
            inconsistencies['initialization_patterns'] = [
                {'pattern': sorted(list(pattern)),
                 'files': files}
                for pattern, files in init_patterns.items()
            ]
            
        return inconsistencies
        
    def generate_report(self) -> str:
        """Generate a detailed report of findings."""
        report = []
        report.append("=== Entity Usage Analysis Report ===\n")
        
        # Field usage across files
        report.append("Field Usage by File:")
        for filename, fields in self.field_usages.items():
            report.append(f"\n{filename}:")
            for field, count in sorted(fields.items()):
                report.append(f"  {field}: {count} uses")
                
        # Initialization patterns
        report.append("\nEntity Initialization Patterns:")
        for filename, patterns in self.initialization_patterns.items():
            if not patterns:  # Skip files with no initializations
                continue
            report.append(f"\n{filename}:")
            unique_patterns = {frozenset(p.keys()) for p in patterns}
            for pattern in unique_patterns:
                report.append(f"  Fields: {sorted(pattern)}")
                
        # Inconsistencies
        inconsistencies = self.find_inconsistencies()
        if inconsistencies['field_mismatches']:
            report.append("\nField Usage Inconsistencies:")
            for mismatch in inconsistencies['field_mismatches']:
                report.append(f"\nField: {mismatch['field']}")
                report.append(f"  Used in: {', '.join(mismatch['used_in'])}")
                report.append(f"  Missing from: {', '.join(mismatch['missing_from'])}")
                
        if inconsistencies['initialization_patterns']:
            report.append("\nInitialization Pattern Inconsistencies:")
            for pattern in inconsistencies['initialization_patterns']:
                report.append(f"\nPattern: {pattern['pattern']}")
                report.append(f"Used in: {', '.join(pattern['files'])}")
                
        return "\n".join(report)

def find_project_root() -> Path:
    """Find the project root directory."""
    current = Path.cwd()
    markers = ['setup.py', 'pyproject.toml', 'requirements.txt', 'evaluation', 'redactor']
    
    while current != current.parent:
        if any((current / marker).exists() for marker in markers):
            return current
        current = current.parent
        
    raise FileNotFoundError("Could not find project root directory")

def main():
    """Main function to run analysis."""
    try:
        project_root = find_project_root()
        print(f"Project root found at: {project_root}")
        
        analyzer = EntityFieldAnalyzer()
        analyzer.analyze_files(project_root)
        print(analyzer.generate_report())
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you're running this script from within the project directory")
        sys.exit(1)

if __name__ == '__main__':
    main()