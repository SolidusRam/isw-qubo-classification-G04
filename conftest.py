import sys
import os

# Add src/ to sys.path so that 'qubo_project' is importable when running pytest
# from the project root, without requiring installation or absolute paths.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
