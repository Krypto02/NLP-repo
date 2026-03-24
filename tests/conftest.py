"""Configure sys.path so test files can import src modules directly."""

import sys
import os

# Flask app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "backend", "flask_app"))
# Scripts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "backend", "scripts"))
