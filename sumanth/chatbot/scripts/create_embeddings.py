#!/usr/bin/env python
"""
Script to create embeddings from documents.
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from chatbot.processors.embeddings import main

if __name__ == "__main__":
    main()

