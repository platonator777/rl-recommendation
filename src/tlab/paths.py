"""Explicit runtime workspace, separate from immutable report evidence."""
import os
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[2]

def workspace_root():
    return Path(os.environ.get('TLAB_WORKSPACE', str(BUNDLE / 'work'))).resolve()
