import sys
import os
from pathlib import Path

rootDirectory = str(Path(__file__).resolve().parents[2])
sys.path.append(rootDirectory)
