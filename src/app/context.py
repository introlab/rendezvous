import sys
import os
from pathlib import Path

rootDirectory = os.path.realpath(Path(__file__).parents[2])
sys.path.append(rootDirectory)
