from __future__ import annotations

import argparse
import random
from pathlib import Path
import runpy
import sys

import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--script", type=Path, required=True)
parser.add_argument("args", nargs=argparse.REMAINDER)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
sys.path.insert(0, str(args.script.resolve().parent))
sys.argv = [str(args.script), *args.args[1:]]
runpy.run_path(str(args.script), run_name="__main__")
