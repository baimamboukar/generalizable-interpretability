import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))  # so geninterp.py imports from here

TASKS = HERE / "tasks"
ARTIFACTS = ROOT / "artifacts" / "reprobe"
RUNS = ARTIFACTS / "runs"
HF_BUCKET = "hf://buckets/baimamboukar/generalizable-interpretability"

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

COMPOSE = {"raw": "compose.yaml", "mcp": "compose.mcp.yaml"}
MCP_SERVER_PATH = "/work/mcp/server.py"
TOOL_TIMEOUT = 1800

DOCKER_IMAGE_ID = "{{.Id}}"
DOCKER_IMAGE_ARCH = "{{.Architecture}}"
DOCKER_VERSION = "{{.Server.Version}}"

ZERO_SCORES = {"scientific": 0, "evidence": 0, "task_pass": 0}

REPRODUCE_SCRIPT = "reproduce.sh"
CODE_SUFFIXES = (".py", ".sh")
REEXECUTION_TOLERANCE = 1e-6

INSTRUCTIONS = """
The task files are read-only under /work/task: inputs/ and reference/. Save everything under /work/output.
There is no internet access. Use only the Python packages already installed.
Where the paper or repository fixes a random seed, keep it. Otherwise seed every split, fold and fit with the SEED environment variable.

When you are done, call submit() with exactly one JSON object in this shape and nothing else:
{submission}
Replace each 0.0 and 0 with your value: results and counts are JSON numbers, not strings.
Artifact paths are lists of files relative to /work/output.
"""

REPRODUCE_INSTRUCTION = """
Also save /work/output/reproduce.sh. We will run it with bash in a fresh copy of this environment, where /work/output holds only your .py and .sh files.
It must recompute every result from /work/task and write /work/output/results.json with the same result ids as your submission.
"""
