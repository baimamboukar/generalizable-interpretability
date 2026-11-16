"""Run one study in a sealed container with Inspect.

    uv run --extra reprobe inspect eval src/reprobe/reprobe.py -T study=lookback-lens -T arm=raw -T model=grok-4.5

Add --epochs 3 to repeat a run three times. INSPECT_EVAL_MAX_SANDBOXES=1 in .env runs the repeats one after another.
"""

import json
import time
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.event import ModelEvent
from inspect_ai.log import transcript
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, mean, scorer
from inspect_ai.tool import bash, mcp_server_sandbox, text_editor, tool
from inspect_ai.util import sandbox

from constants import (ARTIFACTS, COMPOSE, HERE, INSTRUCTIONS, MCP_SERVER_PATH, REPRODUCE_INSTRUCTION, RUNS, TASKS,
                       TOOL_TIMEOUT, ZERO_SCORES)
from geninterp import ARTIFACT_GROUPS, benchmark_record, benchmark_task, score_submission
from helpers import copy_outputs, load_yaml, now, openrouter_cost, parse_submission, register_cost, save_json, started
from manifest import build_manifest
from reexecute import reexecute

STUDIES = load_yaml("studies.yaml")
MODELS = load_yaml("models.yaml")


def submission_template(task_id: str) -> dict:
    public = benchmark_task(task_id)
    result_ids = public["submission_contract"]["result_ids"]
    counts = public.get("expected_counts", {})

    return {
        "task_id": task_id,
        "results": dict.fromkeys(result_ids, 0.0),
        "protocol": {
            "result_count": len(result_ids),
            "scope": "<what you ran>",
            "input_hash": "<sha256 of the inputs you used>",
            "counts": dict.fromkeys(counts, 0),
        },
        "artifacts": {group: ["<path>"] for group in ARTIFACT_GROUPS if group != "trace"},
        "audit_findings": ["<one sentence per protocol problem you found>"],
    }


def prompt(study: str, arm: str) -> str:
    task_text = (TASKS / study / "TASK.md").read_text()
    template = json.dumps(submission_template(STUDIES[study]["task_id"]), indent=2)
    instructions = INSTRUCTIONS.format(submission=template)
    reproduce = REPRODUCE_INSTRUCTION if arm == "raw" else ""
    return f"{task_text}\n{instructions}{reproduce}"


def configured_model(name: str):
    settings = dict(MODELS[name])
    model_id = settings.pop("id")
    register_cost(f"openrouter/{model_id}", openrouter_cost(model_id))
    return get_model(f"openrouter/{model_id}", config=GenerateConfig(**settings))


@tool
def read_file():
    async def execute(path: str, start_line: int = 1, end_line: int = 150):
        """Read lines of a text file or a PDF inside /work. Read long files a range of lines at a time.

        Args:
            path: Absolute path of the file.
            start_line: First line to read.
            end_line: Last line to read.
        """
        if path.endswith(".pdf"):
            result = await sandbox().exec(["pdftotext", "-layout", path, "-"])
            text = result.stdout or result.stderr
        else:
            text = await sandbox().read_file(path)

        lines = text.splitlines()
        shown = "\n".join(lines[start_line - 1:end_line])
        return f"Lines {start_line} to {min(end_line, len(lines))} of {len(lines)}:\n{shown}"
    return execute


@tool
def list_files():
    async def execute(path: str):
        """List the files under a folder inside /work.

        Args:
            path: Absolute path of the folder.
        """
        result = await sandbox().exec(["find", path, "-maxdepth", "3", "-not", "-path", "*/.git/*"])
        return result.stdout or result.stderr
    return execute


def arm_tools(arm: str) -> list:
    if arm == "raw":
        return [bash(timeout=TOOL_TIMEOUT), text_editor()]
    server = mcp_server_sandbox(name="study", command="python", args=[MCP_SERVER_PATH], timeout=TOOL_TIMEOUT)
    return [server, read_file(), list_files()]


def measure_efficiency(state, events: list) -> dict:
    model_calls = [e for e in events if isinstance(e, ModelEvent)]
    api_time = sum(call.working_time or 0 for call in model_calls)
    wall_time = (now() - started(events)).total_seconds()

    return {
        "agent_cost_usd": round(state.cost_usage or 0, 4),
        "api_time_seconds": round(api_time, 1),
        "wall_time_seconds": round(wall_time, 1),
        "mcp_build_time_seconds": 0,
        "human_interventions": 0,
    }


def grade(state, output: Path, efficiency: dict) -> tuple[dict | None, str]:
    try:
        submission = parse_submission(state.output.completion)
        submission.setdefault("artifacts", {})["trace"] = ["trace.json"]
        submission["efficiency"] = efficiency
        return score_submission(benchmark_record(state.metadata["task_id"]), submission, output), ""
    except ValueError as error:
        return None, f"No valid submission: {error}"


def summarize(result: dict | None, error: str) -> tuple[dict, str]:
    if not result:
        return ZERO_SCORES, error

    scientific, evidence = result["scientific"], result["evidence"]
    scores = {"scientific": scientific["score"], "evidence": evidence["score"], "task_pass": int(result["task_pass"])}
    explanation = (f"{scientific['matched']} of {scientific['total']} values within tolerance, "
                   f"{evidence['present']} of {evidence['required']} artifact groups")
    return scores, explanation


def check_reexecution(output: Path, run: Path, config: dict, result: dict | None) -> dict:
    if config["arm"] != "raw":
        return {"status": "not applicable: the study's tool computed the results"}
    if not result:
        return {"status": "no valid submission"}

    details = result["scientific"]["details"]
    submitted = {d["id"]: d["actual"] for d in details if isinstance(d["actual"], (int, float))}
    return reexecute(output, run, config, submitted)


def new_run_folder(state) -> Path:
    model_name = str(state.model).split("/")[-1]
    run = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}-{state.sample_id}-{model_name}-repeat{state.epoch}"
    run.mkdir(parents=True)
    return run


@scorer(metrics={"scientific": [mean()], "evidence": [mean()], "task_pass": [mean()]})
def benchmark():
    async def score(state, target):
        run = new_run_folder(state)
        output = await copy_outputs(run)
        save_json(output / "trace.json", [message.model_dump(mode="json") for message in state.messages])

        events = transcript().events
        efficiency = measure_efficiency(state, events)
        result, error = grade(state, output, efficiency)
        scores, explanation = summarize(result, error)
        reexecution = check_reexecution(output, run, state.metadata, result)

        if result:
            save_json(run / "score.json", result)
        save_json(run / "manifest.json", await build_manifest(state, events, efficiency, scores, reexecution))

        return Score(value=scores, explanation=f"{explanation}; re-execution {reexecution['status']}; saved to {run}", metadata=result)
    return score


def run_config(study: str, arm: str, model: str | None, budget: float | None) -> dict:
    settings = STUDIES[study]

    return {
        "study": study,
        "arm": arm,
        "task_id": settings["task_id"],
        "model": model,
        "model_settings": MODELS.get(model),
        "budget": budget or settings["budget"][arm],
        "time_limit": settings["time_limit"],
        "memory": settings["memory"],
        "seed": settings["seed"],
        "task_dir": str(ARTIFACTS / study),
        "mcp_server": str(HERE / settings["mcp_server"]),
    }


@task
def reprobe(study: str = "lookback-lens", arm: str = "raw", model: str | None = None, budget: float | None = None):
    assert arm in COMPOSE, "arm must be raw or mcp"
    assert model is None or model in MODELS, f"model must be one of {', '.join(MODELS)}"
    config = run_config(study, arm, model, budget)

    return Task(
        dataset=[Sample(id=f"{study}-{arm}", input=prompt(study, arm), metadata=config)],
        solver=react(tools=arm_tools(arm)),
        scorer=benchmark(),
        sandbox=("docker", str(HERE / COMPOSE[arm])),
        time_limit=config["time_limit"],
        model=configured_model(model) if model else None,
        cost_limit=config["budget"] if model else None,
    )
