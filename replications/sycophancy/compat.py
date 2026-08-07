from __future__ import annotations

import json
from pathlib import Path


def prepare(tree: Path, model: str, commit: str) -> None:
    if model != "llama-3.2":
        return
    marker = tree / ".gi-compat.json"
    path = tree / "probe" / "extract_activation.py"
    anchor = "def extract_residual_activation"
    replacement = (
        "layer_output = states[0][0] if isinstance(states[0], tuple) else states[0]\n"
        "            layer_hidden_state = layer_output[:, last_index+1:-1, :]"
    )
    if marker.exists():
        if replacement not in path.read_text().split(anchor, 1)[1]:
            raise RuntimeError("Llama residual compatibility patch is invalid")
        return
    source = path.read_text()
    head, tail = source.split(anchor, 1)
    old = "layer_hidden_state = states[0][0][:, last_index+1:-1, :]"
    if old not in tail:
        raise RuntimeError("Released Llama residual extractor no longer matches")
    path.write_text(head + anchor + tail.replace(old, replacement, 1))
    marker.write_text(
        json.dumps(
            {
                "upstream_commit": commit,
                "reason": "accept tensor or tuple decoder-layer hook outputs",
                "file": "probe/extract_activation.py",
            },
            indent=2,
        )
        + "\n"
    )
