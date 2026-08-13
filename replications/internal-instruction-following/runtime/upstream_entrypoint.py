from __future__ import annotations

import sys
from pathlib import Path


def adapted_source(path: Path) -> str:
    source = path.read_text()
    original = """    target_layers = []
    i = num_layers - 1
    leap = num_layers // 5
    while i > 10:
        target_layers.append(i)
        i = i - leap
"""
    replacement = """    configured_layers = os.environ.get(\"GI_TARGET_LAYERS\")
    if configured_layers:
        target_layers = [int(layer) for layer in configured_layers.split(\",\")]
    else:
        target_layers = []
        i = num_layers - 1
        leap = num_layers // 5
        while i > 10:
            target_layers.append(i)
            i = i - leap
"""
    if original not in source:
        raise RuntimeError("Pinned upstream layer-selection block changed")
    return source.replace(original, replacement, 1)


def main() -> None:
    source = Path(sys.argv[1]).resolve()
    sys.argv = [str(source), *sys.argv[2:]]
    namespace = {"__file__": str(source), "__name__": "__main__"}
    exec(compile(adapted_source(source), str(source), "exec"), namespace)


if __name__ == "__main__":
    main()
