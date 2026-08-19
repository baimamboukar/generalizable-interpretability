from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extension_layers_are_valid_and_well_spaced() -> None:
    extension = load("blp_extension", "extension.py")
    for config in extension.CONFIG["models"].values():
        assert config["layers"] == [16, 23, 30]
        assert config["train_layer"] == 23
        assert max(config["layers"]) < config["num_hidden_layers"]


def test_instrumentation_call_mapping_matches_upstream_order() -> None:
    instrument = load("blp_instrument", "instrument_sweep.py")
    assert [instrument.probe_name(index) for index in range(1, 10)] == [
        "tpc",
        "bilinear",
        "linear",
        "mlp_budget_1",
        "mlp_budget_2",
        "mlp_budget_3",
        "mlp_budget_4",
        "mlp_budget_5",
        "ee_mlp",
    ]
