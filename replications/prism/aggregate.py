from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGETS = ROOT / "data" / "results" / "prism" / "baseline_targets.csv"
OUTPUT = ROOT / "data" / "results" / "prism"
KEYS = ("dataset", "model", "method", "representation", "test_domain", "metric")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[name] for name in KEYS)


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def aggregate(results: Path) -> None:
    observed: dict[tuple[str, ...], dict[str, str]] = {}
    for path in sorted(results.glob("*_summary.csv")):
        for row in read_rows(path):
            row_key = key(row)
            if row_key in observed:
                raise RuntimeError(f"Duplicate result: {row_key}")
            observed[row_key] = row

    rows = read_rows(TARGETS)
    targets = {key(row): row for row in rows}
    matched = 0
    for row_key, result in observed.items():
        target = targets.get(row_key)
        if target is None:
            rows.append(result)
            continue
        value = float(result["replication_value"])
        paper = float(target["paper_value"])
        target["replication_value"] = f"{value:.6f}"
        target["delta"] = f"{value - paper:.6f}"
        target["status"] = "completed"
        matched += 1

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_rows(OUTPUT / "mega_table.csv", rows)
    write_rows(
        OUTPUT / "baseline_summary.csv",
        [
            row
            for row in rows
            if row["test_domain"] == "average" and row["paper_value"]
        ],
    )
    write_rows(
        OUTPUT / "extension_summary.csv",
        [
            row
            for row in rows
            if row["test_domain"] == "average" and not row["paper_value"]
        ],
    )
    write_rows(
        OUTPUT / "cross_model_summary.csv",
        [row for row in rows if row["test_domain"] == "average"],
    )
    print(
        f"baseline={matched}/{len(targets)} "
        f"extensions={len(observed) - matched} total={len(rows)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    aggregate(parser.parse_args().results)
