"""Count associated fastener events from JSONL or CSV inference signals."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

# Allow the script to be launched directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fastener_counter import CounterConfig, run_counter


def read_samples(path: Path):
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    with path.open(newline="", encoding="utf-8-sig") as handle:
        yield from csv.DictReader(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("signals", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--place-threshold", type=float, default=.55)
    parser.add_argument("--tool-threshold", type=float, default=.60)
    parser.add_argument("--release-threshold", type=float, default=.45)
    args = parser.parse_args()
    result = run_counter(read_samples(args.signals), CounterConfig(
        place_threshold=args.place_threshold, tool_threshold=args.tool_threshold,
        release_threshold=args.release_threshold))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"count": result["count"], "events": len(result["events"]),
                      "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
