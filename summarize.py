"""Summarize detection probability and false alarm rate per variant and polarization."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+")
    args = parser.parse_args()
    groups = defaultdict(list)
    seen = set()
    for path in args.metrics:
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                key = tuple(row[field] for field in (
                    "variant", "polarization", "protocol", "epochs", "hidden",
                    "seed", "requested_far", "gate_bias",
                ))
                run_id = (key, row["recording"])
                if run_id in seen:
                    parser.error(f"Duplicate run for recording {row['recording']} and configuration {key}")
                seen.add(run_id)
                groups[key].append(row)
    if not groups:
        parser.error("No completed runs found")
    print("variant,polarization,protocol,epochs,hidden,seed,requested_far,gate_bias,recordings,mean_pd,mean_far")
    for key, rows in sorted(groups.items()):
        pd = mean(float(row["pd"]) for row in rows)
        far = mean(float(row["far"]) for row in rows)
        print(",".join((*key, str(len(rows)), f"{pd:.6f}", f"{far:.6f}")))


if __name__ == "__main__":
    main()
