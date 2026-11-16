"""
Compares a new run of ML_Models_Saved/Model_Benchmark.py with the 2019 results in
Results/Test_Results.xlsx, model by model, on the AVG row (the mean accuracy over the 12
test partitions).

    python scripts/check_2019_results.py --expected Results/Test_Results.xlsx \
        --reproduced sklearn.json cnn.json --out check_2019.json

The partitions hold 52 images each, so AVG is the accuracy on all 624 test images and does
not depend on how the images were shuffled into partitions.
"""

import argparse
import json
import sys

import openpyxl


def read_avg_row(path):
    """{model name: AVG value} from the Results sheet written by Model_Benchmark.py."""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(workbook["Results"].iter_rows(values_only=True))
    header = rows[0]
    avg = next(row for row in rows if row[0] == "AVG")
    return {name: float(value) for name, value in zip(header[1:], avg[1:], strict=True)}


def compare(expected, reproduced, tolerance=1e-6):
    models = {}
    for name, expected_avg in expected.items():
        found = reproduced.get(name)
        if found is None:
            models[name] = {"expected_avg": expected_avg, "reproduced": False, "missing": True}
            continue
        n = found["test_images"]
        avg = found["avg_partition_accuracy"]
        models[name] = {
            "expected_avg": expected_avg,
            "reproduced_avg": avg,
            "difference": avg - expected_avg,
            "expected_correct": round(expected_avg * n),
            "reproduced_correct": found["correct"],
            "test_images": n,
            "reproduced": abs(avg - expected_avg) <= tolerance,
        }
    return {
        "tolerance": tolerance,
        "all_reproduced": all(model["reproduced"] for model in models.values()),
        "models": models,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare a benchmark rerun with Test_Results.xlsx."
    )
    parser.add_argument("--expected", required=True, help="the 2019 Test_Results.xlsx")
    parser.add_argument(
        "--reproduced",
        nargs="+",
        required=True,
        help="--metrics-json files written by Model_Benchmark.py",
    )
    parser.add_argument("--out", required=True, help="JSON file for the comparison")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="largest accepted difference of the AVG values (default: 1e-6)",
    )
    parser.add_argument(
        "--strict", action="store_true", help="exit with status 1 if a model differs"
    )
    args = parser.parse_args(argv)

    reproduced = {}
    for path in args.reproduced:
        with open(path) as f:
            reproduced.update(json.load(f))
    result = compare(read_avg_row(args.expected), reproduced, args.tolerance)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    print(f"{'model':22s} {'2019 AVG':>10s} {'rerun AVG':>10s} {'correct (2019 / rerun)':>24s}")
    for name, model in result["models"].items():
        if model.get("missing"):
            print(f"{name:22s} {model['expected_avg']:10.4f} {'missing':>10s}")
            continue
        print(
            f"{name:22s} {model['expected_avg']:10.4f} {model['reproduced_avg']:10.4f} "
            f"{model['expected_correct']:>11d} / {model['reproduced_correct']:<10d}"
            f"{'' if model['reproduced'] else '  DIFFERENT'}"
        )
    verdict = "reproduced" if result["all_reproduced"] else "NOT reproduced for every model"
    print(f"2019 AVG column {verdict}")
    if args.strict and not result["all_reproduced"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
