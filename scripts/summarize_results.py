#!/usr/bin/env python3

from pathlib import Path
import re


PROJECT_ROOT = Path("/projects/assigned/ChildLang")
RESULTS_DIR = PROJECT_ROOT / "finetune_fairseq" / "results"
OUTPUT_FILE = RESULTS_DIR / "best_results.txt"


def extract_uer(path):
    """Extract the lowest UER reported in a result file."""

    pattern = re.compile
        r"UER[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    )

    values = []

    with path.open(
        encoding="utf-8",
        errors="replace",
    ) as f:
        for line in f:
            for match in pattern.finditer(line):
                values.append(float(match.group(1)))

    return min(values) if values else None


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # test_manifest.sh currently produces:
    #
    # results/<experiment>.txt
    #
    for path in RESULTS_DIR.glob("*.txt"):

        # Do not treat our own summary as an experiment.
        if path.name == OUTPUT_FILE.name:
            continue

        uer = extract_uer(path)

        if uer is None:
            print(f"WARNING: no UER found in {path}")
            continue

        experiment = path.stem

        results.append(
            (uer, experiment, path)
        )

    if not results:
        raise RuntimeError(
            f"No UER results found in {RESULTS_DIR}\n"
            f"Expected files such as:\n"
            f"  {RESULTS_DIR}/experiment_name.txt"
        )

    # Lower UER is better.
    results.sort(key=lambda x: x[0])

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write("Experiment Results Summary\n")
        f.write("==========================\n\n")

        f.write(
            f"{'Rank':<6}"
            f"{'UER':<12}"
            f"Experiment\n"
        )

        f.write(
            f"{'-' * 6}"
            f"{'-' * 12}"
            f"{'-' * 50}\n"
        )

        for rank, (uer, experiment, path) in enumerate(
            results,
            start=1,
        ):
            f.write(
                f"{rank:<6}"
                f"{uer:<12.4f}"
                f"{experiment}\n"
            )

        best_uer, best_experiment, best_path = results[0]

        f.write("\n")
        f.write("Best Result\n")
        f.write("===========\n")
        f.write(
            f"Experiment: {best_experiment}\n"
        )
        f.write(
            f"UER:        {best_uer:.4f}\n"
        )
        f.write(
            f"File:       {best_path}\n"
        )

    print(
        f"Found {len(results)} experiment results."
    )
    print(
        f"Best experiment: {best_experiment}"
    )
    print(
        f"Best UER: {best_uer:.4f}"
    )
    print(
        f"Summary: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
