import csv
from collections import Counter

CSV = "/projects/assigned/ChildLang/data/ChildTalk/childtalk/information/speaker_info.csv"
DIALECT = "mandarin"

counts = Counter()

with open(CSV, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    # Strip whitespace from column names.
    reader.fieldnames = [x.strip() for x in reader.fieldnames]

    for row in reader:
        row = {k.strip(): (v.strip() if v else "") for k, v in row.items()}

        if row["location"].lower() != DIALECT.lower():
            continue

        try:
            age = float(row["age"])
        except (TypeError, ValueError):
            continue

        # All adults belong to one bucket.
        if age >= 18:
            counts["adult"] += 1
        else:
            counts[age] += 1

print(f"\nCurriculum buckets: {DIALECT}\n")
print(f"{'Bucket':>12} {'Recordings':>12}")
print("-" * 25)

if "adult" in counts:
    print(f"{'adult':>12} {counts['adult']:>12,}")

for age in sorted(
    (x for x in counts if x != "adult"),
    reverse=True,
):
    label = str(int(age)) if age.is_integer() else f"{age:g}"
    print(f"{label:>12} {counts[age]:>12,}")

print("-" * 25)
print(f"{'TOTAL':>12} {sum(counts.values()):>12,}")