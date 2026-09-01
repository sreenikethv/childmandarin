#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import soundfile as sf


LABELS = ("T1", "T2", "T3", "T4", "T5")


# ---------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------

def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {path}")

        reader.fieldnames = [
            field.strip() for field in reader.fieldnames
        ]

        for row in reader:
            yield {
                key.strip(): (
                    value.strip() if value is not None else ""
                )
                for key, value in row.items()
            }


def load_speakers(path):
    speakers = {}

    for row in load_csv(path):
        speaker = row.get("speaker id", "")
        age = row.get("age", "")

        if not speaker or not age or age.lower() == "none":
            continue

        try:
            speakers[speaker] = float(age)
        except ValueError:
            print(f"WARNING: invalid age for {speaker}: {age}")

    return speakers


def load_dialogues(path, dialects):
    dialogues = {}

    for row in load_csv(path):
        dialogue = row.get("dialogue_id", "")
        dialect = row.get("dialect", "")

        if not dialogue:
            continue

        if dialects and dialect not in dialects:
            continue

        child = row.get("child_id", "")
        adult = row.get("adult_id", "")

        if not child or not adult:
            continue

        dialogues[dialogue] = {
            "dialect": dialect,
            "child": child,
            "adult": adult,
        }

    return dialogues


def load_wavscp(path):
    wavs = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(maxsplit=1)
            if len(parts) == 2:
                wavs[parts[0]] = parts[1]

    return wavs


def load_tones(path):
    tones = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split()
            if len(parts) >= 2:
                tones[parts[0]] = parts[1:]

    return tones


# ---------------------------------------------------------------------
# Utterance identification
# ---------------------------------------------------------------------

def identify_dialogue(utt, wav, dialogues):
    matches = [
        dialogue
        for dialogue in dialogues
        if dialogue in utt or dialogue in wav
    ]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise ValueError(
            f"Could not identify dialogue for {utt}"
        )

    raise ValueError(
        f"Multiple dialogues found for {utt}: {matches}"
    )


def identify_speaker(utt, dialogue):
    parts = utt.split("_")

    positions = [
        i for i, part in enumerate(parts)
        if part.lower() in {"adult", "child"}
    ]

    if len(positions) != 1:
        raise ValueError(
            f"Could not identify unique role in {utt}"
        )

    role = parts[positions[0]].lower()

    if role == "adult":
        return dialogue["adult"], "adult"

    return dialogue["child"], "child"


# ---------------------------------------------------------------------
# Build utterance metadata
# ---------------------------------------------------------------------

def build_items(
    split,
    childtalk,
    tone_dir,
    speakers,
    dialogues,
):
    info_dir = childtalk / "information" / "short" / split

    wavs = load_wavscp(info_dir / "wav.scp")
    tones = load_tones(tone_dir / f"{split}.tone")

    items = []
    skipped = 0

    for utt, wav in wavs.items():

        if utt not in tones:
            skipped += 1
            continue

        try:
            dialogue_id = identify_dialogue(
                utt,
                wav,
                dialogues,
            )

            dialogue = dialogues[dialogue_id]

            # Ignore utterances from dialects that are not selected.
            if dialogues and dialogue["dialect"] not in {
                item["dialect"] for item in dialogues.values()
            }:
                continue

            speaker, role = identify_speaker(
                utt,
                dialogue,
            )

            if speaker not in speakers:
                raise ValueError(
                    f"Speaker {speaker} missing from speaker_info.csv"
                )

            
            items.append({
                "utt": utt,
                "wav": wav,
                "speaker": speaker,
                "dialogue": dialogue_id,
                "dialect": dialogue["dialect"],
                "role": role,
                "age": speakers[speaker],
                "tones": tones[utt],
            })

        except ValueError as exc:
            print(f"WARNING: skipping {utt}: {exc}")
            skipped += 1

    print(
        f"{split}: loaded={len(items)}, skipped={skipped}"
    )

    return items


# ---------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------

def parse_dialects(value):
    if not value:
        return set()

    return {
        x.strip()
        for x in value.split(",")
        if x.strip()
    }


def matches(
    item,
    dialects,
    role,
    age_min,
    age_max,
):
    if dialects and item["dialect"] not in dialects:
        return False

    if role != "both" and item["role"] != role:
        return False

    if age_min is not None and item["age"] < age_min:
        return False

    if age_max is not None and item["age"] > age_max:
        return False

    return True


def filter_items(
    items,
    dialects,
    role,
    age_min,
    age_max,
):
    return [
        item
        for item in items
        if matches(
            item,
            dialects,
            role,
            age_min,
            age_max,
        )
    ]


# ---------------------------------------------------------------------
# Leakage checking
# ---------------------------------------------------------------------

def check_split_leakage(items_by_split):
    dialogue_splits = {}

    for split, items in items_by_split.items():
        for item in items:
            dialogue_splits.setdefault(
                item["dialogue"],
                set(),
            ).add(split)

    leaked = {
        dialogue: splits
        for dialogue, splits in dialogue_splits.items()
        if len(splits) > 1
    }

    if not leaked:
        return

    print("WARNING: dialogue leakage detected:")

    for dialogue, splits in sorted(leaked.items()):
        print(
            f"  {dialogue}: "
            f"{', '.join(sorted(splits))}"
        )

    raise RuntimeError("Dialogue leakage detected.")


# ---------------------------------------------------------------------
# Manifest writing
# ---------------------------------------------------------------------

def write_split(
    items,
    split,
    output,
    childtalk,
):
    output.mkdir(parents=True, exist_ok=True)

    tsv_path = output / f"{split}.tsv"
    ltr_path = output / f"{split}.ltr"

    written = 0

    with (
        open(tsv_path, "w", encoding="utf-8") as tsv,
        open(ltr_path, "w", encoding="utf-8") as ltr,
    ):
        # Fairseq expects dataset root on first line.
        tsv.write(str(childtalk.resolve()) + "\n")

        for item in items:
            wav_path = childtalk / item["wav"]

            if not wav_path.is_file():
                print(
                    f"WARNING: missing audio: {wav_path}"
                )
                continue

            try:
                frames = sf.info(str(wav_path)).frames
            except Exception as exc:
                print(
                    f"WARNING: cannot read "
                    f"{wav_path}: {exc}"
                )
                continue

            relative = wav_path.relative_to(childtalk)

            tsv.write(
                f"{relative}\t{frames}\n"
            )

            ltr.write(
                " ".join(item["tones"]) + "\n"
            )

            written += 1

    print(f"{split}: wrote={written}")


def write_dictionary(output):
    with open(
        output / "dict.ltr.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for label in LABELS:
            f.write(f"{label} 1\n")


# ---------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------

def build_curriculum_groups(items):
    """
    Automatically determine curriculum order.

    Adults are always first as one group.

    Children follow from oldest to youngest.

    The groups are derived from the already-filtered training
    items, so dialect, role, and age restrictions have already
    been applied.
    """

    groups = []

    adults = [
        item
        for item in items
        if item["role"] == "adult"
    ]

    children = [
        item
        for item in items
        if item["role"] == "child"
    ]

    if adults:
        groups.append(("adult", adults))

    ages = sorted(
        {item["age"] for item in children},
        reverse=True,
    )

    for age in ages:
        age_items = [
            item
            for item in children
            if item["age"] == age
        ]

        groups.append(
            (f"age_{age:g}", age_items)
        )

    return groups

def build_curriculum_stages(items):
    """
    Build cumulative curriculum stages.

    Example:

        stage_01 = adults
        stage_02 = adults + age 8
        stage_03 = adults + age 8 + age 7
        stage_04 = adults + age 8 + age 7 + age 6

    All selected dialects are included within each group.
    """

    groups = build_curriculum_groups(items)

    if not groups:
        raise RuntimeError(
            "Could not construct any curriculum groups."
        )

    stages = []
    cumulative = []

    for index, (label, new_items) in enumerate(
        groups,
        start=1,
    ):
        cumulative.extend(new_items)

        stages.append({
            "name": f"stage_{index:02d}",
            "label": label,
            "items": list(cumulative),
            "new_items": len(new_items),
        })

    return stages

def write_curriculum(
    stages,
    dev,
    output,
    childtalk,
):
    for stage in stages:
        stage_dir = output / stage["name"]

        print()
        print(
            f"Writing {stage['name']} "
            f"({stage['label']}): "
            f"{len(stage['items'])} training utterances "
            f"(+{stage['new_items']})"
        )

        write_split(
            stage["items"],
            "train",
            stage_dir,
            childtalk,
        )

        write_split(
            dev,
            "dev",
            stage_dir,
            childtalk,
        )

        write_dictionary(stage_dir)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Fairseq manifests, optionally "
            "with sequential + replay curriculum stages."
        )
    )

    parser.add_argument(
        "--childtalk",
        required=True,
    )

    parser.add_argument(
        "--tone-dir",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--dialects",
        default="",
    )

    parser.add_argument(
        "--test-dialects",
        default="",
    )

    parser.add_argument(
        "--role",
        choices=("child", "adult", "both"),
        default="both",
    )

    parser.add_argument(
        "--test-role",
        choices=("child", "adult", "both"),
        default=None,
    )

    parser.add_argument(
        "--age-min",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--age-max",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--test-age-min",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--test-age-max",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--curriculum",
        action="store_true",
    )

    args = parser.parse_args()

    childtalk = Path(args.childtalk).resolve()
    tone_dir = Path(args.tone_dir).resolve()
    output = Path(args.output).resolve()

    information = childtalk / "information"

    speakers = load_speakers(
        information / "speaker_info.csv"
    )

    train_dialects = parse_dialects(
        args.dialects
    )

    test_dialects = parse_dialects(
        args.test_dialects
    )

    if not test_dialects:
        test_dialects = train_dialects

    print(f"Loaded {len(speakers)} speakers")

    print(
        "Train dialects: "
        f"{', '.join(sorted(train_dialects)) or 'ALL'}"
    )

    print(
        "Test dialects: "
        f"{', '.join(sorted(test_dialects)) or 'ALL'}"
    )

    train_dialogues = load_dialogues(
        information / "dialogue_info.csv",
        train_dialects,
    )

    test_dialogues = load_dialogues(
        information / "dialogue_info.csv",
        test_dialects,
    )

    print(
        f"Train dialogues: {len(train_dialogues)}"
    )

    print(
        f"Test dialogues: {len(test_dialogues)}"
    )

    # -------------------------------------------------------------
    # Build raw splits
    # -------------------------------------------------------------

    train = build_items(
        "train",
        childtalk,
        tone_dir,
        speakers,
        train_dialogues,
    )

    dev = build_items(
        "dev",
        childtalk,
        tone_dir,
        speakers,
        train_dialogues,
    )

    test = build_items(
        "test",
        childtalk,
        tone_dir,
        speakers,
        test_dialogues,
    )

    check_split_leakage({
        "train": train,
        "dev": dev,
        "test": test,
    })

    # -------------------------------------------------------------
    # Apply normal filtering
    # -------------------------------------------------------------

    train = filter_items(
        train,
        train_dialects,
        args.role,
        args.age_min,
        args.age_max,
    )

    dev = filter_items(
        dev,
        train_dialects,
        args.role,
        args.age_min,
        args.age_max,
    )

    test_role = (
        args.test_role
        if args.test_role is not None
        else args.role
    )

    test = filter_items(
        test,
        test_dialects,
        test_role,
        args.test_age_min,
        args.test_age_max,
    )

    print()
    print(f"Filtered train: {len(train)}")
    print(f"Filtered dev:   {len(dev)}")
    print(f"Filtered test:  {len(test)}")

    if not train:
        raise RuntimeError(
            "No training utterances after filtering."
        )

    if not dev:
        raise RuntimeError(
            "No development utterances after filtering."
        )

    if not test:
        raise RuntimeError(
            "No test utterances after filtering."
        )

    # -------------------------------------------------------------
    # Curriculum
    # -------------------------------------------------------------

    if args.curriculum:
        print()
        print("Automatic curriculum:")

        stages = build_curriculum_stages(train)

        for stage in stages:
            print(
                f"  {stage['name']}: "
                f"{stage['label']} "
                f"(+{stage['new_items']})"
            )

        write_curriculum(
            stages,
            dev,
            output,
            childtalk,
        )

        write_split(
            test,
            "test",
            output,
            childtalk,
        )

        write_dictionary(output)

        print()
        print(
            f"Curriculum manifest written to: {output}"
        )

        return

    # -------------------------------------------------------------
    # Standard non-curriculum manifest
    # -------------------------------------------------------------

    write_split(
        train,
        "train",
        output,
        childtalk,
    )

    write_split(
        dev,
        "dev",
        output,
        childtalk,
    )

    write_split(
        test,
        "test",
        output,
        childtalk,
    )

    write_dictionary(output)

    print()
    print(
        f"Manifest written to: {output}"
    )


if __name__ == "__main__":
    main()