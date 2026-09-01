#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_tone_labels.py

Generate experiment-specific tone labels from ChildTalk transcripts.

The original ChildTalk information/ directory is never modified.

Input:
    ChildTalk/information/short/{train,dev,test}/text

Output:
    <output>/{train,dev,test}.tone

Each output line is:

    <utterance-id> T1 T2 T3 ...

Example:

    U2001_foo 妈妈我要吃饭

becomes:

    U2001_foo T1 T1 T3 T4 T1 T4

Usage:
    python make_tone_labels.py \
        --childtalk /projects/assigned/ChildLang/data/ChildTalk/childtalk \
        --output /projects/assigned/ChildLang/finetune_fairseq/manifest/EXP/tone_labels
"""

import argparse
from pathlib import Path

from pypinyin import Style, lazy_pinyin


def text_to_tones(text):
    """
    Convert Chinese text to T1-T5 labels.

    Pinyin tones 1-4 are preserved.
    Toneless/neutral syllables are mapped to T5.

    Non-pronounceable material is ignored by pypinyin.
    """

    tones = []

    syllables = lazy_pinyin(
        text,
        style=Style.TONE3,
        errors="ignore",
    )

    for syllable in syllables:
        if not syllable:
            continue

        last = syllable[-1]

        if last in "1234":
            tones.append("T" + last)
        elif last == "5":
            tones.append("T5")
        else:
            # Pinyin without an explicit tone is treated as neutral.
            tones.append("T5")

    return tones


def process_split(text_file, output_file):
    written = 0
    skipped = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with (
        text_file.open("r", encoding="utf-8") as fin,
        output_file.open("w", encoding="utf-8") as fout,
    ):
        for line_number, line in enumerate(fin, start=1):
            line = line.rstrip("\n")

            if not line.strip():
                continue

            parts = line.split(maxsplit=1)

            if len(parts) != 2:
                skipped += 1
                continue

            utt, text = parts
            tones = text_to_tones(text)

            if not tones:
                skipped += 1
                continue

            fout.write(
                f"{utt} {' '.join(tones)}\n"
            )

            written += 1

    print(
        f"{text_file.name}: "
        f"wrote={written}, skipped={skipped}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--childtalk",
        required=True,
        help="ChildTalk root directory",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Experiment-specific tone-label directory",
    )

    args = parser.parse_args()

    childtalk = Path(args.childtalk)
    output = Path(args.output)

    text_root = (
        childtalk
        / "information"
        / "short"
    )

    for split in ("train", "dev", "test"):
        text_file = text_root / split / "text"
        output_file = output / f"{split}.tone"

        if not text_file.exists():
            raise FileNotFoundError(
                f"Missing transcript file: {text_file}"
            )

        process_split(
            text_file=text_file,
            output_file=output_file,
        )


if __name__ == "__main__":
    main()