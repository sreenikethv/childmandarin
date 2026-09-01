#!/usr/bin/env python3

"""
Generate ChildTalk experiment configuration files.

Schemas
-------

1. Evaluation only:
       TRAIN_MODEL=false

   Generates 4 configurations, one for each test mode.

2. Standard fine-tuning:
       TRAIN_MODEL=true
       CURRICULUM=false

   Generates 4 x 4 = 16 configurations:
       training mode x testing mode

3. Curriculum fine-tuning:
       TRAIN_MODEL=true
       CURRICULUM=true

   Generates 4 x 4 = 16 configurations:
       training mode x testing mode

Modes
-----

adult
    ROLE=adult
    No age restriction.

child
    ROLE=child
    No age restriction.

child_4_6
    ROLE=child
    AGE_MIN=4
    AGE_MAX=6

both
    ROLE=both
    No age restriction.

The same four modes are used for training and testing.

For curriculum training, the downstream manifest generator is
responsible for discovering the actual ages in speaker_info.csv.
Adults are treated as one bucket; children are subdivided by their
actual ages and ordered from oldest to youngest.

No MANIFEST_NAME variable is generated.
"""

from pathlib import Path
import os

# ============================================================
# Global settings
# ============================================================

DATASET = "/projects/assigned/ChildLang/data/ChildTalk/childtalk"

PRETRAINED_MODEL = (
    # "/projects/assigned/ChildLang/models/libri960_big.pt"
    "/projects/assigned/ChildLang/finetune_fairseq/outputs/2026-07-28/11-34-07/checkpoints/checkpoint_best.pt"

)


DIALECTS = "Mandarin"
TEST_DIALECTS = "Mandarin"

OUTPUT_DIR = Path(
    "/projects/assigned/ChildLang/finetune_fairseq/configs"
)


# ============================================================
# Experiment modes
# ============================================================

MODES = {
    "adult": {
        "role": "adult",
        "age_min": "",
        "age_max": "",
    },
    "child": {
        "role": "child",
        "age_min": "",
        "age_max": "",
    },
    "child_4_6": {
        "role": "child",
        "age_min": "4",
        "age_max": "6",
    },
    "both": {
        "role": "both",
        "age_min": "",
        "age_max": "",
    },
}


# ============================================================
# Helpers
# ============================================================

def mode_short_name(mode_name):
    """Return a compact name suitable for experiment names."""

    return {
        "adult": "adult",
        "child": "child",
        "child_4_6": "child_4_6",
        "both": "both",
    }[mode_name]

def dialect_name(dialects):
    """
    Convert a dialect string into a compact filename component.

    Examples:
        Mandarin                  -> mandarin
        Mandarin,Southwestern     -> mandarin_southwestern
    """
    return "_".join(
        d.strip().lower()
        for d in dialects.split(",")
        if d.strip()
    )

def experiment_name(
    schema,
    train_mode=None,
    test_mode=None,
    train_dialects=DIALECTS,
    test_dialects=TEST_DIALECTS,
):
    """
    Generate a short but distinctive experiment name.

    Examples:
        eval_adult_test-mandarin
        eval_child_4_6_test-mandarin
        ft_adult_child_train-mandarin_test-mandarin
        cur_child_4_6_both_train-mandarin_test-mandarin
    """

    train_lang = dialect_name(train_dialects)
    test_lang = dialect_name(test_dialects)

    if schema == "eval":
        return (
            f"eval_{mode_short_name(test_mode)}"
            f"_test-{test_lang}"
        )

    if schema == "ft":
        return (
            f"ft_{mode_short_name(train_mode)}"
            f"_{mode_short_name(test_mode)}"
            f"_train-{train_lang}"
            f"_test-{test_lang}"
        )

    if schema == "cur":
        return (
            f"cur_{mode_short_name(train_mode)}"
            f"_{mode_short_name(test_mode)}"
            f"_train-{train_lang}"
            f"_test-{test_lang}"
        )

    raise ValueError(f"Unknown schema: {schema}")

def mode_variables(mode):
    """Return training/test config variables for a mode."""

    values = MODES[mode]

    return {
        "ROLE": values["role"],
        "AGE_MIN": values["age_min"],
        "AGE_MAX": values["age_max"],
    }


def write_config(
    path,
    *,
    name,
    train_model,
    curriculum,
    train_mode=None,
    test_mode=None,
):
    """Write one experiment configuration."""

    config = []

    config.append(f'EXPERIMENT_NAME="{name}"')
    config.append("")

    config.append(f'DATASET="{DATASET}"')
    config.append("")

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    if train_model:
        train = mode_variables(train_mode)

        config.append(f'DIALECTS="{DIALECTS}"')
        config.append(f'ROLE="{train["ROLE"]}"')
        config.append(f'AGE_MIN="{train["AGE_MIN"]}"')
        config.append(f'AGE_MAX="{train["AGE_MAX"]}"')
    else:
        # These values are not used for training, but keeping
        # them defined makes every generated config structurally
        # compatible with make_manifest.sh.
        config.append(f'DIALECTS="{DIALECTS}"')
        config.append('ROLE="both"')
        config.append('AGE_MIN=""')
        config.append('AGE_MAX=""')

    config.append("")

    # --------------------------------------------------------
    # Testing configuration
    # --------------------------------------------------------

    test = mode_variables(test_mode)

    config.append(f'TEST_DIALECTS="{TEST_DIALECTS}"')
    config.append(f'TEST_ROLE="{test["ROLE"]}"')
    config.append(f'TEST_AGE_MIN="{test["AGE_MIN"]}"')
    config.append(f'TEST_AGE_MAX="{test["AGE_MAX"]}"')
    config.append("")

    # --------------------------------------------------------
    # Training behavior
    # --------------------------------------------------------

    config.append(
        f'TRAIN_MODEL="{"true" if train_model else "false"}"'
    )

    config.append(
        f'CURRICULUM="{"true" if curriculum else "false"}"'
    )

    config.append("")

    config.append(f'PRETRAINED_MODEL="{PRETRAINED_MODEL}"')
    config.append("")

    path.write_text(
        "\n".join(config),
        encoding="utf-8",
    )


# ============================================================
# Schema generation
# ============================================================

def generate_evaluation_configs():
    """
    Schema 1:

        TRAIN_MODEL=false

    Four configs, one for each test mode.
    """

    count = 0

    for test_mode in MODES:

        name = experiment_name(
            schema="eval",
            test_mode=test_mode,
            test_dialects=TEST_DIALECTS
        )

        path = OUTPUT_DIR / f"{name}.cfg"

        write_config(
            path,
            name=name,
            train_model=False,
            curriculum=False,
            test_mode=test_mode,
        )

        count += 1

    return count


def generate_finetuning_configs():
    """
    Schema 2:

        TRAIN_MODEL=true
        CURRICULUM=false

    Four training modes x four testing modes.
    """

    count = 0

    for train_mode in MODES:
        for test_mode in MODES:

            name = experiment_name(
                schema="ft",
                train_mode=train_mode,
                test_mode=test_mode,
                train_dialects=DIALECTS,
                test_dialects=TEST_DIALECTS,
            )

            path = OUTPUT_DIR / f"{name}.cfg"

            write_config(
                path,
                name=name,
                train_model=True,
                curriculum=False,
                train_mode=train_mode,
                test_mode=test_mode,
            )

            count += 1

    return count


def generate_curriculum_configs():
    """
    Schema 3:

        TRAIN_MODEL=true
        CURRICULUM=true

    Four training modes x four testing modes.
    """

    count = 0

    for train_mode in MODES:
        for test_mode in MODES:

            name = experiment_name(
                schema="cur",
                train_mode=train_mode,
                test_mode=test_mode,
                train_dialects=DIALECTS,
                test_dialects=TEST_DIALECTS,
            )

            path = OUTPUT_DIR / f"{name}.cfg"

            write_config(
                path,
                name=name,
                train_model=True,
                curriculum=True,
                train_mode=train_mode,
                test_mode=test_mode,
            )

            count += 1

    return count

# ============================================================
# Generate .txt file of configs
# ============================================================


PROJECT_ROOT = "/projects/assigned/ChildLang/finetune_fairseq"
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
CONFIG_LIST = os.path.join(CONFIG_DIR, "config_list.txt")

def write_config_list():
    configs = sorted(
        f for f in os.listdir(CONFIG_DIR)
        if f.endswith(".cfg")
    )

    with open(CONFIG_LIST, "w", encoding="utf-8") as f:
        for filename in configs:
            name = os.path.splitext(filename)[0]
            f.write(f"{name} {filename}\n")

    print(f"Wrote config list: {CONFIG_LIST}")
    print(f"Configs listed: {len(configs)}")

# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    counts = {
        # "evaluation": generate_evaluation_configs()
        "fine_tuning": generate_finetuning_configs(),
        # "curriculum": generate_curriculum_configs(),
    }

    total = sum(counts.values())

    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Generated:")
    # print(f"  Evaluation:  {counts['evaluation']:2d}")
    print(f"  Fine-tuning: {counts['fine_tuning']:2d}")
    # print(f"  Curriculum:  {counts['curriculum']:2d}")
    print(f"  Total:       {total:2d}")
    print()

    for path in sorted(OUTPUT_DIR.glob("*.cfg")):
        print(f"  {path.name}")

    write_config_list()

if __name__ == "__main__":
    main()
