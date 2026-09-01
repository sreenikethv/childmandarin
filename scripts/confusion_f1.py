#!/usr/bin/env python3

import sys
from pathlib import Path
from collections import Counter

import torch
from sklearn.metrics import confusion_matrix, f1_score

PROJECT = Path("/projects/assigned/ChildLang")
FT = PROJECT / "finetune_fairseq"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} EXPERIMENT_NAME")

    name = sys.argv[1]

    manifest = FT / "manifest" / name
    output = FT / "outputs" / name
    result_dir = FT / "results" / name
    result_dir.mkdir(parents=True, exist_ok=True)

    # Match test_manifest.sh checkpoint selection.
    stages = sorted(output.glob("stage_*"))

    if stages:
        model_path = stages[-1] / "checkpoint_best.pt"
    else:
        model_path = output / "checkpoint_best.pt"

    if not model_path.is_file():
        raise SystemExit(f"Model does not exist:\n  {model_path}")

    source = FT.parent / "fairseq"

    sys.path.insert(0, str(source))

    from fairseq import checkpoint_utils, utils

    reference_path = manifest / "test.ltr"

    if not reference_path.is_file():
        raise SystemExit(f"Reference does not exist:\n  {reference_path}")

    references = [
        line.strip().split()
        for line in reference_path.read_text().splitlines()
        if line.strip()
    ]

    labels = ["T1", "T2", "T3", "T4", "T5"]

    print(f"Model: {model_path}")
    print(f"Manifest: {manifest}")

    models, saved_cfg, task = (
        checkpoint_utils.load_model_ensemble_and_task(
            [str(model_path)],
            arg_overrides={"task": {"data": str(manifest)}},
        )
    )

    model = models[0]
    model.eval()

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        model.cuda()
        model.half()

    task.load_dataset(
        "test",
        task_cfg=saved_cfg.task,
    )

    dataset = task.dataset("test")

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=1000000,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=True,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=4,
    ).next_epoch_itr(shuffle=False)

    dictionary = task.target_dictionary
    blank = dictionary.index(task.blank_symbol)

    y_true = []
    y_pred = []

    sample_index = 0

    with torch.no_grad():
        for sample in itr:
            if not sample:
                continue

            if use_cuda:
                sample = utils.move_to_cuda(sample)

            net_output = model(**sample["net_input"])

            logits = model.get_logits(
                net_output,
                normalize=False,
            )

            # T x B x C -> B x T x C
            predictions = logits.transpose(0, 1).argmax(dim=-1)

            targets = sample["target"]

            for b in range(predictions.size(0)):
                # CTC greedy decoding.
                tokens = predictions[b].tolist()

                decoded = []
                previous = blank

                for token in tokens:
                    if token == blank:
                        previous = blank
                        continue

                    if token != previous:
                        decoded.append(dictionary[token])

                    previous = token

                if sample_index >= len(references):
                    raise RuntimeError(
                        "More predictions than reference sequences."
                    )

                reference = references[sample_index]

                # F1/confusion matrix requires paired labels.
                # Only compare positions that exist in both sequences.
                n = min(len(reference), len(decoded))

                y_true.extend(reference[:n])
                y_pred.extend(decoded[:n])

                sample_index += 1

    if sample_index != len(references):
        raise RuntimeError(
            f"Processed {sample_index} samples but found "
            f"{len(references)} reference sequences."
        )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    result = result_dir / "confusion_f1.txt"

    with result.open("w") as f:
        f.write("Tone confusion matrix\n")
        f.write("Rows = reference; columns = prediction\n\n")

        f.write("       " + "".join(f"{x:>8}" for x in labels) + "\n")

        for label, row in zip(labels, matrix):
            f.write(
                f"{label:>5} "
                + "".join(f"{x:>8}" for x in row)
                + "\n"
            )

        f.write("\nTone F1\n\n")

        for label, score in zip(labels, f1):
            f.write(f"{label}: {score:.4f}\n")

        f.write(f"\nMacro F1: {macro_f1:.4f}\n")

    print()
    print(result)
    print()
    print("Tone confusion matrix")
    print("Rows = reference; columns = prediction")
    print()
    print("       " + "".join(f"{x:>8}" for x in labels))

    for label, row in zip(labels, matrix):
        print(
            f"{label:>5} "
            + "".join(f"{x:>8}" for x in row)
        )

    print()
    print("Tone F1")
    for label, score in zip(labels, f1):
        print(f"{label}: {score:.4f}")

    print(f"\nMacro F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()