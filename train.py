from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from silent_speech.data import (
    build_label_mapping,
    build_trial_records,
    load_dataframe,
    summarize_trials,
)
from silent_speech.training import (
    SplitIndices,
    remap_labels,
    seed_everything,
    train_svm,
    train_tcn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fixed-command silent-speech SVM and dual-branch TCN models."
    )
    parser.add_argument("--csv", required=True, help="Path to cleaned sensor CSV.")
    parser.add_argument("--output-dir", default="runs/fixed_commands", help="Output directory.")
    parser.add_argument("--model", choices=["svm", "tcn", "both"], default="both")
    parser.add_argument(
        "--split",
        choices=["random", "session"],
        default="random",
        help="Use random only for the first diagnostic session; session split is the real test.",
    )
    parser.add_argument(
        "--test-session",
        type=int,
        default=None,
        help="Required with --split session. Entire session is held out for testing.",
    )
    parser.add_argument(
        "--remove-first-stop",
        action="store_true",
        help="Remove the earliest STOP trial. Do not use this if the CSV is already cleaned.",
    )
    parser.add_argument(
        "--run-ablations",
        action="store_true",
        help="Run rest-only, jaw-only, ref-only, full, and shuffled-label controls.",
    )
    parser.add_argument("--target-hz", type=float, default=50.0)
    parser.add_argument("--max-seconds", type=float, default=3.2)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(name)


def make_split(
    labels: np.ndarray,
    session_ids: np.ndarray,
    *,
    split_mode: str,
    test_session: int | None,
    seed: int,
) -> SplitIndices:
    all_indices = np.arange(len(labels))
    if split_mode == "random":
        train_val, test = train_test_split(
            all_indices,
            test_size=0.15,
            random_state=seed,
            stratify=labels,
        )
        train, val = train_test_split(
            train_val,
            test_size=0.1765,  # approximately 15% of the full dataset
            random_state=seed + 1,
            stratify=labels[train_val],
        )
    else:
        if test_session is None:
            raise ValueError("--test-session is required with --split session.")
        test = all_indices[session_ids == test_session]
        train_val = all_indices[session_ids != test_session]
        if len(test) == 0:
            available = sorted(np.unique(session_ids).tolist())
            raise ValueError(f"Session {test_session} not found. Available sessions: {available}")
        if len(np.unique(session_ids[train_val])) == 0:
            raise ValueError("No training sessions remain after holding out the test session.")
        train, val = train_test_split(
            train_val,
            test_size=0.18,
            random_state=seed + 1,
            stratify=labels[train_val],
        )

    return SplitIndices(
        train=sorted(map(int, train)),
        val=sorted(map(int, val)),
        test=sorted(map(int, test)),
    )


def save_split_manifest(
    output_dir: Path,
    records,
    labels: np.ndarray,
    split: SplitIndices,
    index_to_name: dict[int, str],
) -> None:
    split_name = {}
    for name, indices in (("train", split.train), ("val", split.val), ("test", split.test)):
        for index in indices:
            split_name[index] = name
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "split": split_name[index],
                "session_id": record.key.session_id,
                "trial_id": record.key.trial_id,
                "label_index": int(labels[index]),
                "target_name": index_to_name[int(labels[index])],
                "duration_s": record.duration_s,
                "valid_samples": record.valid_length,
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "split_manifest.csv", index=False)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    print(f"Using device: {device}")

    df = load_dataframe(
        args.csv,
        remove_first_stop=args.remove_first_stop,
    )
    summary = summarize_trials(df)
    summary.to_csv(output_root / "dataset_summary.csv", index=False)
    print("\nUsable dataset:")
    print(summary.to_string(index=False))

    original_to_index, index_to_name = build_label_mapping(df)
    with (output_root / "label_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "original_to_index": original_to_index,
                "index_to_name": index_to_name,
            },
            handle,
            indent=2,
        )

    # Construct mouth records first; all ablations use the same trial split.
    mouth_records = build_trial_records(
        df,
        region="mouth",
        target_hz=args.target_hz,
        max_seconds=args.max_seconds,
        smoothing_window=args.smoothing_window,
    )
    labels = remap_labels(mouth_records, original_to_index)
    sessions = np.asarray([record.key.session_id for record in mouth_records], dtype=np.int64)
    split = make_split(
        labels,
        sessions,
        split_mode=args.split,
        test_session=args.test_session,
        seed=args.seed,
    )
    save_split_manifest(output_root, mouth_records, labels, split, index_to_name)

    config = vars(args).copy()
    config["device_resolved"] = str(device)
    config["num_trials"] = len(mouth_records)
    config["num_classes"] = len(index_to_name)
    with (output_root / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    experiments = [
        {
            "name": "mouth_both",
            "region": "mouth",
            "sensor_mode": "both",
            "randomize_labels": False,
        }
    ]
    if args.run_ablations:
        experiments = [
            {"name": "rest_both", "region": "rest", "sensor_mode": "both", "randomize_labels": False},
            {"name": "mouth_jaw", "region": "mouth", "sensor_mode": "jaw", "randomize_labels": False},
            {"name": "mouth_ref", "region": "mouth", "sensor_mode": "ref", "randomize_labels": False},
            {"name": "mouth_both", "region": "mouth", "sensor_mode": "both", "randomize_labels": False},
            {"name": "mouth_both_shuffled", "region": "mouth", "sensor_mode": "both", "randomize_labels": True},
        ]

    results = []
    rest_records = None
    for experiment_number, experiment in enumerate(experiments):
        print(f"\n=== Experiment: {experiment['name']} ===")
        if experiment["region"] == "mouth":
            records = mouth_records
        else:
            if rest_records is None:
                rest_records = build_trial_records(
                    df,
                    region="rest",
                    target_hz=args.target_hz,
                    max_seconds=args.max_seconds,
                    smoothing_window=args.smoothing_window,
                )
                mouth_keys = [(r.key.session_id, r.key.trial_id) for r in mouth_records]
                rest_by_key = {(r.key.session_id, r.key.trial_id): r for r in rest_records}
                if set(mouth_keys) != set(rest_by_key):
                    raise RuntimeError("Mouth/rest trial keys do not match; cannot reuse the same split.")
                rest_records = [rest_by_key[key] for key in mouth_keys]
            records = rest_records

        experiment_dir = output_root / experiment["name"]
        experiment_dir.mkdir(parents=True, exist_ok=True)
        seed = args.seed + experiment_number * 100

        if args.model in {"svm", "both"}:
            svm_dir = experiment_dir / "svm"
            svm_dir.mkdir(parents=True, exist_ok=True)
            metrics = train_svm(
                records,
                labels,
                split,
                sensor_mode=experiment["sensor_mode"],
                output_dir=svm_dir,
                index_to_name=index_to_name,
                randomize_labels=bool(experiment["randomize_labels"]),
                seed=seed,
            )
            print(f"SVM test metrics: {metrics.as_dict()}")
            results.append(
                {
                    "experiment": experiment["name"],
                    "model": "svm",
                    **metrics.as_dict(),
                }
            )

        if args.model in {"tcn", "both"}:
            tcn_dir = experiment_dir / "tcn"
            tcn_dir.mkdir(parents=True, exist_ok=True)
            metrics = train_tcn(
                records,
                labels,
                split,
                sensor_mode=experiment["sensor_mode"],
                output_dir=tcn_dir,
                index_to_name=index_to_name,
                batch_size=args.batch_size,
                max_epochs=args.epochs,
                patience=args.patience,
                learning_rate=args.learning_rate,
                randomize_labels=bool(experiment["randomize_labels"]),
                seed=seed,
                device=device,
            )
            print(f"TCN test metrics: {metrics.as_dict()}")
            results.append(
                {
                    "experiment": experiment["name"],
                    "model": "tcn",
                    **metrics.as_dict(),
                }
            )

    results_df = pd.DataFrame(results).sort_values(["model", "experiment"])
    results_df.to_csv(output_root / "experiment_results.csv", index=False)
    print("\n=== Final results ===")
    print(results_df.to_string(index=False))
    print(f"\nArtifacts saved to: {output_root}")


if __name__ == "__main__":
    main()
