from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    import serial
except ImportError as exc:
    raise SystemExit(
        "pyserial is required. Install it with: python -m pip install pyserial"
    ) from exc

from compare_svm_variants import (
    make_pipeline,
    rotation_resistant_sequence,
    sequence_statistics,
    trial_frames,
)
from live_svm import (
    auto_detect_port,
    capture_trial,
    drain_until_quiet,
    save_capture,
)
from silent_speech.data import (
    build_label_mapping,
    build_trial_records,
    load_dataframe,
)
from silent_speech.features import engineered_features


SHORT_TARGET_TYPES = {"word", "gesture", "unknown"}
VARIANT_ORDER = [
    "jaw_axis",
    "ref_axis_diagnostic",
    "jaw_ref_axis",
    "rotation_resistant",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture paired-IMU trials live and compare four SVM feature variants "
            "side by side: jaw-only, reference-only, both raw axes, and "
            "rotation-resistant features."
        )
    )
    parser.add_argument("--csv", required=True, help="Cleaned training CSV.")
    parser.add_argument(
        "--port",
        default=None,
        help="Arduino serial port. Omit to auto-detect a single USB serial device.",
    )
    parser.add_argument("--baud-rate", type=int, default=921600)
    parser.add_argument(
        "--mode",
        choices=["short", "all"],
        default="short",
        help="short excludes sentence labels; all includes every active label.",
    )
    parser.add_argument(
        "--trigger-command",
        default="u",
        help="Character sent to the Arduino to start a live trial.",
    )
    parser.add_argument("--target-hz", type=float, default=50.0)
    parser.add_argument("--max-seconds", type=float, default=3.2)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--startup-wait", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-incomplete",
        action="store_true",
        help=(
            "Keep rows whose ready_mask is not 15. By default these stale/partial "
            "sensor rows are removed before prediction."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="runs/live_variant_comparison",
        help="Directory for fitted models and the prediction log.",
    )
    parser.add_argument(
        "--log-dir",
        default="runs/live_demo/captures",
        help="Directory in which each raw live trial is saved.",
    )
    parser.add_argument(
        "--live-csv",
        default=None,
        help="Classify one existing capture instead of opening the serial port.",
    )
    parser.add_argument(
        "--actual-label",
        default=None,
        help="True label for --live-csv, used only for scoring the prediction.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def canonical_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def prepare_training(args: argparse.Namespace) -> tuple[
    dict[str, Any],
    dict[int, str],
    list[str],
]:
    df = load_dataframe(args.csv)
    df["target_type"] = df["target_type"].astype(str).str.strip().str.lower()
    if args.mode == "short":
        df = df.loc[df["target_type"].isin(SHORT_TARGET_TYPES)].copy()
    if df.empty:
        raise ValueError("No usable training rows remain after mode filtering.")

    original_to_index, index_to_name = build_label_mapping(df)
    records = build_trial_records(
        df,
        region="mouth",
        target_hz=args.target_hz,
        max_seconds=args.max_seconds,
        smoothing_window=args.smoothing_window,
    )
    frames = trial_frames(df)
    labels = np.asarray(
        [original_to_index[record.label_original] for record in records],
        dtype=np.int64,
    )

    features = {
        "jaw_axis": np.stack(
            [
                engineered_features(record.sequence, record.mask, sensor_mode="jaw")
                for record in records
            ]
        ),
        "ref_axis_diagnostic": np.stack(
            [
                engineered_features(record.sequence, record.mask, sensor_mode="ref")
                for record in records
            ]
        ),
        "jaw_ref_axis": np.stack(
            [
                engineered_features(record.sequence, record.mask, sensor_mode="both")
                for record in records
            ]
        ),
        "rotation_resistant": np.stack(
            [
                sequence_statistics(
                    rotation_resistant_sequence(
                        frames[(record.key.session_id, record.key.trial_id)],
                        target_hz=args.target_hz,
                        max_seconds=args.max_seconds,
                        smoothing_window=args.smoothing_window,
                    ),
                    max_length=int(np.ceil(args.target_hz * args.max_seconds)),
                )
                for record in records
            ]
        ),
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    models: dict[str, Any] = {}
    for variant in VARIANT_ORDER:
        pipeline = make_pipeline(args.seed)
        pipeline.fit(features[variant], labels)
        models[variant] = pipeline
        joblib.dump(
            {
                "pipeline": pipeline,
                "variant": variant,
                "index_to_name": index_to_name,
                "target_hz": args.target_hz,
                "max_seconds": args.max_seconds,
                "smoothing_window": args.smoothing_window,
                "training_csv": str(Path(args.csv).expanduser().resolve()),
                "num_trials": len(records),
                "trained_at": datetime.now().isoformat(timespec="seconds"),
            },
            output_dir / f"{variant}.joblib",
        )

    class_names = [index_to_name[index] for index in sorted(index_to_name)]
    print(f"Trained four live SVMs on {len(records)} trials and {len(class_names)} classes.")
    print("Classes: " + ", ".join(class_names))
    return models, index_to_name, class_names


def prepare_capture_for_prediction(
    raw: pd.DataFrame,
    *,
    keep_incomplete: bool,
) -> tuple[pd.DataFrame, int]:
    frame = raw.copy()
    removed = 0
    if "ready_mask" in frame.columns and not keep_incomplete:
        complete = pd.to_numeric(frame["ready_mask"], errors="coerce") == 15
        removed = int((~complete).sum())
        frame = frame.loc[complete].copy()

    frame["label"] = 0
    frame["target_name"] = "live"
    frame["target_type"] = "unknown"

    phase_counts = frame["phase"].value_counts().to_dict()
    if int(phase_counts.get(0, 0)) < 10 or int(phase_counts.get(1, 0)) < 10:
        raise ValueError(
            "Too few complete baseline or mouth samples after filtering: "
            f"{phase_counts}. Use --keep-incomplete only as a diagnostic."
        )
    return frame, removed


def live_features(frame: pd.DataFrame, args: argparse.Namespace) -> dict[str, np.ndarray]:
    records = build_trial_records(
        frame,
        region="mouth",
        target_hz=args.target_hz,
        max_seconds=args.max_seconds,
        smoothing_window=args.smoothing_window,
    )
    if len(records) != 1:
        raise ValueError(f"Expected one live trial, found {len(records)}.")
    record = records[0]

    return {
        "jaw_axis": engineered_features(record.sequence, record.mask, sensor_mode="jaw"),
        "ref_axis_diagnostic": engineered_features(
            record.sequence, record.mask, sensor_mode="ref"
        ),
        "jaw_ref_axis": engineered_features(
            record.sequence, record.mask, sensor_mode="both"
        ),
        "rotation_resistant": sequence_statistics(
            rotation_resistant_sequence(
                frame,
                target_hz=args.target_hz,
                max_seconds=args.max_seconds,
                smoothing_window=args.smoothing_window,
            ),
            max_length=int(np.ceil(args.target_hz * args.max_seconds)),
        ),
    }


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exp = np.exp(np.clip(shifted, -60.0, 0.0))
    total = float(exp.sum())
    if total <= 0:
        return np.full(len(scores), 1.0 / len(scores), dtype=np.float64)
    return exp / total


def rank_variant(
    pipeline: Any,
    feature: np.ndarray,
    index_to_name: dict[int, str],
    top_k: int,
) -> dict[str, Any]:
    scores = np.asarray(pipeline.decision_function(feature[None, :]), dtype=np.float64)
    if scores.ndim == 2:
        scores = scores[0]
    classes = np.asarray(pipeline.named_steps["classifier"].classes_, dtype=np.int64)
    if scores.ndim != 1 or len(scores) != len(classes):
        raise RuntimeError(
            f"Unexpected decision score shape {scores.shape} for {len(classes)} classes."
        )

    relative = stable_softmax(scores)
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    second = int(order[1]) if len(order) > 1 else best
    ranking = [
        {
            "name": index_to_name[int(classes[position])],
            "score": float(scores[position]),
            "relative": float(relative[position]),
        }
        for position in order[: max(1, top_k)]
    ]
    return {
        "prediction": ranking[0]["name"],
        "margin": float(scores[best] - scores[second]),
        "ranking": ranking,
    }


def print_results(
    results: dict[str, dict[str, Any]],
    *,
    actual: str | None,
    removed_rows: int,
) -> None:
    predictions = [results[variant]["prediction"] for variant in VARIANT_ORDER]
    counts = Counter(predictions)
    consensus_name, consensus_votes = counts.most_common(1)[0]
    tie = list(counts.values()).count(consensus_votes) > 1

    print("\n" + "=" * 78)
    print("LIVE SVM COMPARISON")
    if removed_rows:
        print(f"Dropped {removed_rows} rows with ready_mask != 15 before prediction.")

    correct_count = 0
    for variant in VARIANT_ORDER:
        result = results[variant]
        prediction = result["prediction"]
        marker = ""
        if actual is not None:
            is_correct = prediction == actual
            correct_count += int(is_correct)
            marker = "  CORRECT" if is_correct else f"  WRONG (actual={actual})"
        print(
            f"{variant:<22} -> {prediction:<18} "
            f"margin={result['margin']:>8.3f}{marker}"
        )
        print(
            "  top: "
            + ", ".join(
                f"{item['name']}:{item['score']:.3f}"
                for item in result["ranking"]
            )
        )

    if tie:
        print("Consensus: TIE — the variants disagree.")
    else:
        print(
            f"Consensus: {consensus_name.upper().replace('_', ' ')} "
            f"({consensus_votes}/{len(VARIANT_ORDER)} variants)"
        )
    if actual is not None:
        print(f"Actual: {actual} | correct variants: {correct_count}/{len(VARIANT_ORDER)}")
    print("=" * 78)


def append_log(
    output_dir: Path,
    *,
    capture_path: Path,
    actual: str | None,
    removed_rows: int,
    results: dict[str, dict[str, Any]],
) -> None:
    row: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "capture_path": str(capture_path),
        "actual": actual,
        "removed_incomplete_rows": removed_rows,
    }
    for variant in VARIANT_ORDER:
        result = results[variant]
        row[f"{variant}_prediction"] = result["prediction"]
        row[f"{variant}_correct"] = (
            None if actual is None else result["prediction"] == actual
        )
        row[f"{variant}_margin"] = result["margin"]
        row[f"{variant}_ranking"] = json.dumps(result["ranking"])

    path = output_dir / "live_test_log.csv"
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def classify_frame(
    raw_frame: pd.DataFrame,
    *,
    models: dict[str, Any],
    index_to_name: dict[int, str],
    args: argparse.Namespace,
    actual: str | None,
    capture_path: Path,
) -> None:
    frame, removed = prepare_capture_for_prediction(
        raw_frame,
        keep_incomplete=args.keep_incomplete,
    )
    features = live_features(frame, args)
    results = {
        variant: rank_variant(
            models[variant],
            features[variant],
            index_to_name,
            args.top_k,
        )
        for variant in VARIANT_ORDER
    }
    print_results(results, actual=actual, removed_rows=removed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    append_log(
        output_dir,
        capture_path=capture_path,
        actual=actual,
        removed_rows=removed,
        results=results,
    )
    print(f"Capture: {capture_path}")
    print(f"Running log: {output_dir / 'live_test_log.csv'}\n")


def run_saved_capture(
    args: argparse.Namespace,
    models: dict[str, Any],
    index_to_name: dict[int, str],
    class_names: list[str],
) -> None:
    path = Path(args.live_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    actual = canonical_name(args.actual_label) if args.actual_label else None
    if actual is not None and actual not in class_names:
        raise ValueError(f"Unknown actual label {actual!r}. Valid labels: {class_names}")
    frame = pd.read_csv(path)
    classify_frame(
        frame,
        models=models,
        index_to_name=index_to_name,
        args=args,
        actual=actual,
        capture_path=path,
    )


def run_live(
    args: argparse.Namespace,
    models: dict[str, Any],
    index_to_name: dict[int, str],
    class_names: list[str],
) -> None:
    port = args.port or auto_detect_port()
    log_dir = Path(args.log_dir).expanduser().resolve()

    print(f"\nOpening {port} at {args.baud_rate} baud...")
    try:
        ser = serial.Serial(
            port=port,
            baudrate=args.baud_rate,
            timeout=0.10,
            write_timeout=2.0,
        )
    except serial.SerialException as exc:
        raise SystemExit(
            f"Could not open {port}: {exc}\n"
            "Close Arduino Serial Monitor and any other program using the port."
        ) from exc

    valid_display = ", ".join(class_names)
    with ser:
        import time

        time.sleep(args.startup_wait)
        drain_until_quiet(ser, quiet_s=0.35, max_wait_s=args.timeout)
        print("Connected.")
        print("At the prompt:")
        print("  • press Enter for an unlabeled capture")
        print("  • type the command name to capture and score it")
        print("  • type q to quit")
        print(f"Valid labels: {valid_display}\n")

        while True:
            try:
                entered = input("[Enter/actual label] capture  |  [q] quit > ").strip()
            except EOFError:
                break

            normalized = canonical_name(entered)
            if normalized in {"q", "quit", "exit"}:
                break
            actual = normalized or None
            if actual is not None and actual not in class_names:
                print(f"Unknown label. Valid labels: {valid_display}\n")
                continue

            try:
                raw = capture_trial(
                    ser,
                    trigger_command=args.trigger_command,
                    timeout_s=args.timeout,
                    short_window_seconds=1.55,
                    verbose=args.verbose,
                )
                capture_path = save_capture(raw, log_dir)
                classify_frame(
                    raw,
                    models=models,
                    index_to_name=index_to_name,
                    args=args,
                    actual=actual,
                    capture_path=capture_path,
                )
            except (TimeoutError, RuntimeError, ValueError) as exc:
                print(f"\nCapture/prediction failed: {exc}\n", file=sys.stderr)
            except serial.SerialException as exc:
                raise SystemExit(f"Serial connection failed: {exc}") from exc

    print("Live comparison stopped.")


def main() -> None:
    args = parse_args()
    if args.target_hz <= 0 or args.max_seconds <= 0:
        raise ValueError("--target-hz and --max-seconds must be positive.")
    if args.smoothing_window < 1 or args.smoothing_window % 2 == 0:
        raise ValueError("--smoothing-window must be an odd positive integer.")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    models, index_to_name, class_names = prepare_training(args)
    if args.live_csv:
        run_saved_capture(args, models, index_to_name, class_names)
    else:
        run_live(args, models, index_to_name, class_names)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise