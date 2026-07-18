from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    import serial
    from serial.tools import list_ports
except ImportError as exc:  # pragma: no cover - depends on the user's environment
    raise SystemExit(
        "pyserial is required. Install it with: python -m pip install pyserial"
    ) from exc

from silent_speech.data import (
    build_label_mapping,
    build_trial_records,
    load_dataframe,
)
from silent_speech.features import engineered_features


# Exact 22-column row emitted by the Arduino firmware. target_name and target_type
# are recorder-side metadata, so this live script adds them after capture.
ARDUINO_COLUMNS = [
    "time_us",
    "dt_us",
    "session_id",
    "trial_id",
    "sample_index",
    "phase",
    "jaw_ax_mg",
    "jaw_ay_mg",
    "jaw_az_mg",
    "jaw_gx_cdeg_s",
    "jaw_gy_cdeg_s",
    "jaw_gz_cdeg_s",
    "ref_ax_mg",
    "ref_ay_mg",
    "ref_az_mg",
    "ref_gx_cdeg_s",
    "ref_gy_cdeg_s",
    "ref_gz_cdeg_s",
    "label",
    "ready_mask",
    "late_us",
    "read_span_us",
]

SHORT_TARGET_TYPES = {"word", "gesture", "unknown"}


@dataclass
class LiveModel:
    pipeline: Pipeline
    index_to_name: dict[int, str]
    index_to_type: dict[int, str]
    mode: str
    target_hz: float
    max_seconds: float
    smoothing_window: int


@dataclass
class Prediction:
    predicted_index: int
    predicted_name: str
    predicted_type: str
    top_candidates: list[tuple[str, str, float, float]]
    margin: float
    relative_confidence: float
    mouth_samples: int
    duration_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture one paired-IMU trial from Arduino and classify it live with "
            "the same engineered-feature linear SVM used by train.py."
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
        choices=["short", "hybrid"],
        default="short",
        help=(
            "short: highest-reliability word/gesture demo using a 1.5 s trial. "
            "hybrid: also supports the two sentence labels using a 3 s trial and "
            "separate short/long feature windows."
        ),
    )
    parser.add_argument(
        "--trigger-command",
        default=None,
        help=(
            "Character sent to Arduino to start a trial. Defaults to 'u' in short "
            "mode and 'w' in hybrid mode."
        ),
    )
    parser.add_argument("--target-hz", type=float, default=50.0)
    parser.add_argument("--max-seconds", type=float, default=3.2)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument(
        "--short-window-seconds",
        type=float,
        default=1.55,
        help="Short-command phase-1 window used by hybrid inference.",
    )
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.15,
        help=(
            "Mark a prediction uncertain when the best SVM decision score is less "
            "than this far above the runner-up. Tune after several live attempts."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-out",
        default="runs/live_demo/live_svm.joblib",
        help="Where to save the final all-data demo SVM bundle.",
    )
    parser.add_argument(
        "--log-dir",
        default="runs/live_demo/captures",
        help="Directory in which every captured raw trial is saved.",
    )
    parser.add_argument(
        "--startup-wait",
        type=float,
        default=2.0,
        help="Seconds to wait after opening serial in case the board resets.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def canonical_target_type(value: object) -> str:
    return str(value).strip().lower()


def train_final_demo_svm(args: argparse.Namespace) -> LiveModel:
    """Fit a final demo model on all validated data.

    This is intentionally different from evaluation: train.py kept a test split to
    estimate accuracy. Once that diagnostic is complete, the hackathon demo model
    can use every clean trial for maximum reliability.
    """
    df = load_dataframe(args.csv)
    df["target_type"] = df["target_type"].map(canonical_target_type)

    if args.mode == "short":
        df = df.loc[df["target_type"].isin(SHORT_TARGET_TYPES)].copy()
        if df.empty:
            raise ValueError("No short word/gesture/unknown trials remain.")

    original_to_index, index_to_name = build_label_mapping(df)

    # Store one type per remapped class. Fail loudly if metadata is inconsistent.
    original_to_type: dict[int, str] = {}
    for original_label, group in df.groupby("label", sort=False):
        types = sorted(set(group["target_type"].map(canonical_target_type)))
        if len(types) != 1:
            raise ValueError(
                f"Original label {int(original_label)} has inconsistent target types: {types}"
            )
        original_to_type[int(original_label)] = types[0]
    index_to_type = {
        mapped: original_to_type[original]
        for original, mapped in original_to_index.items()
    }

    records = build_trial_records(
        df,
        region="mouth",
        target_hz=args.target_hz,
        max_seconds=args.max_seconds,
        smoothing_window=args.smoothing_window,
    )
    labels = np.asarray(
        [original_to_index[record.label_original] for record in records],
        dtype=np.int64,
    )
    features = np.stack(
        [
            engineered_features(record.sequence, record.mask, sensor_mode="both")
            for record in records
        ]
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LinearSVC(
                    C=0.5,
                    class_weight="balanced",
                    dual="auto",
                    max_iter=20_000,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    pipeline.fit(features, labels)

    model = LiveModel(
        pipeline=pipeline,
        index_to_name=index_to_name,
        index_to_type=index_to_type,
        mode=args.mode,
        target_hz=args.target_hz,
        max_seconds=args.max_seconds,
        smoothing_window=args.smoothing_window,
    )

    output_path = Path(args.model_out).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": model.pipeline,
            "index_to_name": model.index_to_name,
            "index_to_type": model.index_to_type,
            "mode": model.mode,
            "target_hz": model.target_hz,
            "max_seconds": model.max_seconds,
            "smoothing_window": model.smoothing_window,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "training_csv": str(Path(args.csv).expanduser().resolve()),
            "num_trials": len(records),
        },
        output_path,
    )

    class_list = ", ".join(index_to_name[i] for i in sorted(index_to_name))
    print(f"Trained final live SVM on {len(records)} trials and {len(index_to_name)} classes.")
    print(f"Classes: {class_list}")
    print(f"Saved live model: {output_path}")
    return model


def auto_detect_port() -> str:
    ports = list(list_ports.comports())
    likely = []
    for port in ports:
        device = port.device
        description = f"{port.description} {port.manufacturer or ''}".lower()
        if (
            "usbmodem" in device.lower()
            or "usbserial" in device.lower()
            or "ttyacm" in device.lower()
            or "arduino" in description
        ):
            likely.append(device)

    likely = sorted(set(likely))
    if len(likely) == 1:
        return likely[0]

    available = "\n".join(
        f"  {p.device}  ({p.description})" for p in ports
    ) or "  <none>"
    if not likely:
        raise RuntimeError(
            "Could not auto-detect an Arduino serial port. Available ports:\n"
            f"{available}\nPass the correct one with --port."
        )
    raise RuntimeError(
        "Multiple likely USB serial ports were found:\n  "
        + "\n  ".join(likely)
        + "\nPass the Arduino port explicitly with --port."
    )


def parse_sensor_row(line: str) -> dict[str, int] | None:
    try:
        fields = next(csv.reader([line]))
    except csv.Error:
        return None
    if len(fields) < len(ARDUINO_COLUMNS):
        return None

    values = fields[: len(ARDUINO_COLUMNS)]
    try:
        # Firmware emits integer-valued measurements. int(float(...)) also tolerates
        # accidental representations such as "15.0" without silently accepting NaN.
        parsed = [int(float(value.strip())) for value in values]
    except (TypeError, ValueError, OverflowError):
        return None
    return dict(zip(ARDUINO_COLUMNS, parsed, strict=True))


def drain_until_quiet(
    ser: serial.Serial,
    *,
    quiet_s: float = 0.35,
    max_wait_s: float = 12.0,
) -> None:
    """Wait until the board has stopped streaming, then clear stale bytes.

    A fixed 0.2-second drain can cut into an already-running trial. Waiting for a
    quiet interval prevents the next capture from accidentally using only the
    baseline or only the articulation portion of a previous trial.
    """
    deadline = time.monotonic() + max_wait_s
    last_rx = time.monotonic()

    while time.monotonic() < deadline:
        if ser.in_waiting:
            ser.readline()
            last_rx = time.monotonic()
            continue
        if time.monotonic() - last_rx >= quiet_s:
            ser.reset_input_buffer()
            return
        time.sleep(0.01)

    raise TimeoutError(
        "The Arduino never became idle. Reset the board, close Serial Monitor, "
        "and try again."
    )


def parse_marker(line: str) -> tuple[str, dict[str, str]] | None:
    """Parse markers such as '# TRIAL_START,session=1,trial=42,...'."""
    if not line.startswith("#"):
        return None

    body = line[1:].strip()
    if not body:
        return None

    parts = [part.strip() for part in body.split(",")]
    marker = parts[0].upper()
    values: dict[str, str] = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            values[key.strip().lower()] = value.strip()
    return marker, values


def capture_trial(
    ser: serial.Serial,
    *,
    trigger_command: str,
    timeout_s: float,
    short_window_seconds: float,
    verbose: bool,
) -> pd.DataFrame:
    if not trigger_command:
        raise ValueError("trigger_command cannot be empty.")

    # Finish/ignore any partial trial left over from an earlier failed attempt.
    drain_until_quiet(ser, quiet_s=0.35, max_wait_s=timeout_s)

    ser.write((trigger_command[0] + "\n").encode("utf-8"))
    ser.flush()

    print("\nRecording started. Keep your head still during the initial baseline...")
    deadline = time.monotonic() + timeout_s
    rows: list[dict[str, int]] = []
    active_key: tuple[int, int] | None = None
    last_phase: int | None = None
    announced_phase: int | None = None
    mouth_start_us: int | None = None
    short_window_notice_printed = False
    saw_start_message = False
    saw_done_message = False

    while time.monotonic() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        marker_info = parse_marker(line)
        if marker_info is not None:
            marker, values = marker_info

            if marker == "TRIAL_START":
                try:
                    marker_key = (int(values["session"]), int(values["trial"]))
                except (KeyError, ValueError):
                    if verbose:
                        print(f"[ignored malformed marker] {line}")
                    continue

                # The first TRIAL_START observed after our trigger defines the one
                # and only trial accepted for this prediction.
                if active_key is None:
                    active_key = marker_key
                    saw_start_message = True
                    if verbose:
                        print(f"[board] {line}")
                elif marker_key != active_key:
                    raise RuntimeError(
                        f"A second trial started before trial {active_key} completed. "
                        "Reset the Arduino and try again."
                    )
                continue

            if marker == "MOUTH_NOW":
                if announced_phase != 1:
                    print("MOUTH NOW — silently articulate your command.")
                    announced_phase = 1
                continue

            if marker == "REST_NOW":
                if announced_phase != 2:
                    print("Done — relax and remain still.")
                    announced_phase = 2
                continue

            if marker == "TRIAL_DONE":
                try:
                    done_key = (int(values["session"]), int(values["trial"]))
                except (KeyError, ValueError):
                    done_key = active_key

                if active_key is not None and done_key == active_key:
                    saw_done_message = True
                    break
                if verbose:
                    print(f"[ignored stale marker] {line}")
                continue

            if verbose:
                print(f"[board] {line}")
            continue

        parsed = parse_sensor_row(line)
        if parsed is None:
            if verbose:
                print(f"[ignored] {line}")
            continue

        key = (parsed["session_id"], parsed["trial_id"])

        # Never let leftover numeric rows define the capture. We wait for the
        # explicit TRIAL_START emitted in response to our trigger command.
        if active_key is None:
            if verbose:
                print(f"[ignored row before TRIAL_START] session={key[0]} trial={key[1]}")
            continue
        if key != active_key:
            if verbose:
                print(f"[ignored row from other trial] session={key[0]} trial={key[1]}")
            continue

        phase = parsed["phase"]
        if phase != last_phase:
            if phase == 0 and announced_phase != 0:
                print("Baseline...")
                announced_phase = 0
            elif phase == 1:
                if announced_phase != 1:
                    print("MOUTH NOW — silently articulate your command.")
                    announced_phase = 1
                mouth_start_us = parsed["time_us"]
            elif phase == 2 and announced_phase != 2:
                print("Done — relax and remain still.")
                announced_phase = 2
            last_phase = phase

        if (
            phase == 1
            and mouth_start_us is not None
            and not short_window_notice_printed
            and (parsed["time_us"] - mouth_start_us)
            >= short_window_seconds * 1_000_000.0
        ):
            print("Short-command window complete; stay still unless finishing a sentence.")
            short_window_notice_printed = True

        rows.append(parsed)

    if not saw_start_message:
        raise TimeoutError(
            "No TRIAL_START marker was received. Confirm that the live Arduino "
            "firmware is uploaded and the baud rate is 921600."
        )
    if not rows:
        raise TimeoutError(
            "The trial started, but no sensor rows were received. Check the serial "
            "connection and both IMUs."
        )

    frame = pd.DataFrame(rows)
    frame["label"] = 0
    frame["target_name"] = "live"
    frame["target_type"] = "unknown"

    phase_counts = {int(k): int(v) for k, v in frame["phase"].value_counts().to_dict().items()}
    if phase_counts.get(0, 0) < 10 or phase_counts.get(1, 0) < 10:
        raise RuntimeError(
            f"Incomplete trial {active_key}: phase sample counts were {phase_counts}. "
            "Need both baseline phase 0 and articulation phase 1. Reset the board "
            "and retry if this repeats."
        )
    if not saw_done_message:
        raise TimeoutError(
            f"Trial {active_key} did not emit TRIAL_DONE before timeout. "
            f"Captured phase counts: {phase_counts}."
        )

    if verbose:
        print(f"[capture] trial={active_key} phase_counts={phase_counts}")
    return frame

def crop_mouth_window(frame: pd.DataFrame, seconds: float) -> pd.DataFrame:
    output = frame.copy()
    mouth = output.loc[output["phase"] == 1]
    if mouth.empty:
        raise ValueError("Captured trial contains no phase-1 mouth samples.")
    start_us = int(mouth["time_us"].min())
    cutoff_us = start_us + int(seconds * 1_000_000.0)
    keep = (output["phase"] != 1) | (output["time_us"] <= cutoff_us)
    return output.loc[keep].copy()


def decision_scores(pipeline: Pipeline, feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(pipeline.decision_function(feature[None, :]), dtype=np.float64)
    if scores.ndim == 2:
        scores = scores[0]
    classes = np.asarray(pipeline.named_steps["classifier"].classes_, dtype=np.int64)

    # LinearSVC returns one score per class for this multiclass problem.
    if scores.ndim != 1 or len(scores) != len(classes):
        raise RuntimeError(
            f"Unexpected LinearSVC score shape {scores.shape} for classes {classes.shape}."
        )
    return classes, scores


def stable_softmax(scores: np.ndarray) -> np.ndarray:
    centered = scores - np.max(scores)
    exp = np.exp(np.clip(centered, -60.0, 0.0))
    denominator = float(exp.sum())
    return exp / denominator if denominator > 0 else np.full_like(exp, 1.0 / len(exp))


def predict_live(
    model: LiveModel,
    frame: pd.DataFrame,
    *,
    short_window_seconds: float,
    top_k: int,
) -> Prediction:
    long_records = build_trial_records(
        frame,
        region="mouth",
        target_hz=model.target_hz,
        max_seconds=model.max_seconds,
        smoothing_window=model.smoothing_window,
    )
    if len(long_records) != 1:
        raise RuntimeError(f"Expected exactly one live trial, got {len(long_records)}.")
    long_record = long_records[0]
    long_feature = engineered_features(long_record.sequence, long_record.mask, sensor_mode="both")
    classes, long_scores = decision_scores(model.pipeline, long_feature)

    if model.mode == "hybrid":
        short_frame = crop_mouth_window(frame, short_window_seconds)
        short_record = build_trial_records(
            short_frame,
            region="mouth",
            target_hz=model.target_hz,
            max_seconds=model.max_seconds,
            smoothing_window=model.smoothing_window,
        )[0]
        short_feature = engineered_features(
            short_record.sequence, short_record.mask, sensor_mode="both"
        )
        short_classes, short_scores = decision_scores(model.pipeline, short_feature)
        if not np.array_equal(classes, short_classes):
            raise RuntimeError("SVM class ordering changed between live feature windows.")

        combined_scores = short_scores.copy()
        for position, class_index in enumerate(classes):
            if model.index_to_type[int(class_index)] == "sentence":
                combined_scores[position] = long_scores[position]
        effective_record = long_record
    else:
        combined_scores = long_scores
        effective_record = long_record

    probabilities = stable_softmax(combined_scores)
    order = np.argsort(combined_scores)[::-1]
    best_position = int(order[0])
    second_position = int(order[1]) if len(order) > 1 else best_position
    predicted_index = int(classes[best_position])
    margin = float(combined_scores[best_position] - combined_scores[second_position])

    candidates: list[tuple[str, str, float, float]] = []
    for position in order[: max(1, top_k)]:
        class_index = int(classes[int(position)])
        candidates.append(
            (
                model.index_to_name[class_index],
                model.index_to_type[class_index],
                float(combined_scores[int(position)]),
                float(probabilities[int(position)]),
            )
        )

    return Prediction(
        predicted_index=predicted_index,
        predicted_name=model.index_to_name[predicted_index],
        predicted_type=model.index_to_type[predicted_index],
        top_candidates=candidates,
        margin=margin,
        relative_confidence=float(probabilities[best_position]),
        mouth_samples=int((frame["phase"] == 1).sum()),
        duration_s=float(effective_record.duration_s),
    )


def save_capture(frame: pd.DataFrame, log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    path = log_dir / f"live_trial_{timestamp}.csv"
    frame.to_csv(path, index=False)
    return path


def print_prediction(prediction: Prediction, min_margin: float) -> None:
    uncertain = prediction.margin < min_margin
    status = "UNCERTAIN" if uncertain else "DETECTED"
    display_name = prediction.predicted_name.replace("_", " ").upper()
    print("\n" + "=" * 60)
    print(f"{status}: {display_name}")
    print(
        f"SVM margin: {prediction.margin:.3f} | "
        f"relative confidence: {prediction.relative_confidence:.1%} | "
        f"mouth samples: {prediction.mouth_samples}"
    )
    print("Top candidates:")
    for rank, (name, target_type, score, relative) in enumerate(
        prediction.top_candidates, start=1
    ):
        print(
            f"  {rank}. {name.replace('_', ' '):<24} "
            f"type={target_type:<8} score={score:>7.3f} relative={relative:>6.1%}"
        )
    if uncertain:
        print("Low separation between the top two classes — repeat the command.")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    if args.target_hz <= 0 or args.max_seconds <= 0:
        raise ValueError("--target-hz and --max-seconds must be positive.")
    if args.smoothing_window < 1 or args.smoothing_window % 2 == 0:
        raise ValueError("--smoothing-window must be an odd positive integer.")
    if args.short_window_seconds <= 0:
        raise ValueError("--short-window-seconds must be positive.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive.")

    trigger_command = args.trigger_command or ("u" if args.mode == "short" else "w")
    model = train_final_demo_svm(args)
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
            f"Could not open {port}: {exc}\nClose Arduino Serial Monitor and any recorder using the port."
        ) from exc

    with ser:
        time.sleep(args.startup_wait)
        drain_until_quiet(ser, quiet_s=0.35, max_wait_s=args.timeout)
        print("Connected.")
        if args.mode == "short":
            print("Live mode: short words/gesture only. Press Enter, then mouth one command.")
        else:
            print(
                "Live mode: hybrid. Press Enter, then start immediately when MOUTH NOW appears.\n"
                "For a short command, finish within about 1.5 s and stay still; sentences may use the full window."
            )
        print("Type q and press Enter to quit.\n")

        while True:
            try:
                command = input("[Enter] detect command  |  [q] quit > ").strip().lower()
            except EOFError:
                break
            if command in {"q", "quit", "exit"}:
                break
            if command:
                print("Unknown input. Press Enter to detect or q to quit.")
                continue

            try:
                frame = capture_trial(
                    ser,
                    trigger_command=trigger_command,
                    timeout_s=args.timeout,
                    short_window_seconds=args.short_window_seconds,
                    verbose=args.verbose,
                )
                capture_path = save_capture(frame, log_dir)
                prediction = predict_live(
                    model,
                    frame,
                    short_window_seconds=args.short_window_seconds,
                    top_k=args.top_k,
                )
                print_prediction(prediction, args.min_margin)
                print(f"Captured trial saved to: {capture_path}\n")
            except (TimeoutError, RuntimeError, ValueError) as exc:
                print(f"\nCapture/prediction failed: {exc}\n", file=sys.stderr)
            except serial.SerialException as exc:
                raise SystemExit(f"Serial connection failed: {exc}") from exc

    print("Live demo stopped.")


if __name__ == "__main__":
    main()