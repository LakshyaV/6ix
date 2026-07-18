from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from silent_speech.data import (
    SENSOR_COLUMNS,
    build_label_mapping,
    build_trial_records,
    load_dataframe,
)
from silent_speech.features import engineered_features

SHORT_TARGET_TYPES = {"word", "gesture", "unknown"}


@dataclass(frozen=True)
class TrialData:
    key: tuple[int, int]
    frame: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare jaw-only axis features, current jaw+reference axis features, "
            "and rotation-resistant paired-IMU features."
        )
    )
    parser.add_argument("--csv", required=True, help="Cleaned training CSV.")
    parser.add_argument(
        "--live-csv",
        default=None,
        help="Optional saved live capture to classify with all three models.",
    )
    parser.add_argument(
        "--actual-label",
        default=None,
        help="Optional true name of the live capture, such as start.",
    )
    parser.add_argument(
        "--mode",
        choices=["short", "all"],
        default="short",
        help="short excludes sentence commands; all keeps every active class.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-hz", type=float, default=50.0)
    parser.add_argument("--max-seconds", type=float, default=3.2)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument(
        "--keep-incomplete",
        action="store_true",
        help="Keep live rows whose ready_mask is not 15. By default they are dropped.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/feature_comparison",
        help="Directory for trained model bundles and results.",
    )
    return parser.parse_args()


def make_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LinearSVC(
                    C=0.5,
                    class_weight="balanced",
                    dual="auto",
                    max_iter=20_000,
                    random_state=seed,
                ),
            ),
        ]
    )


def safe_unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return vector / norm


def decompose_against_gravity(
    vectors: np.ndarray, gravity_unit: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return signed vertical, horizontal magnitude, and total magnitude.

    These scalars are invariant to yaw and remain stable under pitch/roll changes
    estimated from the trial's baseline gravity direction.
    """
    vertical = vectors @ gravity_unit
    horizontal_vectors = vectors - vertical[:, None] * gravity_unit[None, :]
    horizontal = np.linalg.norm(horizontal_vectors, axis=1)
    magnitude = np.linalg.norm(vectors, axis=1)
    return vertical, horizontal, magnitude


def resample_values(
    values: np.ndarray,
    times_us: np.ndarray,
    *,
    target_hz: float,
    max_seconds: float,
) -> np.ndarray:
    order = np.argsort(times_us, kind="mergesort")
    times = np.asarray(times_us, dtype=np.float64)[order]
    values = np.asarray(values, dtype=np.float64)[order]

    unique_times, inverse = np.unique(times, return_inverse=True)
    if len(unique_times) != len(times):
        sums = np.zeros((len(unique_times), values.shape[1]), dtype=np.float64)
        counts = np.zeros(len(unique_times), dtype=np.int64)
        np.add.at(sums, inverse, values)
        np.add.at(counts, inverse, 1)
        values = sums / counts[:, None]
        times = unique_times

    times_s = (times - times[0]) / 1_000_000.0
    step = 1.0 / target_hz
    max_length = int(np.ceil(target_hz * max_seconds))
    new_times = np.arange(0.0, times_s[-1] + step * 0.25, step)[:max_length]
    if len(new_times) < 2:
        raise ValueError("Trial is too short after resampling.")

    return np.column_stack(
        [np.interp(new_times, times_s, values[:, index]) for index in range(values.shape[1])]
    ).astype(np.float32)


def rotation_resistant_sequence(
    trial: pd.DataFrame,
    *,
    target_hz: float,
    max_seconds: float,
    smoothing_window: int,
) -> np.ndarray:
    baseline = trial.loc[trial["phase"] == 0]
    mouth = trial.loc[trial["phase"] == 1]
    if len(baseline) < 2 or len(mouth) < 2:
        raise ValueError("Trial needs baseline phase 0 and mouth phase 1.")

    baseline_values = baseline[SENSOR_COLUMNS].to_numpy(dtype=np.float64)
    mouth_values = mouth[SENSOR_COLUMNS].to_numpy(dtype=np.float64)

    jaw_acc_bias = baseline_values[:, 0:3].mean(axis=0)
    jaw_gyro_bias = baseline_values[:, 3:6].mean(axis=0)
    ref_acc_bias = baseline_values[:, 6:9].mean(axis=0)
    ref_gyro_bias = baseline_values[:, 9:12].mean(axis=0)

    jaw_gravity = safe_unit(jaw_acc_bias)
    ref_gravity = safe_unit(ref_acc_bias)

    jaw_acc = mouth_values[:, 0:3] - jaw_acc_bias
    jaw_gyro = mouth_values[:, 3:6] - jaw_gyro_bias
    ref_acc = mouth_values[:, 6:9] - ref_acc_bias
    ref_gyro = mouth_values[:, 9:12] - ref_gyro_bias

    jav, jah, jam = decompose_against_gravity(jaw_acc, jaw_gravity)
    jgv, jgh, jgm = decompose_against_gravity(jaw_gyro, jaw_gravity)
    rav, rah, ram = decompose_against_gravity(ref_acc, ref_gravity)
    rgv, rgh, rgm = decompose_against_gravity(ref_gyro, ref_gravity)

    jaw_scalars = np.column_stack([jav, jah, jam, jgv, jgh, jgm])
    ref_scalars = np.column_stack([rav, rah, ram, rgv, rgh, rgm])

    # Scalar residuals remain valid despite independent yaw rotations of the IMUs.
    residuals = jaw_scalars - ref_scalars
    interactions = jaw_scalars * ref_scalars
    values = np.column_stack([jaw_scalars, ref_scalars, residuals, interactions])

    if smoothing_window > 1:
        values = uniform_filter1d(
            values,
            size=smoothing_window,
            axis=0,
            mode="nearest",
        )

    return resample_values(
        values,
        mouth["time_us"].to_numpy(),
        target_hz=target_hz,
        max_seconds=max_seconds,
    )


def linear_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    t = np.linspace(-1.0, 1.0, len(values), dtype=np.float64)
    centered = values.astype(np.float64) - float(np.mean(values))
    denominator = float(np.dot(t, t))
    return float(np.dot(t, centered) / denominator) if denominator > 0 else 0.0


def peak_features(values: np.ndarray) -> tuple[float, float]:
    absolute = np.abs(values.astype(np.float64))
    prominence = max(0.25 * float(np.std(absolute)), 1e-6)
    peaks, properties = find_peaks(absolute, prominence=prominence)
    density = float(len(peaks)) / max(len(values), 1)
    if len(peaks) == 0:
        return density, 0.0
    return density, float(np.mean(properties.get("prominences", np.zeros(len(peaks)))))


def sequence_statistics(
    sequence: np.ndarray,
    *,
    temporal_bins: int = 4,
    dct_coefficients: int = 6,
    max_length: int = 160,
) -> np.ndarray:
    output: list[float] = []
    for channel in range(sequence.shape[1]):
        x = sequence[:, channel].astype(np.float64)
        dx = np.diff(x)
        peak_density, mean_prominence = peak_features(x)
        output.extend(
            [
                float(np.mean(x)),
                float(np.std(x)),
                float(np.min(x)),
                float(np.max(x)),
                float(np.ptp(x)),
                float(np.mean(np.abs(x))),
                float(np.sqrt(np.mean(np.square(x)))),
                float(np.mean(np.square(x))),
                linear_slope(x),
                float(np.mean(dx)) if len(dx) else 0.0,
                float(np.std(dx)) if len(dx) else 0.0,
                peak_density,
                mean_prominence,
            ]
        )
        coefficients = dct(x, type=2, norm="ortho")
        output.extend(
            float(coefficients[index]) if index < len(coefficients) else 0.0
            for index in range(dct_coefficients)
        )
        for segment in np.array_split(x, temporal_bins):
            output.extend(
                [
                    float(np.mean(segment)) if len(segment) else 0.0,
                    float(np.std(segment)) if len(segment) else 0.0,
                ]
            )

    output.extend([float(len(sequence)) / float(max_length), float(len(sequence))])
    return np.asarray(output, dtype=np.float32)


def trial_frames(df: pd.DataFrame) -> dict[tuple[int, int], pd.DataFrame]:
    return {
        (int(session_id), int(trial_id)): frame.copy()
        for (session_id, trial_id), frame in df.groupby(
            ["session_id", "trial_id"], sort=False
        )
    }


def prepare_live_frame(path: str, *, keep_incomplete: bool) -> pd.DataFrame:
    frame = pd.read_csv(Path(path).expanduser().resolve())
    if "ready_mask" in frame.columns and not keep_incomplete:
        complete = (pd.to_numeric(frame["ready_mask"], errors="coerce") == 15)
        removed = int((~complete).sum())
        frame = frame.loc[complete].copy()
        print(f"Dropped {removed} live rows with ready_mask != 15.")
    frame["label"] = 0
    frame["target_name"] = "live"
    frame["target_type"] = "unknown"
    return frame


def score_names(
    pipeline: Pipeline,
    feature: np.ndarray,
    index_to_name: dict[int, str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    scores = np.asarray(pipeline.decision_function(feature[None, :]), dtype=np.float64)
    if scores.ndim == 2:
        scores = scores[0]
    classes = np.asarray(pipeline.named_steps["classifier"].classes_, dtype=np.int64)
    order = np.argsort(scores)[::-1]
    return [
        (index_to_name[int(classes[position])], float(scores[position]))
        for position in order[:top_k]
    ]


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")

    df = load_dataframe(args.csv)
    df["target_type"] = df["target_type"].astype(str).str.strip().str.lower()
    if args.mode == "short":
        df = df.loc[df["target_type"].isin(SHORT_TARGET_TYPES)].copy()

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
        [original_to_index[record.label_original] for record in records], dtype=np.int64
    )

    min_class_count = int(np.bincount(labels).min())
    if args.folds > min_class_count:
        raise ValueError(
            f"--folds={args.folds} exceeds smallest class count {min_class_count}."
        )

    axis_jaw = np.stack(
        [
            engineered_features(record.sequence, record.mask, sensor_mode="jaw")
            for record in records
        ]
    )
    axis_ref = np.stack(
        [
            engineered_features(record.sequence, record.mask, sensor_mode="ref")
            for record in records
        ]
    )
    axis_both = np.stack(
        [
            engineered_features(record.sequence, record.mask, sensor_mode="both")
            for record in records
        ]
    )
    invariant = np.stack(
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
    )

    feature_sets = {
        "jaw_axis": axis_jaw,
        "ref_axis_diagnostic": axis_ref,
        "jaw_ref_axis": axis_both,
        "rotation_resistant": invariant,
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows: list[dict[str, object]] = []
    fitted: dict[str, Pipeline] = {}

    print(f"Training trials: {len(records)} | classes: {len(index_to_name)}")
    print("Random same-session cross-validation (diagnostic only):")
    for name, features in feature_sets.items():
        pipeline = make_pipeline(args.seed)
        predicted = cross_val_predict(pipeline, features, labels, cv=cv)
        metrics = {
            "variant": name,
            "accuracy": float(accuracy_score(labels, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "macro_f1": float(f1_score(labels, predicted, average="macro")),
        }
        rows.append(metrics)
        print(
            f"  {name:<20} accuracy={metrics['accuracy']:.3f} "
            f"balanced={metrics['balanced_accuracy']:.3f} "
            f"macro_f1={metrics['macro_f1']:.3f}"
        )
        pipeline.fit(features, labels)
        fitted[name] = pipeline
        joblib.dump(
            {
                "pipeline": pipeline,
                "variant": name,
                "index_to_name": index_to_name,
                "target_hz": args.target_hz,
                "max_seconds": args.max_seconds,
                "smoothing_window": args.smoothing_window,
            },
            output_dir / f"{name}.joblib",
        )

    pd.DataFrame(rows).to_csv(output_dir / "same_session_cv.csv", index=False)

    if args.live_csv:
        live = prepare_live_frame(args.live_csv, keep_incomplete=args.keep_incomplete)
        live_record = build_trial_records(
            live,
            region="mouth",
            target_hz=args.target_hz,
            max_seconds=args.max_seconds,
            smoothing_window=args.smoothing_window,
        )[0]
        live_features = {
            "jaw_axis": engineered_features(
                live_record.sequence, live_record.mask, sensor_mode="jaw"
            ),
            "ref_axis_diagnostic": engineered_features(
                live_record.sequence, live_record.mask, sensor_mode="ref"
            ),
            "jaw_ref_axis": engineered_features(
                live_record.sequence, live_record.mask, sensor_mode="both"
            ),
            "rotation_resistant": sequence_statistics(
                rotation_resistant_sequence(
                    live,
                    target_hz=args.target_hz,
                    max_seconds=args.max_seconds,
                    smoothing_window=args.smoothing_window,
                ),
                max_length=int(np.ceil(args.target_hz * args.max_seconds)),
            ),
        }

        print("\nLive capture predictions:")
        live_rows: list[dict[str, object]] = []
        actual = args.actual_label.strip().lower() if args.actual_label else None
        for name, feature in live_features.items():
            ranking = score_names(fitted[name], feature, index_to_name)
            best_name, best_score = ranking[0]
            marker = ""
            if actual is not None:
                marker = " CORRECT" if best_name == actual else f" WRONG (actual={actual})"
            print(f"  {name:<20} -> {best_name}{marker}")
            print("    " + ", ".join(f"{label}:{score:.3f}" for label, score in ranking))
            live_rows.append(
                {
                    "variant": name,
                    "prediction": best_name,
                    "actual": actual,
                    "correct": None if actual is None else best_name == actual,
                    "top_five": json.dumps(ranking),
                }
            )
        pd.DataFrame(live_rows).to_csv(output_dir / "live_predictions.csv", index=False)

    print(f"\nArtifacts saved to: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise