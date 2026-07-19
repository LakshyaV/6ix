"""Runtime preprocessing for the 6ix dual-IMU jaw-gesture models.

This is the inference-only subset of the exact jaw_gesture_ml.py used during
training. Do not change feature ordering or feature names unless the models are
retrained with the same change.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

AXES = ("x", "y", "z")
LOGICAL_NAMES = tuple(
    [f"jaw_a{axis}" for axis in AXES]
    + [f"jaw_g{axis}" for axis in AXES]
    + [f"ref_a{axis}" for axis in AXES]
    + [f"ref_g{axis}" for axis in AXES]
    + [f"rel_a{axis}" for axis in AXES]
    + [f"rel_g{axis}" for axis in AXES]
)


def _robust_scale(values: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0) * 1.4826
    return np.maximum(mad, floor)


def _moving_average(sequence: np.ndarray, width: int = 3) -> np.ndarray:
    if width <= 1 or len(sequence) < width:
        return sequence.copy()
    padded = np.pad(sequence, ((width - 1, 0), (0, 0)), mode="edge")
    kernel = np.ones(width, dtype=float) / width
    return np.vstack(
        [
            np.convolve(padded[:, col], kernel, mode="valid")
            for col in range(sequence.shape[1])
        ]
    ).T


def _resample(sequence: np.ndarray, target_samples: int) -> np.ndarray:
    if len(sequence) == 0:
        raise ValueError("Cannot resample an empty sequence")
    if len(sequence) == 1:
        return np.repeat(sequence, target_samples, axis=0)
    source_x = np.linspace(0.0, 1.0, len(sequence))
    target_x = np.linspace(0.0, 1.0, target_samples)
    return np.column_stack(
        [
            np.interp(target_x, source_x, sequence[:, index])
            for index in range(sequence.shape[1])
        ]
    )


def _build_sequence(jaw: np.ndarray, ref: np.ndarray) -> np.ndarray:
    relative = jaw - ref
    return np.concatenate([jaw, ref, relative], axis=1)


def _event_window(
    gesture_sequence: np.ndarray,
    baseline_relative: np.ndarray,
    observed_rate_hz: float,
    window_seconds: float,
    target_samples: int,
) -> np.ndarray:
    if len(gesture_sequence) < 2:
        raise ValueError("Gesture phase contains fewer than two samples")

    relative = gesture_sequence[:, 12:18]
    scale = _robust_scale(baseline_relative, floor=0.01)
    z = relative / scale
    energy = np.sqrt(np.mean(z * z, axis=1))
    energy = _moving_average(energy[:, None], width=5)[:, 0]
    peak = int(np.argmax(energy))

    desired = max(8, int(round(window_seconds * observed_rate_hz)))
    start = peak - desired // 3
    end = start + desired
    if start < 0:
        end -= start
        start = 0
    if end > len(gesture_sequence):
        start = max(0, start - (end - len(gesture_sequence)))
        end = len(gesture_sequence)
    window = gesture_sequence[start:end]
    return _moving_average(_resample(window, target_samples), width=3)


def _feature_vector(
    sequence: np.ndarray,
    channel_names: Sequence[str],
    bins: int = 8,
) -> tuple[np.ndarray, tuple[str, ...]]:
    values: list[float] = []
    names: list[str] = []
    x = np.linspace(-1.0, 1.0, len(sequence))

    for column, channel_name in enumerate(channel_names):
        signal = sequence[:, column].astype(float)
        derivative = np.diff(signal, prepend=signal[0])
        stats = {
            "mean": np.mean(signal),
            "std": np.std(signal),
            "min": np.min(signal),
            "max": np.max(signal),
            "range": np.ptp(signal),
            "rms": np.sqrt(np.mean(signal * signal)),
            "abs_mean": np.mean(np.abs(signal)),
            "p10": np.quantile(signal, 0.10),
            "p25": np.quantile(signal, 0.25),
            "median": np.quantile(signal, 0.50),
            "p75": np.quantile(signal, 0.75),
            "p90": np.quantile(signal, 0.90),
            "delta": signal[-1] - signal[0],
            "slope": np.polyfit(x, signal, deg=1)[0],
            "deriv_std": np.std(derivative),
            "deriv_max_abs": np.max(np.abs(derivative)),
            "peak_position": np.argmax(signal) / max(len(signal) - 1, 1),
            "trough_position": np.argmin(signal) / max(len(signal) - 1, 1),
        }
        for stat_name, stat_value in stats.items():
            values.append(float(stat_value))
            names.append(f"{channel_name}__{stat_name}")

        for bin_index, indices in enumerate(np.array_split(np.arange(len(signal)), bins)):
            chunk = signal[indices]
            values.append(float(np.mean(chunk)))
            names.append(f"{channel_name}__bin{bin_index}_mean")

    groups: dict[str, np.ndarray] = {}
    if sequence.shape[1] == 18:
        groups = {
            "jaw_accel_norm": sequence[:, 0:3],
            "jaw_gyro_norm": sequence[:, 3:6],
            "ref_accel_norm": sequence[:, 6:9],
            "ref_gyro_norm": sequence[:, 9:12],
            "rel_accel_norm": sequence[:, 12:15],
            "rel_gyro_norm": sequence[:, 15:18],
        }
    elif sequence.shape[1] == 6:
        prefix = str(channel_names[0]).split("_")[0] if channel_names else "signal"
        groups = {
            f"{prefix}_accel_norm": sequence[:, 0:3],
            f"{prefix}_gyro_norm": sequence[:, 3:6],
        }

    for group_name, group in groups.items():
        norm = np.linalg.norm(group, axis=1)
        for stat_name, stat_value in (
            ("mean", np.mean(norm)),
            ("std", np.std(norm)),
            ("max", np.max(norm)),
            ("rms", np.sqrt(np.mean(norm * norm))),
            ("delta", norm[-1] - norm[0]),
        ):
            values.append(float(stat_value))
            names.append(f"{group_name}__{stat_name}")

    if sequence.shape[1] == 18:
        for axis_index, axis_name in enumerate(("ax", "ay", "az", "gx", "gy", "gz")):
            jaw_signal = sequence[:, axis_index]
            ref_signal = sequence[:, 6 + axis_index]
            if np.std(jaw_signal) < 1e-8 or np.std(ref_signal) < 1e-8:
                corr = 0.0
            else:
                corr = float(np.corrcoef(jaw_signal, ref_signal)[0, 1])
            values.append(corr)
            names.append(f"jaw_ref_corr__{axis_name}")

    output = np.nan_to_num(
        np.asarray(values, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return output, tuple(names)
