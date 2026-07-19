from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.fft import dct
from scipy.signal import find_peaks

SensorMode = Literal["jaw", "ref", "both"]


def select_channels(sequence: np.ndarray, sensor_mode: SensorMode) -> np.ndarray:
    if sensor_mode == "jaw":
        return sequence[:, :6]
    if sensor_mode == "ref":
        return sequence[:, 6:]
    if sensor_mode == "both":
        return sequence
    raise ValueError(f"Unknown sensor mode: {sensor_mode}")


def _linear_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    t = np.linspace(-1.0, 1.0, len(values), dtype=np.float64)
    centered = values.astype(np.float64) - float(np.mean(values))
    denominator = float(np.dot(t, t))
    return float(np.dot(t, centered) / denominator) if denominator > 0 else 0.0


def _peak_features(values: np.ndarray) -> tuple[float, float]:
    absolute = np.abs(values.astype(np.float64))
    scale = float(np.std(absolute))
    prominence = max(0.25 * scale, 1e-6)
    peaks, properties = find_peaks(absolute, prominence=prominence)
    peak_count = float(len(peaks)) / max(len(values), 1)
    if len(peaks) == 0:
        return peak_count, 0.0
    prominences = properties.get("prominences", np.zeros(len(peaks)))
    return peak_count, float(np.mean(prominences))


def engineered_features(
    sequence: np.ndarray,
    mask: np.ndarray,
    *,
    sensor_mode: SensorMode,
    temporal_bins: int = 4,
    dct_coefficients: int = 6,
) -> np.ndarray:
    valid = select_channels(sequence[mask], sensor_mode)
    if len(valid) < 2:
        raise ValueError("A trial needs at least two valid samples for feature extraction.")

    all_features: list[float] = []
    for channel in range(valid.shape[1]):
        x = valid[:, channel].astype(np.float64)
        dx = np.diff(x)
        peak_density, mean_prominence = _peak_features(x)

        all_features.extend(
            [
                float(np.mean(x)),
                float(np.std(x)),
                float(np.min(x)),
                float(np.max(x)),
                float(np.ptp(x)),
                float(np.mean(np.abs(x))),
                float(np.sqrt(np.mean(np.square(x)))),
                float(np.mean(np.square(x))),  # motion energy
                _linear_slope(x),
                float(np.mean(dx)) if len(dx) else 0.0,
                float(np.std(dx)) if len(dx) else 0.0,
                peak_density,
                mean_prominence,
            ]
        )

        # Low-order DCT captures the broad temporal trajectory without a large feature vector.
        coefficients = dct(x, type=2, norm="ortho")
        for i in range(dct_coefficients):
            all_features.append(float(coefficients[i]) if i < len(coefficients) else 0.0)

        # Coarse temporal means/stds preserve where motion happens in the trial.
        for segment in np.array_split(x, temporal_bins):
            if len(segment) == 0:
                all_features.extend([0.0, 0.0])
            else:
                all_features.extend([float(np.mean(segment)), float(np.std(segment))])

    # Duration/valid fraction is legitimate information, especially for sentence commands.
    all_features.extend(
        [
            float(mask.sum()) / float(len(mask)),
            float(mask.sum()),
        ]
    )
    return np.asarray(all_features, dtype=np.float32)
