from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

Region = Literal["mouth", "rest"]

JAW_COLUMNS = [
    "jaw_ax_mg",
    "jaw_ay_mg",
    "jaw_az_mg",
    "jaw_gx_cdeg_s",
    "jaw_gy_cdeg_s",
    "jaw_gz_cdeg_s",
]
REF_COLUMNS = [
    "ref_ax_mg",
    "ref_ay_mg",
    "ref_az_mg",
    "ref_gx_cdeg_s",
    "ref_gy_cdeg_s",
    "ref_gz_cdeg_s",
]
SENSOR_COLUMNS = JAW_COLUMNS + REF_COLUMNS

REQUIRED_COLUMNS = {
    "time_us",
    "session_id",
    "trial_id",
    "phase",
    "label",
    "target_name",
    *SENSOR_COLUMNS,
}

EXCLUDED_TARGETS = {"meme", "generator", "double_clench"}


@dataclass(frozen=True)
class TrialKey:
    session_id: int
    trial_id: int


@dataclass
class TrialRecord:
    key: TrialKey
    label_original: int
    target_name: str
    target_type: str
    sequence: np.ndarray  # [T, 12], padded with zero
    mask: np.ndarray  # [T], bool
    duration_s: float

    @property
    def valid_length(self) -> int:
        return int(self.mask.sum())


def _canonical_target(value: object) -> str:
    return str(value).strip().lower()


def load_dataframe(
    csv_path: str | Path,
    *,
    remove_first_stop: bool = False,
    excluded_targets: Iterable[str] = EXCLUDED_TARGETS,
) -> pd.DataFrame:
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.copy()
    df["target_name"] = df["target_name"].map(_canonical_target)
    if "target_type" not in df.columns:
        df["target_type"] = "unknown"

    # Numeric coercion with explicit failure instead of silently training on bad rows.
    numeric_columns = [
        "time_us",
        "session_id",
        "trial_id",
        "phase",
        "label",
        *SENSOR_COLUMNS,
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    excluded = {_canonical_target(x) for x in excluded_targets}
    df = df.loc[~df["target_name"].isin(excluded)].copy()

    if remove_first_stop:
        stop_trials = (
            df.loc[df["target_name"] == "stop", ["session_id", "trial_id", "time_us"]]
            .groupby(["session_id", "trial_id"], as_index=False)["time_us"]
            .min()
            .sort_values("time_us")
        )
        if stop_trials.empty:
            raise ValueError("--remove-first-stop was requested, but no STOP trial exists.")
        first = stop_trials.iloc[0]
        remove_mask = (
            (df["session_id"] == first["session_id"])
            & (df["trial_id"] == first["trial_id"])
        )
        removed = int(remove_mask.sum())
        print(
            "Removed first STOP trial from input: "
            f"session={int(first['session_id'])}, trial={int(first['trial_id'])}, rows={removed}"
        )
        df = df.loc[~remove_mask].copy()

    # Keep acquisition order deterministic.
    df = df.sort_values(["session_id", "trial_id", "time_us"], kind="mergesort")
    if df.empty:
        raise ValueError("No usable rows remain after filtering.")

    return df.reset_index(drop=True)


def summarize_trials(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["label", "target_name", "target_type"], as_index=False)
        .agg(
            trials=("trial_id", "nunique"),
            sessions=("session_id", "nunique"),
            rows=("trial_id", "size"),
        )
        .sort_values("label")
        .reset_index(drop=True)
    )


def build_label_mapping(df: pd.DataFrame) -> tuple[dict[int, int], dict[int, str]]:
    label_name_pairs = (
        df[["label", "target_name"]]
        .drop_duplicates()
        .sort_values("label")
        .itertuples(index=False, name=None)
    )
    original_to_index: dict[int, int] = {}
    index_to_name: dict[int, str] = {}
    for index, (original_label, name) in enumerate(label_name_pairs):
        original = int(original_label)
        original_to_index[original] = index
        index_to_name[index] = str(name)
    return original_to_index, index_to_name


def _resample_by_time(
    values: np.ndarray,
    times_us: np.ndarray,
    *,
    target_hz: float,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if values.ndim != 2 or values.shape[1] != len(SENSOR_COLUMNS):
        raise ValueError(f"Expected [N, 12] sensor values, got {values.shape}")
    if len(values) < 2:
        raise ValueError("A trial region must contain at least two samples.")

    order = np.argsort(times_us, kind="mergesort")
    times = np.asarray(times_us, dtype=np.float64)[order]
    values = np.asarray(values, dtype=np.float32)[order]

    # Some serial logs can contain duplicate timestamps. Average duplicates so np.interp is valid.
    unique_times, inverse = np.unique(times, return_inverse=True)
    if len(unique_times) != len(times):
        sums = np.zeros((len(unique_times), values.shape[1]), dtype=np.float64)
        counts = np.zeros(len(unique_times), dtype=np.int64)
        np.add.at(sums, inverse, values)
        np.add.at(counts, inverse, 1)
        values = (sums / counts[:, None]).astype(np.float32)
        times = unique_times

    times_s = (times - times[0]) / 1_000_000.0
    duration_s = float(times_s[-1])
    step = 1.0 / float(target_hz)
    new_times = np.arange(0.0, duration_s + (step * 0.25), step, dtype=np.float64)
    if len(new_times) == 0:
        new_times = np.array([0.0], dtype=np.float64)
    new_times = new_times[:max_length]

    resampled = np.empty((len(new_times), values.shape[1]), dtype=np.float32)
    for channel in range(values.shape[1]):
        resampled[:, channel] = np.interp(new_times, times_s, values[:, channel]).astype(
            np.float32
        )

    padded = np.zeros((max_length, values.shape[1]), dtype=np.float32)
    mask = np.zeros(max_length, dtype=bool)
    padded[: len(resampled)] = resampled
    mask[: len(resampled)] = True
    return padded, mask, duration_s


def build_trial_records(
    df: pd.DataFrame,
    *,
    region: Region = "mouth",
    target_hz: float = 50.0,
    max_seconds: float = 3.2,
    smoothing_window: int = 3,
) -> list[TrialRecord]:
    if target_hz <= 0:
        raise ValueError("target_hz must be positive.")
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive.")
    if smoothing_window < 1 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be an odd positive integer.")

    max_length = int(np.ceil(target_hz * max_seconds))
    records: list[TrialRecord] = []
    phase_to_use = 1 if region == "mouth" else 0

    grouped = df.groupby(["session_id", "trial_id"], sort=False)
    for (session_id, trial_id), trial in grouped:
        baseline_rows = trial.loc[trial["phase"] == 0]
        region_rows = trial.loc[trial["phase"] == phase_to_use]
        if baseline_rows.empty or region_rows.empty:
            print(
                f"Skipping session={session_id}, trial={trial_id}: "
                f"missing phase 0 or phase {phase_to_use}."
            )
            continue

        labels = trial["label"].dropna().unique()
        names = trial["target_name"].dropna().unique()
        if len(labels) != 1 or len(names) != 1:
            raise ValueError(
                f"Trial session={session_id}, trial={trial_id} has inconsistent labels/names."
            )

        baseline = baseline_rows[SENSOR_COLUMNS].to_numpy(dtype=np.float32).mean(axis=0)
        values = region_rows[SENSOR_COLUMNS].to_numpy(dtype=np.float32) - baseline[None, :]

        if smoothing_window > 1:
            values = uniform_filter1d(
                values,
                size=smoothing_window,
                axis=0,
                mode="nearest",
            ).astype(np.float32)

        sequence, mask, duration_s = _resample_by_time(
            values,
            region_rows["time_us"].to_numpy(),
            target_hz=target_hz,
            max_length=max_length,
        )

        records.append(
            TrialRecord(
                key=TrialKey(int(session_id), int(trial_id)),
                label_original=int(labels[0]),
                target_name=str(names[0]),
                target_type=str(trial["target_type"].iloc[0]),
                sequence=sequence,
                mask=mask,
                duration_s=duration_s,
            )
        )

    if not records:
        raise ValueError("No valid trials were constructed.")
    return records


@dataclass
class SequenceNormalizer:
    mean: np.ndarray  # [12]
    std: np.ndarray  # [12]

    @classmethod
    def fit(cls, records: Sequence[TrialRecord], indices: Sequence[int]) -> "SequenceNormalizer":
        if not indices:
            raise ValueError("Cannot fit normalizer on an empty training split.")
        total = np.zeros(len(SENSOR_COLUMNS), dtype=np.float64)
        total_sq = np.zeros(len(SENSOR_COLUMNS), dtype=np.float64)
        count = 0
        for index in indices:
            record = records[index]
            valid = record.sequence[record.mask].astype(np.float64)
            total += valid.sum(axis=0)
            total_sq += np.square(valid).sum(axis=0)
            count += len(valid)
        if count == 0:
            raise ValueError("Training split has no valid samples.")
        mean = total / count
        variance = np.maximum((total_sq / count) - np.square(mean), 1e-8)
        std = np.sqrt(variance)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, sequence: np.ndarray, mask: np.ndarray) -> np.ndarray:
        output = np.zeros_like(sequence, dtype=np.float32)
        output[mask] = (sequence[mask] - self.mean[None, :]) / self.std[None, :]
        return output

    def save(self, path: str | Path) -> None:
        np.savez(Path(path), mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str | Path) -> "SequenceNormalizer":
        data = np.load(Path(path))
        return cls(mean=data["mean"].astype(np.float32), std=data["std"].astype(np.float32))
