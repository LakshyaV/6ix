#!/usr/bin/env python3
"""Live dual-IMU jaw-gesture inference for Arduino UNO Q.

STM32 side:
    reads the built-in reference IMU and external 0x6A jaw IMU.
Qualcomm Linux side:
    obtains buffered samples over RouterBridge and runs the saved joblib models.
"""

from __future__ import annotations

import math
import os
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from arduino.app_utils import App, Bridge

from jaw_gesture_ml import (
    LOGICAL_NAMES,
    _build_sequence,
    _event_window,
    _feature_vector,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

COMMAND_NAMES = {
    "shift_right": "START",
    "shift_left": "STOP",
    "push_forward": "MAKE_APP",
    "pull_backward": "WEATHER",
}

# These match the working Mac live-test settings.
BASELINE_SECONDS = 1.2
START_Z = 3.0
END_Z = 1.6
START_FRAMES = 2
QUIET_FRAMES = 7
MIN_EVENT_SECONDS = 0.25
MAX_EVENT_SECONDS = 2.5
PRE_EVENT_SECONDS = 0.40
LISTEN_TIMEOUT = 8.0
COOLDOWN_SECONDS = 0.8
FALLBACK_RATE_HZ = 35.36

# The working diagnostic configuration flipped the wake model's two labels.
INVERT_WAKE_LABELS = os.getenv("INVERT_WAKE_LABELS", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DEBUG = os.getenv("JAW_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}


def log(message: str = "") -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class LiveSample:
    host_time: float
    device_time_us: int
    raw: np.ndarray  # jaw accel/gyro, then reference accel/gyro; g and deg/s


@dataclass
class EventDetector:
    start_z: float = START_Z
    end_z: float = END_Z
    start_frames: int = START_FRAMES
    quiet_frames: int = QUIET_FRAMES
    min_event_s: float = MIN_EVENT_SECONDS
    max_event_s: float = MAX_EVENT_SECONDS

    active: bool = False
    high_count: int = 0
    quiet_count: int = 0
    started_at: float = 0.0
    event_samples: list[LiveSample] | None = None

    def reset(self) -> None:
        self.active = False
        self.high_count = 0
        self.quiet_count = 0
        self.started_at = 0.0
        self.event_samples = None


class DeviceClock:
    """Maps the STM32 32-bit micros() clock onto a continuous host timeline."""

    WRAP = 1 << 32

    def __init__(self) -> None:
        self._last_raw: int | None = None
        self._wrap_offset = 0
        self._anchor_device: int | None = None
        self._anchor_host: float | None = None

    def host_time(self, raw_time_us: int) -> float:
        raw = int(raw_time_us) & 0xFFFFFFFF
        if self._last_raw is not None and raw < self._last_raw:
            if self._last_raw - raw > (self.WRAP // 2):
                self._wrap_offset += self.WRAP
        self._last_raw = raw
        unwrapped = self._wrap_offset + raw

        if self._anchor_device is None:
            self._anchor_device = unwrapped
            self._anchor_host = time.monotonic()

        assert self._anchor_host is not None
        return self._anchor_host + (unwrapped - self._anchor_device) / 1_000_000.0


def load_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model: {path}\n"
            "Copy wake_model.joblib and command_model.joblib into python/models/."
        )
    artifact = joblib.load(path)
    required = {"model", "classes", "feature_mode", "target_samples", "window_seconds"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"{path.name} is missing fields: {sorted(missing)}")
    if not hasattr(artifact["model"], "predict_proba"):
        raise TypeError(f"Model in {path.name} does not support predict_proba()")
    return artifact


def normalize_bridge_payload(payload: Any) -> str:
    # Some RPC versions may wrap a single return value in a one-element sequence.
    while isinstance(payload, (list, tuple)) and len(payload) == 1:
        payload = payload[0]
    if payload is None:
        return ""
    if isinstance(payload, (bytes, bytearray)):
        return payload.decode("utf-8", errors="strict")
    return str(payload)


def parse_batch(payload: Any, clock: DeviceClock) -> list[LiveSample]:
    """Parse rows returned by get_imu_batch().

    Schema per row:
      seq,time_us,
      jaw_ax_mg,jaw_ay_mg,jaw_az_mg,
      jaw_gx_cdeg_s,jaw_gy_cdeg_s,jaw_gz_cdeg_s,
      ref_ax_mg,ref_ay_mg,ref_az_mg,
      ref_gx_cdeg_s,ref_gy_cdeg_s,ref_gz_cdeg_s,
      ready_mask
    """
    text = normalize_bridge_payload(payload).strip()
    if not text:
        return []

    samples: list[LiveSample] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 15:
            if DEBUG:
                log(f"[debug] skipped malformed bridge row ({len(parts)} fields): {line[:120]}")
            continue
        try:
            device_time_us = int(parts[1])
            ready_mask = int(parts[14])
            if ready_mask != 15:
                continue

            values = np.asarray(
                [
                    float(parts[2]) / 1000.0,
                    float(parts[3]) / 1000.0,
                    float(parts[4]) / 1000.0,
                    float(parts[5]) / 100.0,
                    float(parts[6]) / 100.0,
                    float(parts[7]) / 100.0,
                    float(parts[8]) / 1000.0,
                    float(parts[9]) / 1000.0,
                    float(parts[10]) / 1000.0,
                    float(parts[11]) / 100.0,
                    float(parts[12]) / 100.0,
                    float(parts[13]) / 100.0,
                ],
                dtype=np.float64,
            )
            samples.append(
                LiveSample(
                    host_time=clock.host_time(device_time_us),
                    device_time_us=device_time_us,
                    raw=values,
                )
            )
        except (TypeError, ValueError, OverflowError):
            if DEBUG:
                log(f"[debug] failed to parse bridge row: {line[:120]}")
    return samples


def robust_scale(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0) * 1.4826
    floor = np.asarray([0.008, 0.008, 0.008, 0.8, 0.8, 0.8] * 2, dtype=float)
    return np.maximum(mad, floor)


def motion_score(sample: LiveSample, baseline: np.ndarray) -> float:
    center = np.median(baseline, axis=0)
    centered = sample.raw - center

    relative = centered[0:6] - centered[6:12]
    relative_baseline = (
        (baseline[:, 0:6] - center[0:6])
        - (baseline[:, 6:12] - center[6:12])
    )
    rel_scale = robust_scale(
        np.concatenate([relative_baseline, relative_baseline], axis=1)
    )[:6]
    z = relative / rel_scale
    return float(np.sqrt(np.mean(z * z)))


def observed_rate(samples: list[LiveSample], fallback: float = FALLBACK_RATE_HZ) -> float:
    if len(samples) < 3:
        return fallback
    timestamps = np.asarray([sample.host_time for sample in samples], dtype=float)
    dt = np.diff(timestamps)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if not len(dt):
        return fallback
    return float(1.0 / np.median(dt))


def event_feature(
    event_samples: list[LiveSample],
    baseline: np.ndarray,
    artifact: dict[str, Any],
) -> tuple[np.ndarray, float]:
    raw = np.vstack([sample.raw for sample in event_samples])
    center = np.median(baseline, axis=0)
    centered_event = raw - center
    centered_baseline = baseline - center

    sequence = _build_sequence(centered_event[:, 0:6], centered_event[:, 6:12])
    baseline_sequence = _build_sequence(
        centered_baseline[:, 0:6], centered_baseline[:, 6:12]
    )

    rate = observed_rate(event_samples)
    event = _event_window(
        sequence,
        baseline_relative=baseline_sequence[:, 12:18],
        observed_rate_hz=rate,
        window_seconds=float(artifact["window_seconds"]),
        target_samples=int(artifact["target_samples"]),
    )

    mode = str(artifact["feature_mode"])
    if mode == "jaw":
        selected = event[:, 0:6]
        names = LOGICAL_NAMES[0:6]
    elif mode == "relative":
        selected = event[:, 12:18]
        names = LOGICAL_NAMES[12:18]
    elif mode == "all":
        selected = event
        names = LOGICAL_NAMES
    else:
        raise ValueError(f"Unknown feature mode in artifact: {mode}")

    feature, feature_names = _feature_vector(selected, names)
    expected = artifact.get("feature_names")
    if expected is not None and list(feature_names) != list(expected):
        raise RuntimeError(
            "Live feature names differ from the trained model. Do not modify "
            "python/jaw_gesture_ml.py without retraining."
        )
    return feature.reshape(1, -1), rate


def predict(
    artifact: dict[str, Any],
    feature: np.ndarray,
) -> tuple[str, float, list[tuple[str, float]]]:
    model = artifact["model"]
    probabilities = np.asarray(model.predict_proba(feature)[0], dtype=float)
    classes = np.asarray(model.classes_).astype(str)
    order = np.argsort(probabilities)[::-1]
    ranked = [(str(classes[index]), float(probabilities[index])) for index in order]
    return ranked[0][0], ranked[0][1], ranked


def format_ranked(ranked: list[tuple[str, float]]) -> str:
    return " | ".join(f"{label}={probability:.2f}" for label, probability in ranked)


def handle_command(command: str, source_label: str, confidence: float) -> None:
    """Current routing hook. Add camera-context actions here later."""
    log(f"\n>>> COMMAND: {command} [{source_label}] ({confidence:.1%}) <<<\n")


class JawGestureDecoder:
    def __init__(self) -> None:
        self.wake_artifact = load_artifact(MODEL_DIR / "wake_model.joblib")
        self.command_artifact = load_artifact(MODEL_DIR / "command_model.joblib")

        for field in ("feature_mode", "target_samples", "window_seconds"):
            if self.wake_artifact[field] != self.command_artifact[field]:
                raise ValueError(f"Wake and command artifacts disagree on {field}")

        self.wake_threshold = float(
            self.wake_artifact.get("confidence_threshold", 0.70)
        )
        self.command_threshold = float(
            self.command_artifact.get("confidence_threshold", 0.60)
        )

        baseline_capacity = max(20, int(math.ceil(BASELINE_SECONDS * 60)))
        pre_capacity = max(5, int(math.ceil(PRE_EVENT_SECONDS * 60)))
        self.baseline_buffer: deque[LiveSample] = deque(maxlen=baseline_capacity)
        self.pre_buffer: deque[LiveSample] = deque(maxlen=pre_capacity)
        self.detector = EventDetector()
        self.clock = DeviceClock()

        self.state = "WAITING_FOR_CLENCH"
        self.armed_at = 0.0
        self.cooldown_until = 0.0
        self.last_debug = 0.0
        self.last_bridge_error = 0.0
        self.bridge_ready = False

        log(
            "Loaded models: "
            f"mode={self.wake_artifact['feature_mode']}, "
            f"window={self.wake_artifact['window_seconds']}s, "
            f"wake_threshold={self.wake_threshold:.2f}, "
            f"command_threshold={self.command_threshold:.2f}"
        )
        log(f"Wake-label inversion: {'ON' if INVERT_WAKE_LABELS else 'OFF'}")
        log("Flow: CLENCH once, return neutral, then perform one command gesture.\n")

    def wait_for_bridge(self) -> None:
        while True:
            try:
                status = normalize_bridge_payload(Bridge.call("get_imu_status"))
                log(f"STM32 bridge ready: {status}")
                self.bridge_ready = True
                return
            except Exception as exc:
                log(f"Waiting for STM32 bridge route: {exc}")
                time.sleep(1.0)

    def process_sample(self, sample: LiveSample) -> None:
        now = sample.host_time

        if self.state == "LISTENING_FOR_COMMAND" and now - self.armed_at > LISTEN_TIMEOUT:
            log("Listening timed out. Clench again.")
            self.state = "WAITING_FOR_CLENCH"
            self.detector.reset()

        if now < self.cooldown_until:
            self.baseline_buffer.append(sample)
            self.pre_buffer.append(sample)
            return

        minimum_baseline = max(15, int(BASELINE_SECONDS * 25))
        if len(self.baseline_buffer) < minimum_baseline:
            self.baseline_buffer.append(sample)
            self.pre_buffer.append(sample)
            if len(self.baseline_buffer) % 15 == 0:
                log(
                    f"Learning neutral baseline... "
                    f"{len(self.baseline_buffer)}/{minimum_baseline}"
                )
            return

        baseline_raw = np.vstack([item.raw for item in self.baseline_buffer])
        score = motion_score(sample, baseline_raw)

        if DEBUG and now - self.last_debug >= 0.25:
            log(f"[debug] state={self.state} motion_z={score:.2f}")
            self.last_debug = now

        if not self.detector.active:
            self.pre_buffer.append(sample)
            if score >= self.detector.start_z:
                self.detector.high_count += 1
            else:
                self.detector.high_count = 0
                self.baseline_buffer.append(sample)

            if self.detector.high_count >= self.detector.start_frames:
                self.detector.active = True
                self.detector.started_at = now
                self.detector.quiet_count = 0
                self.detector.event_samples = list(self.pre_buffer)
                # Preserve the same segmentation behavior as the working Mac script.
                self.detector.event_samples.append(sample)
                if DEBUG:
                    log(f"[debug] event started at motion_z={score:.2f}")
            return

        assert self.detector.event_samples is not None
        self.detector.event_samples.append(sample)
        elapsed = now - self.detector.started_at

        if score <= self.detector.end_z:
            self.detector.quiet_count += 1
        else:
            self.detector.quiet_count = 0

        finished = (
            elapsed >= self.detector.min_event_s
            and self.detector.quiet_count >= self.detector.quiet_frames
        ) or elapsed >= self.detector.max_event_s

        if not finished:
            return

        event_samples = self.detector.event_samples
        self.detector.reset()

        try:
            artifact = (
                self.wake_artifact
                if self.state == "WAITING_FOR_CLENCH"
                else self.command_artifact
            )
            feature, rate = event_feature(event_samples, baseline_raw, artifact)
            label, confidence, ranked = predict(artifact, feature)
        except Exception as exc:
            log(f"Event preprocessing failed: {exc}")
            self.baseline_buffer.clear()
            self.pre_buffer.clear()
            return

        log(
            f"Detected event ({len(event_samples)} samples, {rate:.1f} Hz): "
            f"{format_ranked(ranked)}"
        )

        if self.state == "WAITING_FOR_CLENCH":
            probabilities = dict(ranked)
            if INVERT_WAKE_LABELS:
                clench_probability = probabilities.get("not_clench", 0.0)
                wake_label = "clench" if label == "not_clench" else "not_clench"
            else:
                clench_probability = probabilities.get("clench", 0.0)
                wake_label = label

            if wake_label == "clench" and clench_probability >= self.wake_threshold:
                log("\n>>> CLENCH DETECTED — LISTENING FOR COMMAND <<<\n")
                self.state = "LISTENING_FOR_COMMAND"
                self.armed_at = now
                self.cooldown_until = now + COOLDOWN_SECONDS
            else:
                log("Not accepted as wake clench.")
        else:
            if confidence >= self.command_threshold and label in COMMAND_NAMES:
                handle_command(COMMAND_NAMES[label], label, confidence)
                self.state = "WAITING_FOR_CLENCH"
                self.cooldown_until = now + COOLDOWN_SECONDS
            else:
                log(
                    f"Low-confidence command ({label}, {confidence:.1%}). "
                    "Still listening; try the command again."
                )
                self.armed_at = now

        self.baseline_buffer.clear()
        self.pre_buffer.clear()

    def run_forever(self) -> None:
        self.wait_for_bridge()
        while True:
            try:
                payload = Bridge.call("get_imu_batch")
                for sample in parse_batch(payload, self.clock):
                    self.process_sample(sample)
            except Exception as exc:
                now = time.monotonic()
                if now - self.last_bridge_error >= 1.0:
                    log(f"Bridge read error: {exc}")
                    self.last_bridge_error = now
                self.bridge_ready = False
                time.sleep(0.10)
            else:
                time.sleep(0.008)


def worker() -> None:
    try:
        JawGestureDecoder().run_forever()
    except Exception:
        log("\nJaw gesture worker stopped with an error:")
        traceback.print_exc()
        while True:
            time.sleep(60)


log("Starting 6ix jaw-gesture app on Qualcomm Linux...")
threading.Thread(target=worker, name="jaw-gesture-worker", daemon=True).start()
App.run()
