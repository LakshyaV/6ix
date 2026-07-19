from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import SequenceNormalizer, TrialRecord
from .features import engineered_features
from .model import DualBranchTCN


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    val: list[int]
    test: list[int]


@dataclass
class Metrics:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms can be much slower; seeded trial-level experiments are sufficient here.


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Metrics:
    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


def save_evaluation(
    output_dir: Path,
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    index_to_name: dict[int, str],
    prefix: str,
) -> Metrics:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(index_to_name)
    target_names = [index_to_name[index] for index in labels]
    metrics = compute_metrics(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    with (output_dir / f"{prefix}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": metrics.as_dict(), "classification_report": report}, handle, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(
        output_dir / f"{prefix}_confusion_matrix.csv"
    )

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    figure_size = max(8.0, len(labels) * 0.72)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    image = ax.imshow(cm_normalized, vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(target_names)), target_names, rotation=45, ha="right")
    ax.set_yticks(range(len(target_names)), target_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{prefix}: normalized confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=180)
    plt.close(fig)
    return metrics


def remap_labels(
    records: Sequence[TrialRecord],
    original_to_index: dict[int, int],
) -> np.ndarray:
    return np.asarray([original_to_index[r.label_original] for r in records], dtype=np.int64)


def train_svm(
    records: Sequence[TrialRecord],
    labels: np.ndarray,
    split: SplitIndices,
    *,
    sensor_mode: str,
    output_dir: Path,
    index_to_name: dict[int, str],
    randomize_labels: bool,
    seed: int,
) -> Metrics:
    x = np.stack(
        [
            engineered_features(r.sequence, r.mask, sensor_mode=sensor_mode)  # type: ignore[arg-type]
            for r in records
        ]
    )
    y_train = labels[split.train].copy()
    if randomize_labels:
        rng = np.random.default_rng(seed)
        rng.shuffle(y_train)

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
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(x[split.train], y_train)
    predictions = pipeline.predict(x[split.test])
    metrics = save_evaluation(
        output_dir,
        y_true=labels[split.test],
        y_pred=predictions,
        index_to_name=index_to_name,
        prefix="svm_test",
    )
    joblib.dump(pipeline, output_dir / "svm.joblib")
    return metrics


class IMUTrialDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        records: Sequence[TrialRecord],
        labels: np.ndarray,
        indices: Sequence[int],
        normalizer: SequenceNormalizer,
        *,
        augment: bool,
        seed: int,
    ) -> None:
        self.records = records
        self.labels = labels
        self.indices = list(indices)
        self.normalizer = normalizer
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def _augment(self, sequence: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid_length = int(mask.sum())
        valid = sequence[:valid_length].copy()
        max_length = len(mask)

        # Shared time stretch for all 12 channels keeps both IMUs synchronized.
        factor = float(self.rng.uniform(0.90, 1.10))
        stretched_length = int(np.clip(round(valid_length * factor), 2, max_length))
        old_t = np.linspace(0.0, 1.0, valid_length, dtype=np.float32)
        new_t = np.linspace(0.0, 1.0, stretched_length, dtype=np.float32)
        stretched = np.empty((stretched_length, valid.shape[1]), dtype=np.float32)
        for channel in range(valid.shape[1]):
            stretched[:, channel] = np.interp(new_t, old_t, valid[:, channel]).astype(np.float32)

        # Small synchronized temporal shift inside the padded canvas.
        max_shift = min(5, max_length - stretched_length)
        shift = int(self.rng.integers(0, max_shift + 1)) if max_shift > 0 else 0
        output = np.zeros_like(sequence, dtype=np.float32)
        output_mask = np.zeros_like(mask, dtype=bool)
        output[shift : shift + stretched_length] = stretched
        output_mask[shift : shift + stretched_length] = True

        # Gentle amplitude scaling, independently for jaw and reference branches.
        output[:, :6] *= float(self.rng.uniform(0.94, 1.06))
        output[:, 6:] *= float(self.rng.uniform(0.94, 1.06))

        # Low noise after normalization; only valid samples receive noise.
        noise_std = float(self.rng.uniform(0.0, 0.025))
        if noise_std > 0:
            output[output_mask] += self.rng.normal(
                0.0, noise_std, size=output[output_mask].shape
            ).astype(np.float32)

        # Occasional single-channel dropout improves tolerance to a noisy axis.
        if self.rng.random() < 0.08:
            channel = int(self.rng.integers(0, output.shape[1]))
            output[:, channel] = 0.0

        return output, output_mask

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[self.indices[item]]
        sequence = self.normalizer.transform(record.sequence, record.mask)
        mask = record.mask.copy()
        if self.augment:
            sequence, mask = self._augment(sequence, mask)
        label = int(self.labels[self.indices[item]])
        return (
            torch.from_numpy(sequence),
            torch.from_numpy(mask),
            torch.tensor(label, dtype=torch.long),
        )


def _predict_tcn(
    model: DualBranchTCN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    true_labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for sequence, mask, labels in loader:
            sequence = sequence.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            logits = model(sequence, mask)
            probs = torch.softmax(logits, dim=1)
            true_labels.append(labels.numpy())
            predictions.append(probs.argmax(dim=1).cpu().numpy())
            probabilities.append(probs.cpu().numpy())
    return (
        np.concatenate(true_labels),
        np.concatenate(predictions),
        np.concatenate(probabilities),
    )


def train_tcn(
    records: Sequence[TrialRecord],
    labels: np.ndarray,
    split: SplitIndices,
    *,
    sensor_mode: str,
    output_dir: Path,
    index_to_name: dict[int, str],
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    randomize_labels: bool,
    seed: int,
    device: torch.device,
) -> Metrics:
    normalizer = SequenceNormalizer.fit(records, split.train)
    normalizer.save(output_dir / "normalizer.npz")

    train_labels = labels.copy()
    validation_labels = labels.copy()
    if randomize_labels:
        rng = np.random.default_rng(seed)
        shuffled_train = train_labels[split.train].copy()
        shuffled_val = validation_labels[split.val].copy()
        rng.shuffle(shuffled_train)
        rng.shuffle(shuffled_val)
        train_labels[split.train] = shuffled_train
        validation_labels[split.val] = shuffled_val

    train_dataset = IMUTrialDataset(
        records,
        train_labels,
        split.train,
        normalizer,
        augment=True,
        seed=seed,
    )
    val_dataset = IMUTrialDataset(
        records,
        validation_labels,
        split.val,
        normalizer,
        augment=False,
        seed=seed + 1,
    )
    test_dataset = IMUTrialDataset(
        records,
        labels,
        split.test,
        normalizer,
        augment=False,
        seed=seed + 2,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = DualBranchTCN(
        num_classes=len(index_to_name),
        sensor_mode=sensor_mode,  # type: ignore[arg-type]
    ).to(device)
    print(f"TCN parameters ({sensor_mode}): {model.parameter_count():,}")

    counts = np.bincount(train_labels[split.train], minlength=len(index_to_name)).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_balanced_accuracy = -np.inf
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        examples = 0
        for sequence, mask, target in train_loader:
            sequence = sequence.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(sequence, mask)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += float(loss.item()) * len(target)
            examples += len(target)

        train_loss = running_loss / max(examples, 1)

        model.eval()
        val_loss_sum = 0.0
        val_examples = 0
        val_true: list[np.ndarray] = []
        val_pred: list[np.ndarray] = []
        with torch.no_grad():
            for sequence, mask, target in val_loader:
                sequence = sequence.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                target_device = target.to(device, non_blocking=True)
                logits = model(sequence, mask)
                loss = criterion(logits, target_device)
                val_loss_sum += float(loss.item()) * len(target)
                val_examples += len(target)
                val_true.append(target.numpy())
                val_pred.append(logits.argmax(dim=1).cpu().numpy())

        val_loss = val_loss_sum / max(val_examples, 1)
        val_true_array = np.concatenate(val_true)
        val_pred_array = np.concatenate(val_pred)
        val_balanced = float(balanced_accuracy_score(val_true_array, val_pred_array))
        scheduler.step(val_loss)
        learning_rate_now = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_balanced_accuracy": val_balanced,
                "learning_rate": learning_rate_now,
            }
        )
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_bal_acc={val_balanced:.4f} | "
            f"lr={learning_rate_now:.2e}"
        )

        if val_balanced > best_balanced_accuracy + 1e-6:
            best_balanced_accuracy = val_balanced
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping after epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)

    checkpoint = {
        "model_state": model.state_dict(),
        "num_classes": len(index_to_name),
        "sensor_mode": sensor_mode,
        "hidden_channels": model.hidden_channels,
        "index_to_name": index_to_name,
        "parameter_count": model.parameter_count(),
    }
    torch.save(checkpoint, output_dir / "best_tcn.pt")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    y_true, y_pred, probabilities = _predict_tcn(model, test_loader, device)
    np.save(output_dir / "test_probabilities.npy", probabilities)
    return save_evaluation(
        output_dir,
        y_true=y_true,
        y_pred=y_pred,
        index_to_name=index_to_name,
        prefix="tcn_test",
    )
