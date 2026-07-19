from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from silent_speech.data import SequenceNormalizer, build_trial_records, load_dataframe
from silent_speech.features import engineered_features
from silent_speech.model import DualBranchTCN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict commands for trial-formatted IMU CSV data.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--artifacts", required=True, help="Path to one experiment's svm/ or tcn/ folder.")
    parser.add_argument("--model", choices=["svm", "tcn"], required=True)
    parser.add_argument("--sensor-mode", choices=["jaw", "ref", "both"], default="both")
    parser.add_argument("--region", choices=["mouth", "rest"], default="mouth")
    parser.add_argument("--target-hz", type=float, default=50.0)
    parser.add_argument("--max-seconds", type=float, default=3.2)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifacts).expanduser().resolve()
    df = load_dataframe(args.csv)
    records = build_trial_records(
        df,
        region=args.region,
        target_hz=args.target_hz,
        max_seconds=args.max_seconds,
        smoothing_window=args.smoothing_window,
    )

    rows = []
    if args.model == "svm":
        pipeline = joblib.load(artifact_dir / "svm.joblib")
        x = np.stack(
            [
                engineered_features(r.sequence, r.mask, sensor_mode=args.sensor_mode)
                for r in records
            ]
        )
        predictions = pipeline.predict(x)
        # Label names are stored at run root, two levels above model folder.
        mapping_path = artifact_dir.parents[1] / "label_mapping.json"
        with mapping_path.open("r", encoding="utf-8") as handle:
            mapping = json.load(handle)
        index_to_name = {int(k): v for k, v in mapping["index_to_name"].items()}
        for record, prediction in zip(records, predictions):
            rows.append(
                {
                    "session_id": record.key.session_id,
                    "trial_id": record.key.trial_id,
                    "predicted_index": int(prediction),
                    "predicted_name": index_to_name[int(prediction)],
                }
            )
    else:
        device = torch.device(args.device)
        checkpoint = torch.load(artifact_dir / "best_tcn.pt", map_location=device, weights_only=False)
        model = DualBranchTCN(
            checkpoint["num_classes"],
            sensor_mode=checkpoint["sensor_mode"],
            hidden_channels=checkpoint["hidden_channels"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        normalizer = SequenceNormalizer.load(artifact_dir / "normalizer.npz")
        index_to_name = {int(k): v for k, v in checkpoint["index_to_name"].items()}

        with torch.no_grad():
            for record in records:
                sequence = normalizer.transform(record.sequence, record.mask)
                sequence_tensor = torch.from_numpy(sequence)[None].to(device)
                mask_tensor = torch.from_numpy(record.mask)[None].to(device)
                probabilities = torch.softmax(model(sequence_tensor, mask_tensor), dim=1)[0]
                prediction = int(probabilities.argmax().item())
                rows.append(
                    {
                        "session_id": record.key.session_id,
                        "trial_id": record.key.trial_id,
                        "predicted_index": prediction,
                        "predicted_name": index_to_name[prediction],
                        "confidence": float(probabilities[prediction].item()),
                    }
                )

    output = pd.DataFrame(rows)
    print(output.to_string(index=False))
    output.to_csv("predictions.csv", index=False)
    print("\nSaved predictions.csv")


if __name__ == "__main__":
    main()
