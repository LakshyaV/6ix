from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from silent_speech.data import build_label_mapping, build_trial_records, load_dataframe
from silent_speech.features import engineered_features
from silent_speech.model import DualBranchTCN


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python test_pipeline.py /path/to/data.csv")
    path = Path(sys.argv[1])
    df = load_dataframe(path, remove_first_stop=True)
    mapping, index_to_name = build_label_mapping(df)
    records = build_trial_records(df)
    assert len(records) > 0
    assert records[0].sequence.shape == (160, 12)
    assert records[0].mask.dtype == np.bool_
    features = engineered_features(records[0].sequence, records[0].mask, sensor_mode="both")
    assert np.isfinite(features).all()
    model = DualBranchTCN(len(index_to_name), sensor_mode="both")
    batch = torch.from_numpy(np.stack([records[0].sequence, records[1].sequence]))
    mask = torch.from_numpy(np.stack([records[0].mask, records[1].mask]))
    logits = model(batch, mask)
    assert logits.shape == (2, len(index_to_name))
    print(f"Smoke test passed. Trials={len(records)}, classes={len(mapping)}, params={model.parameter_count():,}")


if __name__ == "__main__":
    main()
