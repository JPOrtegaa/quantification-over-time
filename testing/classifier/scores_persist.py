"""
Save classifier softmax scores under ``classifier/outputs`` (see ``config.CLASSIFIER_OUTPUT_DIR``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _json_friendly_classes(classes: Sequence) -> List:
    out = []
    for x in classes:
        if isinstance(x, np.generic):
            out.append(x.item())
        else:
            out.append(x)
    return out


def batch_split_window_keys(
    data_dict: Mapping[int, Any],
    train_set_size: int,
    test_set_size: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """
    Split temporal batch indices: first ``train_set_size`` windows for training, then test windows.

    Test windows are the remaining timeline order, optionally capped at ``test_set_size`` batches.
    """
    ordered = sorted(data_dict.keys())
    n = len(ordered)
    if train_set_size < 0:
        raise ValueError(f"train_set_size must be >= 0, got {train_set_size}")
    if train_set_size > n:
        raise ValueError(
            f"train_set_size={train_set_size} exceeds number of batches ({n})"
        )
    train_keys = ordered[:train_set_size]
    remaining = ordered[train_set_size:]
    if test_set_size is None:
        test_keys = remaining
    else:
        if test_set_size < 0:
            raise ValueError(f"test_set_size must be >= 0 or None, got {test_set_size}")
        test_keys = remaining[:test_set_size]
    return train_keys, test_keys


def _safe_path_fragment(name: str, max_len: int = 120) -> str:
    s = name.replace("/", "_").replace(":", "_").replace(" ", "_")
    return s[:max_len] if len(s) > max_len else s


def save_split_classifier_scores(
    output_root: Path,
    dataset_name: str,
    classifier_model_id: str,
    classes: Sequence,
    scores_train: Dict[int, np.ndarray],
    scores_test: Dict[int, np.ndarray],
    train_keys: Sequence[int],
    test_keys: Sequence[int],
    train_set_size: int,
    test_set_size: Optional[int],
    random_state: Optional[int] = None,
) -> Path:
    """
    Write ``scores.npz`` (per-window arrays) and ``meta.json`` under
    ``output_root / <dataset> / <classifier_fragment> /``.
    """
    output_root = Path(output_root)
    out_dir = output_root / _safe_path_fragment(str(dataset_name)) / _safe_path_fragment(
        classifier_model_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_payload: Dict[str, np.ndarray] = {}
    for k in train_keys:
        npz_payload[f"train_w{k}"] = scores_train[k]
    for k in test_keys:
        npz_payload[f"test_w{k}"] = scores_test[k]
    npz_path = out_dir / "scores.npz"
    np.savez_compressed(npz_path, **npz_payload)

    meta = {
        "dataset": dataset_name,
        "classifier_model_id": classifier_model_id,
        "classes": _json_friendly_classes(classes),
        "train_window_keys": list(train_keys),
        "test_window_keys": list(test_keys),
        "train_set_size": train_set_size,
        "test_set_size": test_set_size,
        "random_state": random_state,
        "scores_file": "scores.npz",
        "note": (
            "HF classifier is frozen; train windows only define the portion used for "
            "downstream training (e.g. quantifiers). Scores are still computed by inference."
        ),
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return out_dir
