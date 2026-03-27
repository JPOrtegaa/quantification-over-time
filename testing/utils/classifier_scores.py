"""
Precompute HuggingFace classifier softmax scores per time window (no interpolation).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Union

import numpy as np
import pandas as pd

import config
from classifier.HuggingFaceModel import SentiAnalyzer


def _sentiment_class_from_label(lab) -> Union[int, None]:
    """Map common HF sentiment label strings to dataset codes {-1, 0, 1}."""
    s = str(lab).lower().strip()
    if s in ("-1", "0", "1"):
        return int(s)
    if "positive" in s or s in ("pos", "1", "label_2"):
        return 1
    if "negative" in s or s in ("neg", "-1", "label_0"):
        return -1
    if "neutral" in s or s in ("neu", "label_1"):
        return 0
    return None


def _output_columns_for_classes(model_config, classes: Sequence) -> List[int]:
    """
    Map each dataset class value to the corresponding model output index (softmax dimension).
    """
    id2label = model_config.id2label
    n_out = len(id2label)
    cols: List[int] = []
    for c in classes:
        idx = None
        for j in range(n_out):
            lab = id2label[j]
            if lab == c or str(lab) == str(c):
                idx = j
                break
            try:
                if int(lab) == int(c):
                    idx = j
                    break
            except (TypeError, ValueError):
                continue
        if idx is None:
            c_int = int(c) if isinstance(c, (int, np.integer)) else None
            for j in range(n_out):
                mapped = _sentiment_class_from_label(id2label[j])
                if mapped is not None and c_int is not None and mapped == c_int:
                    idx = j
                    break
        if idx is None:
            raise ValueError(
                f"Cannot map dataset class {c!r} to model labels {dict(id2label)}"
            )
        cols.append(idx)
    return cols


def _align_softmax(batch_scores: np.ndarray, col_indices: Sequence[int]) -> np.ndarray:
    """Reorder softmax columns to match the order of ``classes``."""
    return batch_scores[:, list(col_indices)]


def compute_window_classifier_scores(
    data_dict: Mapping[int, pd.DataFrame],
    classifier_model_id: str,
    classes: Sequence,
    *,
    text_column: str = "text",
    batch_size: Union[int, None] = None,
    show_progress: bool = True,
) -> Dict[int, np.ndarray]:
    """
    Run the classifier on every row of every time window and return aligned softmax scores.

    Windows are processed in increasing key order (temporal batches as produced by the loader).
    No interpolation: each window is scored independently on its own rows.

    Parameters
    ----------
    data_dict
        Mapping from window index to a DataFrame with a text column.
    classifier_model_id
        HuggingFace model id for ``SentiAnalyzer``.
    classes
        Ordered label values for the task (used to permute model outputs to this order).
    text_column
        Name of the text column in each chunk DataFrame.

    Returns
    -------
    dict
        ``window_index -> array of shape (n_rows_in_window, len(classes))``.
    """
    analyzer = SentiAnalyzer(classifier_model_id)
    col_ix = _output_columns_for_classes(analyzer.config, classes)
    if batch_size is None:
        batch_size = getattr(config, "HF_INFERENCE_BATCH_SIZE", 8)

    out: Dict[int, np.ndarray] = {}
    ordered_keys = sorted(data_dict.keys())

    for w in ordered_keys:
        df = data_dict[w]
        if text_column not in df.columns:
            raise KeyError(
                f"Window {w}: missing column {text_column!r}; columns={list(df.columns)}"
            )
        texts = df[text_column].astype(str).tolist()
        raw, _ = analyzer.batch_analyze(
            texts,
            batch_size=batch_size,
            show_progress=show_progress,
            hf_context=f"window {w}",
            announce=True,
        )
        out[w] = _align_softmax(np.asarray(raw, dtype=np.float64), col_ix)

    return out
