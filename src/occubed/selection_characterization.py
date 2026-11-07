"""Compact, pre-decision acquisition diagnostics and valid-pool covariate features."""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def selection_pool_arrays(selection_state, diagnostics=None, used_bootstrap=False):
    mask = np.asarray(selection_state["reviewable"], dtype=bool)
    ids = np.flatnonzero(mask)
    result = dict(candidate_id=ids, reviewable_shape=np.asarray(mask.shape),
                  used_max_score_bootstrap=np.asarray(bool(used_bootstrap)))
    fields = dict(score=selection_state.get("max_score"),
                  p_positive=selection_state.get("mean_prob_positive"),
                  bald_score=selection_state.get("expected_gain_score"),
                  ppu_score=selection_state.get("posterior_predictive_uncertainty"),
                  label_entropy=selection_state.get("posterior_predictive_uncertainty"))
    for key in ("p_positive", "target_eig", "target_eig_raw", "label_entropy", "ess_positive", "ess_negative"):
        if diagnostics is not None and key in diagnostics:
            fields[key] = diagnostics[key]
        elif key not in fields:
            fields[key] = None
    for key, value in fields.items():
        if value is None:
            result[key] = np.full(ids.size, np.nan)
        else:
            value = np.asarray(value, dtype=float)
            if value.shape != mask.shape:
                raise ValueError(f"Candidate diagnostic {key} has incompatible shape {value.shape}.")
            result[key] = value.reshape(-1)[ids]
    return result


def read_candidate_diagnostics(step_dir):
    path = Path(step_dir) / "selection_pool.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as arrays:
        ids = arrays["candidate_id"]
        fields = {key: arrays[key] for key in arrays.files
                  if key not in {"candidate_id", "reviewable_shape", "used_max_score_bootstrap"}}
        return {int(candidate_id): {key: float(value[index]) for key, value in fields.items()}
                for index, candidate_id in enumerate(ids)}


def masked_leverage(matrix, valid):
    matrix, valid = np.asarray(matrix, dtype=float), np.asarray(valid, dtype=bool)
    result = np.full(matrix.shape[0], np.nan)
    valid = valid & np.isfinite(matrix).all(axis=1)
    if matrix.shape[1] == 0 or valid.sum() < 2:
        return result
    values = matrix[valid]
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / (len(values) - 1)
    inverse = np.linalg.pinv(covariance, rcond=1e-8)
    result[valid] = np.einsum("ni,ij,nj->n", centered, inverse, centered)
    return result


def covariate_novelty(matrix, reference_mask):
    matrix = np.asarray(matrix, dtype=float)
    valid = np.isfinite(matrix).all(axis=1)
    reference = valid & np.asarray(reference_mask, dtype=bool)
    result = np.full(matrix.shape[0], np.nan)
    if matrix.shape[1] and reference.any():
        result[valid] = cKDTree(matrix[reference]).query(matrix[valid])[0] / np.sqrt(matrix.shape[1])
    return result


def discovery_credits(selected, labels, reviewed):
    selected, labels, reviewed = np.asarray(selected, bool), np.asarray(labels), np.asarray(reviewed)
    axes = tuple(range(1, selected.ndim))
    unconfirmed = (np.isfinite(reviewed) & (reviewed > 0.5)).sum(axis=axes) == 0
    positive = selected & (labels > 0.5)
    count = positive.sum(axis=axes)
    weight = np.divide(unconfirmed.astype(float), count, out=np.zeros(len(count)), where=count > 0)
    return positive * weight.reshape(-1, *([1] * len(axes)))
