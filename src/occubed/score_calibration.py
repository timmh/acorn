from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScoreMixtureCalibration:
    """Externally fitted score-distribution parameters."""

    mu0: float
    sigma0: float
    mu1: float
    sigma1: float
    n_negative: int
    n_positive: int


def fit_score_mixture(
    obs,
    verified_f,
    *,
    min_class_size: int = 2,
    min_sigma: float = 1e-6,
) -> ScoreMixtureCalibration:
    """Fit class-conditional Normal score distributions from verified labels.

    Parameters
    ----------
    obs:
        Continuous score observations.
    verified_f:
        Binary labels matching ``obs``. Finite values are treated as verified
        labels, and ``NaN`` entries are ignored.
    min_class_size:
        Minimum number of verified examples required in each class.
    min_sigma:
        Lower bound applied to the fitted standard deviations.

    Returns
    -------
    ScoreMixtureCalibration
        Fitted class-conditional Normal parameters on the original score scale.
    """

    obs_arr = np.asarray(obs, dtype=float)
    verified_arr = np.asarray(verified_f, dtype=float)
    valid = np.isfinite(obs_arr) & np.isfinite(verified_arr)
    if not np.any(valid):
        raise ValueError(
            "No overlapping finite entries found in `obs` and `verified_f`."
        )

    labels = verified_arr[valid] > 0.5
    scores = obs_arr[valid]
    negative_scores = scores[~labels]
    positive_scores = scores[labels]
    if negative_scores.size < min_class_size or positive_scores.size < min_class_size:
        raise ValueError(
            "External score calibration requires verified positives and negatives. "
            f"Got {negative_scores.size} negatives and {positive_scores.size} positives."
        )

    sigma0 = float(np.maximum(negative_scores.std(ddof=1), min_sigma))
    sigma1 = float(np.maximum(positive_scores.std(ddof=1), min_sigma))
    return ScoreMixtureCalibration(
        mu0=float(negative_scores.mean()),
        sigma0=sigma0,
        mu1=float(positive_scores.mean()),
        sigma1=sigma1,
        n_negative=int(negative_scores.size),
        n_positive=int(positive_scores.size),
    )


def fit_score_mixture_with_shrinkage(
    obs,
    verified_f,
    base_calibration: ScoreMixtureCalibration,
    *,
    prior_strength: float = 10.0,
    min_sigma: float = 1e-6,
    min_mean_separation: float = 1e-3,
) -> ScoreMixtureCalibration:
    """Update a score calibration from reviewed labels with shrinkage to a baseline.

    The reviewed labels provide direct supervision for the score calibration, but
    refitting the class-conditional score distributions from only a small review
    batch can be unstable. This helper instead treats ``base_calibration`` as a
    stable prior fit and blends in the reviewed class-conditional score moments
    as additional evidence.

    Parameters
    ----------
    obs:
        Continuous score observations.
    verified_f:
        Binary reviewed labels matching ``obs``. Finite values are treated as
        verified labels, and ``NaN`` entries are ignored.
    base_calibration:
        Stable external calibration used as the shrinkage target.
    prior_strength:
        Pseudo-count assigned to the baseline calibration for each class.
    min_sigma:
        Lower bound applied to the adapted standard deviations.
    min_mean_separation:
        Lower bound enforced on ``mu1 - mu0`` to avoid label-swapping
        degeneracies when the review set is tiny.

    Returns
    -------
    ScoreMixtureCalibration
        Adapted class-conditional Normal parameters on the original score scale.
    """

    if prior_strength < 0:
        raise ValueError("`prior_strength` must be non-negative.")

    obs_arr = np.asarray(obs, dtype=float)
    verified_arr = np.asarray(verified_f, dtype=float)
    valid = np.isfinite(obs_arr) & np.isfinite(verified_arr)
    scores = obs_arr[valid]
    labels = verified_arr[valid] > 0.5
    negative_scores = scores[~labels]
    positive_scores = scores[labels]

    def _blend_class_scores(class_scores, base_mean, base_sigma):
        if class_scores.size == 0:
            return float(base_mean), float(np.maximum(base_sigma, min_sigma))

        total_weight = prior_strength + class_scores.size
        prior_second_moment = base_mean**2 + base_sigma**2
        blended_mean = (prior_strength * base_mean + class_scores.sum()) / total_weight
        blended_second_moment = (
            prior_strength * prior_second_moment + np.square(class_scores).sum()
        ) / total_weight
        blended_variance = np.maximum(
            blended_second_moment - blended_mean**2,
            min_sigma**2,
        )
        return float(blended_mean), float(np.sqrt(blended_variance))

    mu0, sigma0 = _blend_class_scores(
        negative_scores, base_calibration.mu0, base_calibration.sigma0
    )
    mu1, sigma1 = _blend_class_scores(
        positive_scores, base_calibration.mu1, base_calibration.sigma1
    )
    if mu1 <= mu0:
        midpoint = 0.5 * (mu0 + mu1)
        mu0 = midpoint - 0.5 * min_mean_separation
        mu1 = midpoint + 0.5 * min_mean_separation

    return ScoreMixtureCalibration(
        mu0=mu0,
        sigma0=sigma0,
        mu1=mu1,
        sigma1=sigma1,
        n_negative=int(negative_scores.size),
        n_positive=int(positive_scores.size),
    )


def score_log_bayes_factor(obs, calibration: ScoreMixtureCalibration):
    """Compute log p(score | f=1) - log p(score | f=0) from external calibration."""

    obs_arr = np.asarray(obs, dtype=float)
    valid = np.isfinite(obs_arr)
    log_bf = np.full(obs_arr.shape, np.nan, dtype=float)
    centered0 = (obs_arr[valid] - calibration.mu0) / calibration.sigma0
    centered1 = (obs_arr[valid] - calibration.mu1) / calibration.sigma1
    log_bf[valid] = (
        -0.5 * centered1**2
        - np.log(calibration.sigma1)
        + 0.5 * centered0**2
        + np.log(calibration.sigma0)
    )
    return log_bf
