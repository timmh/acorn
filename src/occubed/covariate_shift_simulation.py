"""Paired simulation study of covariate shift in continuous-score occupancy models.

The study compares the original joint continuous-score model of Rhinehart et al.
(``occu_cs``) with ACORN's cut/decoupled model (``occu_cs_calibrated``).  It is
deliberately review-free: the score mixture is estimated without verified labels.

Three data-generating conditions isolate two questions:

``no_shift``
    Occupancy covariates follow the reference distribution, X ~ Normal(0, 1),
    and the class-conditional score model is correctly specified.
``occupancy_covariate_shift``
    X ~ Normal(delta, 1), but the ecological regression and score emissions are
    otherwise unchanged.  This is ordinary covariate shift: p(X) changes while
    p(z | X), p(f | z, W), and p(s | f) remain invariant.
``occupancy_and_score_shift``
    The same shift in X is accompanied by covariate-dependent score locations,
    so the fitted models' shared p(s | f) assumption is misspecified.

Simulation replicates are paired across conditions by reusing the same underlying
standard-Normal covariates and random uniforms.  This reduces Monte Carlo noise in
both condition contrasts and model comparisons.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from occubed.cs_models import occu_cs_init_strategy
from occubed.fit_utils import disable_numpyro_mcmc_caching
from occubed.models import occu_cs, occu_cs_calibrated
from occubed.paths import ARTIFACTS_DIR
from occubed.review_policies import fit_cut_normal_score_calibration

DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "covariate_shift_simulation"
MODEL_NAMES = ("coupled", "decoupled")

disable_numpyro_mcmc_caching()


@dataclass(frozen=True)
class Scenario:
    """One covariate-shift data-generating condition."""

    name: str
    occupancy_covariate_mean: float
    negative_score_covariate_effect: float = 0.0
    positive_score_covariate_effect: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class SimulationConfig:
    """Fixed parameters shared by all paired simulation conditions."""

    n_sites: int = 100
    n_replicates: int = 10
    occupancy_intercept: float = -0.5
    occupancy_slope: float = 0.9
    detection_intercept: float = -0.4
    detection_slope: float = 0.5
    score_mu0: float = 0.0
    score_sigma0: float = 1.0
    score_mu1: float = 2.5
    score_sigma1: float = 1.0
    occupancy_shift: float = 1.5
    score_covariate_effect: float = 0.75


def build_scenarios(config: SimulationConfig) -> tuple[Scenario, ...]:
    """Return the pre-registered conditions for the two experiments."""

    return (
        Scenario(
            name="no_shift",
            occupancy_covariate_mean=0.0,
            description=(
                "Reference X distribution; homogeneous class-conditional scores."
            ),
        ),
        Scenario(
            name="occupancy_covariate_shift",
            occupancy_covariate_mean=config.occupancy_shift,
            description=(
                "Shifted X distribution with invariant ecological and score models."
            ),
        ),
        Scenario(
            name="occupancy_and_score_shift",
            occupancy_covariate_mean=config.occupancy_shift,
            negative_score_covariate_effect=config.score_covariate_effect,
            positive_score_covariate_effect=config.score_covariate_effect,
            description=(
                "Shifted X distribution and X-dependent score locations for both "
                "classes; pointwise class separation remains unchanged."
            ),
        ),
    )


def _sigmoid(values):
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))


def simulate_dataset(
    config: SimulationConfig,
    scenario: Scenario,
    replicate: int,
) -> tuple[dict, dict]:
    """Simulate one paired data set in the shapes expected by the fitted models."""

    if config.n_sites < 2:
        raise ValueError("n_sites must be at least 2.")
    if config.n_replicates < 1:
        raise ValueError("n_replicates must be positive.")

    # The same seed generates the same base X, W, and random uniforms in every
    # condition. Transforming X and inverse-CDF sampling preserve pairing even
    # though occupancy and detections can change when their probabilities change.
    rng = np.random.default_rng(int(replicate))
    x_base = rng.normal(size=config.n_sites)
    x = x_base + scenario.occupancy_covariate_mean
    w = rng.normal(size=(config.n_sites, config.n_replicates))
    occupancy_uniform = rng.random(config.n_sites)
    detection_uniform = rng.random((config.n_sites, config.n_replicates))
    score_noise = rng.normal(size=(config.n_sites, config.n_replicates))

    psi = _sigmoid(config.occupancy_intercept + config.occupancy_slope * x)
    z = (occupancy_uniform < psi).astype(int)
    p = _sigmoid(config.detection_intercept + config.detection_slope * w)
    f = (detection_uniform < z[:, None] * p).astype(int)

    negative_location = (
        config.score_mu0 + scenario.negative_score_covariate_effect * x[:, None]
    )
    positive_location = (
        config.score_mu1 + scenario.positive_score_covariate_effect * x[:, None]
    )
    score_location = np.where(f == 1, positive_location, negative_location)
    score_scale = np.where(
        f == 1,
        config.score_sigma1,
        config.score_sigma0,
    )
    obs = score_location + score_scale * score_noise

    data = {
        "site_covs": x[:, None],
        "obs_covs": w[:, None, :, None],
        "obs": obs[None, :, None, :],
    }
    truth = {
        "x_base": x_base,
        "x": x,
        "w": w,
        "psi": psi,
        "z": z,
        "p": p,
        "f": f,
        "score_location": score_location,
        "occupancy_prevalence": float(z.mean()),
        "detection_prevalence": float(f.mean()),
    }
    return data, truth


def _sample_vector(samples, name):
    values = np.asarray(samples[name], dtype=float)
    return values.reshape(values.shape[0], -1)[:, 0]


def _parameter_metrics(samples, config: SimulationConfig) -> dict[str, float]:
    truths = {
        "occupancy_intercept": ("cov_state_0", config.occupancy_intercept),
        "occupancy_slope": ("cov_state_1", config.occupancy_slope),
        "detection_intercept": ("cov_det_0", config.detection_intercept),
        "detection_slope": ("cov_det_1", config.detection_slope),
    }
    metrics: dict[str, float] = {}
    for label, (sample_name, truth) in truths.items():
        values = _sample_vector(samples, sample_name)
        lower, upper = np.quantile(values, [0.05, 0.95])
        posterior_mean = float(values.mean())
        metrics[f"{label}_true"] = float(truth)
        metrics[f"{label}_mean"] = posterior_mean
        metrics[f"{label}_bias"] = posterior_mean - float(truth)
        metrics[f"{label}_abs_error"] = abs(posterior_mean - float(truth))
        metrics[f"{label}_q05"] = float(lower)
        metrics[f"{label}_q95"] = float(upper)
        metrics[f"{label}_covered_90"] = float(lower <= truth <= upper)
    return metrics


def _psi_metrics(samples, truth, config: SimulationConfig) -> dict[str, float]:
    posterior_psi = np.asarray(samples["psi"], dtype=float)
    posterior_psi_mean = posterior_psi.mean(axis=0).reshape(-1)
    true_psi = np.asarray(truth["psi"], dtype=float).reshape(-1)
    if posterior_psi_mean.size != true_psi.size:
        raise ValueError(
            "Posterior psi size does not match simulated sites: "
            f"{posterior_psi_mean.size} != {true_psi.size}."
        )
    site_errors = posterior_psi_mean - true_psi

    reference_x = np.linspace(-3.0, 3.0, 121)
    beta0 = _sample_vector(samples, "cov_state_0")
    beta1 = _sample_vector(samples, "cov_state_1")
    posterior_reference = _sigmoid(
        beta0[:, None] + beta1[:, None] * reference_x[None, :]
    ).mean(axis=0)
    true_reference = _sigmoid(
        config.occupancy_intercept + config.occupancy_slope * reference_x
    )
    reference_errors = posterior_reference - true_reference
    return {
        "psi_site_rmse": float(np.sqrt(np.mean(np.square(site_errors)))),
        "psi_site_mae": float(np.mean(np.abs(site_errors))),
        "psi_site_mean_bias": float(site_errors.mean()),
        "psi_reference_rmse": float(np.sqrt(np.mean(np.square(reference_errors)))),
        "psi_reference_mae": float(np.mean(np.abs(reference_errors))),
    }


def _mcmc_metrics(result) -> dict[str, float]:
    extra_fields = result.mcmc.get_extra_fields()
    diverging = np.asarray(extra_fields.get("diverging", []), dtype=float)
    grouped_samples = result.mcmc.get_samples(group_by_chain=True)
    max_r_hat = float("nan")
    min_ess = float("nan")
    if (
        grouped_samples
        and np.asarray(next(iter(grouped_samples.values()))).shape[0] > 1
    ):
        from numpyro.diagnostics import effective_sample_size, gelman_rubin

        diagnostic_names = {
            "alpha",
            "beta",
            "mu0_std",
            "mu_gap_std",
            "sigma0_std",
            "sigma1_std",
        }
        r_hats = []
        effective_sizes = []
        for name, values in grouped_samples.items():
            if name not in diagnostic_names:
                continue
            values = np.asarray(values, dtype=float)
            r_hats.extend(np.asarray(gelman_rubin(values)).reshape(-1))
            effective_sizes.extend(
                np.asarray(effective_sample_size(values)).reshape(-1)
            )
        finite_r_hats = np.asarray(r_hats)[np.isfinite(r_hats)]
        finite_effective_sizes = np.asarray(effective_sizes)[
            np.isfinite(effective_sizes)
        ]
        if finite_r_hats.size:
            max_r_hat = float(finite_r_hats.max())
        if finite_effective_sizes.size:
            min_ess = float(finite_effective_sizes.min())
    return {
        "divergence_fraction": (
            float(diverging.mean()) if diverging.size else float("nan")
        ),
        "max_r_hat": max_r_hat,
        "min_effective_sample_size": min_ess,
        "num_posterior_draws": int(
            np.asarray(next(iter(result.samples.values()))).shape[0]
        ),
    }


def _score_parameter_metrics(samples, calibration, config, scenario):
    if calibration is None:
        estimates = {
            name: float(_sample_vector(samples, name).mean())
            for name in ("mu0", "sigma0", "mu1", "sigma1")
        }
    else:
        estimates = {
            name: float(getattr(calibration, name))
            for name in ("mu0", "sigma0", "mu1", "sigma1")
        }
    # Under score shift there is no single true component location. Reporting the
    # location at mean X makes this departure from the fitted model explicit.
    target_mu0 = (
        config.score_mu0
        + scenario.negative_score_covariate_effect * scenario.occupancy_covariate_mean
    )
    target_mu1 = (
        config.score_mu1
        + scenario.positive_score_covariate_effect * scenario.occupancy_covariate_mean
    )
    return {
        **{f"score_{name}_mean": value for name, value in estimates.items()},
        "score_mu0_at_mean_x_true": float(target_mu0),
        "score_mu1_at_mean_x_true": float(target_mu1),
        "score_mu0_at_mean_x_bias": float(estimates["mu0"] - target_mu0),
        "score_mu1_at_mean_x_bias": float(estimates["mu1"] - target_mu1),
    }


def fit_one(
    config: SimulationConfig,
    scenario: Scenario,
    replicate: int,
    model: str,
    *,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 2,
    score_num_warmup: int = 500,
    score_num_samples: int = 500,
    score_prevalence_mean: float = 0.15,
    score_prevalence_strength: float = 2.0,
) -> dict[str, float | int | str]:
    """Simulate and fit one scenario/replicate/model task."""

    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODEL_NAMES}.")
    data, truth = simulate_dataset(config, scenario, replicate)
    mcmc_seed = 100_000 + int(replicate)
    calibration = None
    calibration_seconds = 0.0

    if model == "decoupled":
        verified_f = np.full_like(data["obs"], np.nan, dtype=float)
        calibration_start = time.perf_counter()
        calibration, score_log_bf, _score_log_bf_draws = (
            fit_cut_normal_score_calibration(
                data["obs"],
                verified_f,
                {
                    "score_calibration_num_warmup": score_num_warmup,
                    "score_calibration_num_samples": score_num_samples,
                    "score_calibration_num_chains": num_chains,
                    "score_calibration_num_draws": min(32, score_num_samples),
                    "score_calibration_prevalence_mean": float(score_prevalence_mean),
                    "score_calibration_prevalence_strength": (
                        score_prevalence_strength
                    ),
                    "score_calibration_random_seed": mcmc_seed,
                },
            )
        )
        calibration_seconds = time.perf_counter() - calibration_start
        from biolith.utils import fit

        fit_start = time.perf_counter()
        result = fit(
            occu_cs_calibrated,
            site_covs=data["site_covs"],
            obs_covs=data["obs_covs"],
            score_log_bf=score_log_bf,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            random_seed=mcmc_seed,
        )
    else:
        from biolith.utils import fit

        fit_start = time.perf_counter()
        result = fit(
            occu_cs,
            site_covs=data["site_covs"],
            obs_covs=data["obs_covs"],
            obs=data["obs"],
            init_strategy=occu_cs_init_strategy(data["obs"]),
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            random_seed=mcmc_seed,
        )
    fit_seconds = time.perf_counter() - fit_start

    return {
        "scenario": scenario.name,
        "replicate": int(replicate),
        "model": model,
        "n_sites": int(config.n_sites),
        "n_replicates": int(config.n_replicates),
        "occupancy_covariate_mean": float(scenario.occupancy_covariate_mean),
        "negative_score_covariate_effect": float(
            scenario.negative_score_covariate_effect
        ),
        "positive_score_covariate_effect": float(
            scenario.positive_score_covariate_effect
        ),
        "realized_occupancy": float(truth["occupancy_prevalence"]),
        "realized_detection_prevalence": float(truth["detection_prevalence"]),
        "fit_wall_time_seconds": float(fit_seconds),
        "score_calibration_wall_time_seconds": float(calibration_seconds),
        **_parameter_metrics(result.samples, config),
        **_psi_metrics(result.samples, truth, config),
        **_score_parameter_metrics(result.samples, calibration, config, scenario),
        **_mcmc_metrics(result),
    }


def _write_csv(path: Path, rows) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def result_path(output_dir, scenario, replicate, model) -> Path:
    return (
        Path(output_dir)
        / "runs"
        / scenario
        / f"replicate_{int(replicate):03d}"
        / f"{model}.csv"
    )


def run_one(
    output_dir=DEFAULT_OUTPUT_DIR,
    scenario="no_shift",
    replicate=0,
    model="coupled",
    n_sites=100,
    n_replicates=10,
    occupancy_shift=1.5,
    score_covariate_effect=0.75,
    num_warmup=500,
    num_samples=500,
    num_chains=2,
    score_num_warmup=500,
    score_num_samples=500,
    score_prevalence_mean=0.15,
    score_prevalence_strength=2.0,
    resume=True,
):
    """Run one resumable task and write its single-row artifact."""

    config = SimulationConfig(
        n_sites=int(n_sites),
        n_replicates=int(n_replicates),
        occupancy_shift=float(occupancy_shift),
        score_covariate_effect=float(score_covariate_effect),
    )
    scenario_lookup = {item.name: item for item in build_scenarios(config)}
    if scenario not in scenario_lookup:
        raise ValueError(
            f"Unknown scenario {scenario!r}; expected {tuple(scenario_lookup)}."
        )
    path = result_path(output_dir, scenario, replicate, model)
    if bool(resume) and path.exists():
        return path
    row = fit_one(
        config,
        scenario_lookup[scenario],
        int(replicate),
        model,
        num_warmup=int(num_warmup),
        num_samples=int(num_samples),
        num_chains=int(num_chains),
        score_num_warmup=int(score_num_warmup),
        score_num_samples=int(score_num_samples),
        score_prevalence_mean=float(score_prevalence_mean),
        score_prevalence_strength=float(score_prevalence_strength),
    )
    _write_csv(path, [row])
    return path


def _all_result_rows(output_dir) -> pd.DataFrame:
    paths = sorted(Path(output_dir).glob("runs/*/replicate_*/*.csv"))
    if not paths:
        raise FileNotFoundError(f"No run artifacts found under {output_dir}.")
    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


ERROR_METRICS = (
    "psi_site_rmse",
    "psi_reference_rmse",
    "occupancy_intercept_abs_error",
    "occupancy_slope_abs_error",
    "detection_intercept_abs_error",
    "detection_slope_abs_error",
)
MAX_ACCEPTABLE_R_HAT = 1.05
MAX_ACCEPTABLE_DIVERGENCE_FRACTION = 0.01


def diagnostically_acceptable_rows(rows: pd.DataFrame) -> pd.Series:
    """Flag fits suitable for convergence-conditioned recovery summaries."""

    r_hat = pd.to_numeric(rows["max_r_hat"], errors="coerce")
    divergence = pd.to_numeric(rows["divergence_fraction"], errors="coerce")
    # A one-chain smoke run has no R-hat; retain it based on divergences alone.
    r_hat_ok = r_hat.isna() | (r_hat <= MAX_ACCEPTABLE_R_HAT)
    return r_hat_ok & (divergence <= MAX_ACCEPTABLE_DIVERGENCE_FRACTION)


def summarize_results(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build model summaries and paired decoupled-minus-coupled contrasts."""

    records = []
    for (scenario, model), group in rows.groupby(["scenario", "model"]):
        acceptable = diagnostically_acceptable_rows(group)
        record = {
            "scenario": scenario,
            "model": model,
            "n_replicates": int(group["replicate"].nunique()),
            "mean_realized_occupancy": float(group["realized_occupancy"].mean()),
            "mean_divergence_fraction": float(group["divergence_fraction"].mean()),
            "max_r_hat": float(group["max_r_hat"].max()),
            "min_effective_sample_size": float(
                group["min_effective_sample_size"].min()
            ),
            "n_diagnostically_acceptable": int(acceptable.sum()),
            "diagnostically_acceptable_fraction": float(acceptable.mean()),
            "mean_fit_wall_time_seconds": float(group["fit_wall_time_seconds"].mean()),
        }
        for metric in ERROR_METRICS:
            values = group[metric].to_numpy(dtype=float)
            record[f"mean_{metric}"] = float(np.mean(values))
            record[f"se_{metric}"] = (
                float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else float("nan")
            )
        for parameter in (
            "occupancy_intercept",
            "occupancy_slope",
            "detection_intercept",
            "detection_slope",
        ):
            record[f"coverage_90_{parameter}"] = float(
                group[f"{parameter}_covered_90"].mean()
            )
            record[f"mean_bias_{parameter}"] = float(group[f"{parameter}_bias"].mean())
        records.append(record)
    summary = pd.DataFrame.from_records(records).sort_values(["scenario", "model"])

    paired = rows.pivot_table(
        index=["scenario", "replicate"],
        columns="model",
        values=list(ERROR_METRICS),
        aggfunc="first",
    )
    contrast_records = []
    if set(MODEL_NAMES).issubset(paired.columns.get_level_values(1)):
        for scenario in rows["scenario"].unique():
            scenario_pair = paired.loc[scenario]
            for metric in ERROR_METRICS:
                difference = (
                    scenario_pair[(metric, "decoupled")]
                    - scenario_pair[(metric, "coupled")]
                ).dropna()
                n = len(difference)
                mean = float(difference.mean()) if n else float("nan")
                se = (
                    float(difference.std(ddof=1) / math.sqrt(n))
                    if n > 1
                    else float("nan")
                )
                contrast_records.append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "n_pairs": n,
                        "mean_decoupled_minus_coupled": mean,
                        "se": se,
                        "ci95_lower": mean - 1.96 * se,
                        "ci95_upper": mean + 1.96 * se,
                        "fraction_pairs_decoupled_better": (
                            float(np.mean(difference < 0.0)) if n else float("nan")
                        ),
                    }
                )
    contrasts = pd.DataFrame.from_records(contrast_records)
    return summary, contrasts


def summarize_shift_effects(rows: pd.DataFrame) -> pd.DataFrame:
    """Compute paired shifted-minus-reference recovery changes for each model."""

    reference = "no_shift"
    shifted_scenarios = (
        "occupancy_covariate_shift",
        "occupancy_and_score_shift",
    )
    records = []
    for model in MODEL_NAMES:
        model_rows = rows.loc[rows["model"] == model]
        wide = model_rows.pivot_table(
            index="replicate",
            columns="scenario",
            values=list(ERROR_METRICS),
            aggfunc="first",
        )
        for scenario in shifted_scenarios:
            if reference not in wide.columns.get_level_values(
                1
            ) or scenario not in wide.columns.get_level_values(1):
                continue
            for metric in ERROR_METRICS:
                difference = (
                    wide[(metric, scenario)] - wide[(metric, reference)]
                ).dropna()
                n = len(difference)
                mean = float(difference.mean()) if n else float("nan")
                se = (
                    float(difference.std(ddof=1) / math.sqrt(n))
                    if n > 1
                    else float("nan")
                )
                records.append(
                    {
                        "model": model,
                        "shifted_scenario": scenario,
                        "metric": metric,
                        "n_pairs": n,
                        "mean_shifted_minus_no_shift": mean,
                        "se": se,
                        "ci95_lower": mean - 1.96 * se,
                        "ci95_upper": mean + 1.96 * se,
                        "fraction_pairs_shift_increased_error": (
                            float(np.mean(difference > 0.0)) if n else float("nan")
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)




def plot_results(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot the two primary recovery metrics with Monte Carlo error bars."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - CSV output remains usable.
        return

    all_scenarios = [
        ("no_shift", "No shift"),
        ("occupancy_covariate_shift", "Occupancy X shift"),
        ("occupancy_and_score_shift", "X + score shift"),
    ]
    present_scenarios = set(summary["scenario"])
    scenario_order = [
        item for item, _label in all_scenarios if item in present_scenarios
    ]
    scenario_labels = [
        label for item, label in all_scenarios if item in present_scenarios
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("psi_site_rmse", "occupancy_slope_abs_error"),
        ("Occupancy-probability recovery", "Occupancy-slope recovery"),
        strict=True,
    ):
        for offset, model in zip((-0.07, 0.07), MODEL_NAMES, strict=True):
            indexed = summary.loc[summary["model"] == model].set_index("scenario")
            if not set(scenario_order).issubset(indexed.index):
                continue
            values = np.asarray(
                [indexed.loc[item, f"mean_{metric}"] for item in scenario_order]
            )
            errors = np.asarray(
                [indexed.loc[item, f"se_{metric}"] for item in scenario_order]
            )
            axis.errorbar(
                np.arange(len(scenario_order)) + offset,
                values,
                yerr=1.96 * errors,
                marker="o",
                capsize=3,
                label=model.capitalize(),
            )
        axis.set_xticks(np.arange(len(scenario_order)), scenario_labels, rotation=15)
        axis.set_ylabel("RMSE" if "rmse" in metric else "Absolute error")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(output_dir / "covariate_shift_recovery.png", dpi=200)
    fig.savefig(output_dir / "covariate_shift_recovery.pdf")
    plt.close(fig)


def analyze(output_dir=DEFAULT_OUTPUT_DIR):
    """Aggregate completed tasks into numerical summaries and figures."""

    output_dir = Path(output_dir)
    rows = _all_result_rows(output_dir)
    summary, contrasts = summarize_results(rows)
    shift_effects = summarize_shift_effects(rows)
    converged_rows = rows.loc[diagnostically_acceptable_rows(rows)].copy()
    converged_summary, converged_contrasts = summarize_results(converged_rows)
    converged_shift_effects = summarize_shift_effects(converged_rows)
    rows.to_csv(output_dir / "simulation_results.csv", index=False)
    summary.to_csv(output_dir / "recovery_summary.csv", index=False)
    contrasts.to_csv(output_dir / "paired_model_contrasts.csv", index=False)
    shift_effects.to_csv(output_dir / "paired_shift_effects.csv", index=False)
    converged_summary.to_csv(output_dir / "recovery_summary_converged.csv", index=False)
    converged_contrasts.to_csv(
        output_dir / "paired_model_contrasts_converged.csv", index=False
    )
    converged_shift_effects.to_csv(
        output_dir / "paired_shift_effects_converged.csv", index=False
    )
    plot_results(converged_summary, output_dir)
    return output_dir


def run(
    output_dir=DEFAULT_OUTPUT_DIR,
    n_simulation_replicates=30,
    n_sites=100,
    n_replicates=10,
    occupancy_shift=1.5,
    score_covariate_effect=0.75,
    num_warmup=500,
    num_samples=500,
    num_chains=2,
    score_num_warmup=500,
    score_num_samples=500,
    score_prevalence_mean=0.15,
    score_prevalence_strength=2.0,
    resume=True,
):
    """Run the full study locally, writing each fit as soon as it completes."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig(
        n_sites=int(n_sites),
        n_replicates=int(n_replicates),
        occupancy_shift=float(occupancy_shift),
        score_covariate_effect=float(score_covariate_effect),
    )
    manifest = {
        **asdict(config),
        "n_simulation_replicates": int(n_simulation_replicates),
        "num_warmup": int(num_warmup),
        "num_samples": int(num_samples),
        "num_chains": int(num_chains),
        "score_num_warmup": int(score_num_warmup),
        "score_num_samples": int(score_num_samples),
        "score_prevalence_mean": float(score_prevalence_mean),
        "score_prevalence_strength": float(score_prevalence_strength),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "scenarios": [asdict(item) for item in build_scenarios(config)],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    total_tasks = len(build_scenarios(config)) * int(n_simulation_replicates) * 2
    completed = 0
    for scenario in build_scenarios(config):
        for replicate in range(int(n_simulation_replicates)):
            for model in MODEL_NAMES:
                path = result_path(output_dir, scenario.name, replicate, model)
                was_complete = path.exists()
                run_one(
                    output_dir=output_dir,
                    scenario=scenario.name,
                    replicate=replicate,
                    model=model,
                    n_sites=n_sites,
                    n_replicates=n_replicates,
                    occupancy_shift=occupancy_shift,
                    score_covariate_effect=score_covariate_effect,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    score_num_warmup=score_num_warmup,
                    score_num_samples=score_num_samples,
                    score_prevalence_mean=score_prevalence_mean,
                    score_prevalence_strength=score_prevalence_strength,
                    resume=resume,
                )
                completed += 1
                status = "cached" if bool(resume) and was_complete else "fit"
                print(
                    f"[{completed}/{total_tasks}] {scenario.name} replicate "
                    f"{replicate} {model}: {status}",
                    flush=True,
                )
    return analyze(output_dir)


def main(action="run", **kwargs):
    """CLI entry point used by ``scripts/06_covariate_shift_simulation.py``."""

    actions = {"run": run, "run_one": run_one, "analyze": analyze}
    if action not in actions:
        raise ValueError(
            f"Unknown action {action!r}; expected one of {tuple(actions)}."
        )
    return actions[action](**kwargs)
