# Export CSV Column Meanings

This file documents the meanings of the CSV files exported by `export_sweep.py`.

### Metadata

- `sweep_name`: Sweep directory name (includes date and time).
- `species_index`: Zero-based species order in the sweep.
- `dataset_kind`: Dataset source for the row.
- `species`: Target species name.
- `target_label`: Short target label used in the data.
- `experiment_id`: Convenience copy of the experiment identifier.
- `step_index`: Step number.

### Sweep Metadata

- `sweep__dataset_kind`: Dataset selection recorded for the sweep.
- `sweep__dataset_count`: Number of datasets included in the sweep.
- `sweep__species_count`: Number of species requested in the sweep.
- `sweep__num_warmup`: Sweep-level MCMC warmup draws per chain.
- `sweep__num_samples`: Sweep-level MCMC posterior draws per chain.
- `sweep__num_chains`: Sweep-level MCMC chain count.
- `sweep__random_seed`: Sweep-level random seed.
- `sweep__reviews_per_step`: Sweep-level labels reviewed per step.
- `sweep__max_reviews`: Sweep-level review budget cap.
- `sweep__top_k`: Sweep-level species cap per dataset.
- `sweep__job_spec_build_wall_time_seconds`: Time spent building sweep job specs.
- `sweep__batch_submission_wall_time_seconds`: Time spent submitting or resubmitting jobs.
- `sweep__completion_wait_wall_time_seconds`: Time spent waiting for sweep jobs to finish.
- `sweep__skipped_completed_job_count`: Jobs skipped because results already existed.
- `sweep__resubmitted_job_count`: Jobs actually submitted or resubmitted.
- `sweep__sweep_wall_time_seconds`: Total sweep runtime.
- `sweep__completed_job_count`: Jobs that finished successfully.
- `sweep__failed_job_count`: Jobs that finished with failure.
- `sweep__total_job_wall_time_seconds`: Sum of per-job wall times.
- `sweep__mean_job_wall_time_seconds`: Mean per-job wall time.
- `sweep__max_job_wall_time_seconds`: Longest per-job wall time.

### Per-Species Run Metadata

- `run__dataset_kind`: Dataset source recorded in the species run.
- `run__dataset_name`: Human-readable dataset name.
- `run__target_species`: Target species recorded in the run manifest.
- `run__target_label`: Target label recorded in the run manifest.
- `run__num_experiments`: Number of experiment variants run for the species.
- `run__num_shared_parameters`: Count of parameters compared to the oracle.
- `run__num_warmup`: Species-run MCMC warmup draws per chain.
- `run__num_samples`: Species-run MCMC posterior draws per chain.
- `run__num_chains`: Species-run MCMC chain count.
- `run__random_seed`: Species-run random seed.
- `run__n_reviews_per_step`: Labels reviewed per active-learning step.
- `run__max_reviews`: Maximum labels reviewed for the species.
- `run__artifact_write_wall_time_seconds`: Time spent writing final species artifacts.
- `run__dataset_load_wall_time_seconds`: Time spent loading the dataset.
- `run__oracle_mcmc_wall_time_seconds`: Time spent fitting the oracle model.
- `run__experiments_wall_time_seconds`: Time spent running all experiment variants.
- `run__dataset_compute_wall_time_seconds`: End-to-end species runtime.
- `run__base_score_calibration_wall_time_seconds`: Time spent fitting the baseline score calibration.
- `run__base_score_log_bayes_factor_wall_time_seconds`: Time spent computing baseline score Bayes factors.
- `run__experiment_spec_build_wall_time_seconds`: Time spent building per-species experiment specs.

### Dataset Shape Metadata

- `dataset__dataset_kind`: Dataset source recorded in dataset metadata.
- `dataset__dataset_name`: Human-readable dataset name from dataset metadata.
- `dataset__target_species`: Target species recorded in dataset metadata.
- `dataset__target_label`: Target label recorded in dataset metadata.
- `dataset__n_sites`: Number of sites.
- `dataset__n_periods`: Number of periods.
- `dataset__n_replicates`: Number of replicate observations per period.
- `dataset__n_site_covariates`: Number of site covariates.
- `dataset__n_obs_covariates`: Number of observation covariates.
- `dataset__n_reviewable`: Number of reviewable labels.

### Oracle Diagnostics

- `oracle__num_samples`: Oracle MCMC posterior draws per chain.
- `oracle__num_warmup`: Oracle MCMC warmup draws per chain.
- `oracle__num_chains`: Oracle MCMC chain count.
- `oracle__posterior_draws`: Total oracle posterior draws across chains.
- `oracle__num_parameters`: Number of summarized oracle parameters.
- `oracle__mean_r_hat`: Mean oracle R-hat.
- `oracle__max_r_hat`: Max oracle R-hat.
- `oracle__min_r_hat`: Min oracle R-hat.
- `oracle__mean_n_eff`: Mean oracle effective sample size.
- `oracle__min_n_eff`: Min oracle effective sample size.
- `oracle__max_n_eff`: Max oracle effective sample size.
- `oracle__mean_frac_eff`: Mean oracle effective sample fraction.
- `oracle__min_frac_eff`: Min oracle effective sample fraction.
- `oracle__num_diverging`: Number of oracle divergent transitions.
- `oracle__frac_diverging`: Fraction of oracle divergent transitions.
- `oracle__mean_beta_sd`: Mean oracle beta posterior SD summary.
- `oracle__mean_alpha_sd`: Mean oracle alpha posterior SD summary.
- `oracle__num_oracle_parameters`: Number of parameters used in oracle-distance comparisons.
- `oracle__num_positive_labels`: Count of positive true labels.
- `oracle__num_total_labels`: Count of finite true labels.
- `oracle__oracle_mcmc_wall_time_seconds`: Time spent fitting the oracle model.

### Oracle Parameter Summaries

- `oracle__parameter_mean__<parameter_slug>`: Oracle posterior mean for a shared coefficient parameter.
- `oracle__parameter_q05__<parameter_slug>`: Oracle posterior 5th percentile for a shared coefficient parameter.
- `oracle__parameter_q95__<parameter_slug>`: Oracle posterior 95th percentile for a shared coefficient parameter.
- `<parameter_slug>` is the slugified `parameter` value from `oracle/parameter_summary.csv` for occupancy coefficients (`cov_state_*`) and detection coefficients (`cov_det_*`).

### Null-Oracle Parameter Summaries

- `null_oracle__parameter_mean__<parameter_slug>`: Null-oracle posterior mean for a shared coefficient parameter.
- `null_oracle__parameter_q05__<parameter_slug>`: Null-oracle posterior 5th percentile for a shared coefficient parameter.
- `null_oracle__parameter_q95__<parameter_slug>`: Null-oracle posterior 95th percentile for a shared coefficient parameter.

### Experiment Metadata

- `experiment__experiment_id`: Experiment identifier.
- `experiment__display_name`: Experiment name.
- `experiment__model_family`: Model family.
- `experiment__model_label`: Specific model label.
- `experiment__selection_method`: Acquisition rule.
- `experiment__n_steps`: Number of saved review steps.
- `experiment__final_review_count`: Total reviewed labels at the last step.
- `experiment__final_mean_oracle_distance`: Final mean oracle distance.
- `experiment__total_score_calibration_wall_time_seconds`: Total score-calibration time across steps.
- `experiment__total_mcmc_wall_time_seconds`: Total MCMC time across steps.
- `experiment__total_selection_wall_time_seconds`: Total selection time across steps.
- `experiment__total_step_wall_time_seconds`: Total per-step runtime across steps.
- `experiment__experiment_wall_time_seconds`: End-to-end experiment runtime.
- `experiment__covariate_variant`: Covariate setting (`full` or `null`).
- `experiment__n_site_covariates`: Site covariate count used by the experiment.
- `experiment__n_obs_covariates`: Observation covariate count used by the experiment.

### Step Trace Metrics

- `trace__step_index`: Step number.
- `trace__review_count`: Cumulative reviewed labels at the step.
- `trace__selected_count`: Labels selected at the step.
- `trace__mean_oracle_distance`: Mean oracle distance at the step.
- `trace__num_samples`: Step-level MCMC posterior draws per chain.
- `trace__num_warmup`: Step-level MCMC warmup draws per chain.
- `trace__num_chains`: Step-level MCMC chain count.
- `trace__posterior_draws`: Total step-level posterior draws across chains.
- `trace__num_parameters`: Number of summarized step-level parameters.
- `trace__mean_r_hat`: Mean step-level R-hat.
- `trace__max_r_hat`: Max step-level R-hat.
- `trace__min_r_hat`: Min step-level R-hat.
- `trace__mean_n_eff`: Mean step-level effective sample size.
- `trace__min_n_eff`: Min step-level effective sample size.
- `trace__max_n_eff`: Max step-level effective sample size.
- `trace__mean_frac_eff`: Mean step-level effective sample fraction.
- `trace__min_frac_eff`: Min step-level effective sample fraction.
- `trace__num_diverging`: Number of divergent transitions at the step.
- `trace__frac_diverging`: Fraction of divergent transitions at the step.
- `trace__mean_beta_sd`: Mean beta posterior SD summary at the step.
- `trace__mean_alpha_sd`: Mean alpha posterior SD summary at the step.
- `trace__score_calibration_wall_time_seconds`: Time spent calibrating scores at the step.
- `trace__mcmc_wall_time_seconds`: Time spent fitting the step model.
- `trace__selection_wall_time_seconds`: Time spent selecting labels at the step.
- `trace__step_wall_time_seconds`: Total runtime for the step.
- `trace__grouped_oracle_distance__overall_occupancy_psi`: Oracle distance for occupancy probabilities.
- `trace__grouped_oracle_distance__occupancy_coefficients`: Oracle distance for occupancy coefficients as a group.
- `trace__grouped_oracle_distance__detection_coefficients`: Oracle distance for detection coefficients as a group.

### Agreement Inputs

- `oracle__psi_mean_values`: Species-row only. `|`-separated oracle site-level posterior mean occupancy values.
- `trace__psi_mean_values`: Full-model step-row only. `|`-separated step-level site posterior mean occupancy values.
- Coefficient agreement should be computed downstream from the exported raw interval summaries: `oracle__parameter_q05__cov_state_*`, `oracle__parameter_q95__cov_state_*`, `oracle__parameter_q05__cov_det_*`, `oracle__parameter_q95__cov_det_*`, and matching `parameter_q05__*` / `parameter_q95__*` step columns.

### Oracle Distances

- `parameter_oracle_distance__psi`: Oracle distance for occupancy probability `psi`.
- `parameter_oracle_distance__cov_state_0`: Oracle distance for occupancy intercept.
- `parameter_oracle_distance__cov_state_x`: Oracle distance for occupancy coefficient x.
- `parameter_oracle_distance__cov_det_0`: Oracle distance for detection intercept.
- `parameter_oracle_distance__cov_det_1`: Oracle distance for detection coefficient x.

### Step Parameter Summaries

- `parameter_mean__<parameter_slug>`: Step-level posterior mean for a fitted shared coefficient parameter. Use `experiment__covariate_variant` to distinguish default (`full`) versus null experiment fits.
- `parameter_std__psi`: Step-level mean posterior SD for site-level occupancy probability `psi`, averaged across finite site entries. This scalar supports stopping diagnostics without reopening per-step parameter-summary artifacts.
- `parameter_q05__<parameter_slug>`: Step-level posterior 5th percentile for a fitted shared coefficient parameter.
- `parameter_q95__<parameter_slug>`: Step-level posterior 95th percentile for a fitted shared coefficient parameter.
- Step summaries export occupancy coefficients (`cov_state_*`) and detection coefficients (`cov_det_*`). Site-level `psi` agreement is computed downstream from `oracle__psi_mean_values` and `trace__psi_mean_values`.

### Score Calibration

- `score_calibration__mu0`: Fitted negative-class score mean.
- `score_calibration__sigma0`: Fitted negative-class score SD.
- `score_calibration__mu1`: Fitted positive-class score mean.
- `score_calibration__sigma1`: Fitted positive-class score SD.
- `score_calibration__n_negative`: Reviewed negative examples used in calibration.
- `score_calibration__n_positive`: Reviewed positive examples used in calibration.
