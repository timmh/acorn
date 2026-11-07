"""Configurations shared by the single-species runner and manuscript sweep."""

MAIN_EXPERIMENT_IDS = (
    "reviewed_random",
    "reviewed_max_score",
    "reviewed_bald",
    "reviewed_posterior_predictive_uncertainty",
    "reviewed_target_eig",
    "cut_normal_calibrated_random",
    "cut_normal_calibrated_max_score",
    "cut_normal_calibrated_bald",
    "cut_normal_calibrated_posterior_predictive_uncertainty",
    "cut_normal_calibrated_target_eig",
    "original_occu_cs_bald",
    "original_occu_cs_target_eig",
)
TARGET_EXPERIMENT_IDS = {
    "combined": "cut_normal_calibrated_target_eig",
    "site_occupancy": "cut_normal_calibrated_target_eig_psi",
    "occupancy_coefficients": "cut_normal_calibrated_target_eig_occupancy_coefficients",
    "detection_coefficients": "cut_normal_calibrated_target_eig_detection_coefficients",
}
TASK_EXPERIMENT_IDS = tuple(value for key, value in TARGET_EXPERIMENT_IDS.items() if key != "combined")
FINAL_EXPERIMENT_IDS = MAIN_EXPERIMENT_IDS + TASK_EXPERIMENT_IDS
