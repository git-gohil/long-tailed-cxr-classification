PATIENT-LEVEL BOOTSTRAP RESULTS

Method
- Resampling unit: patient, retaining all images for each sampled patient.
- Shared patient draws across methods and seeds allow paired comparisons.
- Point estimate: mean test mAP across training seeds.
- 95% confidence interval: percentile interval of mean mAP after patient resampling, conditional on the trained seed models.
- Training-seed variability: standard deviation across seeds, reported separately.
- Resamples: 1000
- Bootstrap RNG seed: 42
- Unique test patients: 2797

Files
- patient_bootstrap_summary.csv: overall mean mAP, seed standard deviation, and patient-bootstrap interval.
- patient_bootstrap_group_summary.csv: head/medium/tail mAP and intervals.
- patient_bootstrap_seed_metrics.csv: mAP for each method and training seed.
- patient_bootstrap_vs_bce.csv: paired mAP differences from BCE and intervals.
- patient_bootstrap_distributions.npz: bootstrap distributions.
- patient_bootstrap_details.json: configuration, methods, counts, and output descriptions.
- patient_bootstrap_map_ci.png: mean test mAP with patient-bootstrap intervals.
