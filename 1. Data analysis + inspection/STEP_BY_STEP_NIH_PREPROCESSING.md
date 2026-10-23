# Step-by-Step Guide: NIH ChestX-ray14 Training-Only Preprocessing and EDA

This guide explains how to run the safe inspection, dry run, and full SLURM job on the Wits Mathematical Sciences HPC cluster.

Important safety rule: do not analyse, preprocess, count labels from, or open test images. The scripts only use a test split file as a filename exclusion list.

## 1. Log in to the cluster

Run this on your own computer terminal.

```bash
ssh pgohil@<cluster-login-address>
```

This logs you in to the HPC cluster. Replace `<cluster-login-address>` with the actual Wits cluster login address.

## 2. Go to your home/work directory

Run this on the cluster login node.

```bash
cd /home-mscluster/pgohil
```

This moves you to your home/work directory.

## 3. Enter the preprocessing directory

Run this on the cluster login node.

```bash
cd "/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection"
```

This moves you into the preprocessing folder inside `codex research 2`.
The relative commands below run from this folder.

If the folder does not exist yet, create it first:

```bash
mkdir -p "/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection"
cd "/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection"
```

This creates the preprocessing folder and enters it.

## 4. Create the required folders

Run this inside `/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection`.

```bash
mkdir -p scripts slurm logs results/data results/figures
```

This creates folders for scripts, SLURM jobs, logs, CSV/JSON outputs, and figures.

## 5. Check that the three files are present

Run this inside `/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection`.

```bash
ls -lh scripts slurm
```

This lists the script and SLURM folders.

You should see:

```text
scripts/inspect_nih_dataset.py
scripts/preprocess_and_analyse.py
slurm/preprocess_nih.sh
```

If these files are not there, copy them into the matching folders before continuing.

## 6. Load Conda and activate your environment

Run this on the cluster login node.

```bash
module load anaconda/anaconda-2025-12-2
eval "$(conda shell.bash hook)"
conda activate cxr-longtail
```

This loads Anaconda and activates your existing `cxr-longtail` Python environment.

Check Python:

```bash
which python
python --version
```

This confirms which Python will run the scripts.

## 7. Safely inspect the dataset structure

Run this on the cluster login node.

```bash
python scripts/inspect_nih_dataset.py --dataset-root /datasets/pgohil
```

This safely searches `/datasets/pgohil` without opening images and without entering directories clearly named test, testing, heldout, or held-out.

Look for these sections in the output:

```text
Likely NIH metadata CSV files
Likely training/validation split or list files
Likely test filename exclusion files
Top directories containing images
```

Write down the best paths for:

```text
metadata CSV
training list, validation list, or train-validation list
test filename exclusion list
```

## 8. Decide which arguments to use

If the inspection finds a metadata CSV and split files, use their exact paths.

Example:

```text
metadata CSV: /datasets/pgohil/Data_Entry_2017.csv
train/validation list: /datasets/pgohil/train_val_list.txt
test exclusion list: /datasets/pgohil/test_list.txt
```

If you have separate train and validation files, use `--train-list` and `--val-list`.

If you have one combined train-validation file, use `--train-val-list`.

The test file must only be used with `--test-list`.

Use either `--train-val-list` or separate `--train-list`/`--val-list`
arguments. Explicit lists take precedence over automatic discovery; supplying
both forms is rejected. Automatic validation-list discovery excludes combined
train/validation lists.

Before replacing results, preprocessing requires every selected filename to
have metadata and exactly one image path, and opens every training and
validation image to check readability. Missing, duplicate or unreadable required
images, missing metadata, and empty supplied split lists stop the run while
preserving existing result files. Repair the reported inputs before rerunning.
Training EDA still uses training images only; test images are never opened.

## 9. Run a small dry run

Run this on the cluster login node only because it processes a small debug subset.

Edit the paths if your inspection output showed different files.

```bash
python scripts/preprocess_and_analyse.py \
  --dataset-root /datasets/pgohil \
  --project-root "/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection" \
  --metadata-csv /datasets/pgohil/Data_Entry_2017.csv \
  --train-val-list /datasets/pgohil/train_val_list.txt \
  --test-list /datasets/pgohil/test_list.txt \
  --dry-run \
  --max-images 500 \
  --skip-hashes \
  --num-workers 1 \
  --verify-samples 4
```

This builds a tiny training-only manifest, creates a validation split, calculates small EDA outputs, and verifies preprocessing on a few training images.
Both `--dry-run` and any `--max-images` subset write to `results_debug/`
under the supplied `--project-root`, preserving the full-run `results/` files.
Repeated debug runs replace only the previous debug outputs. Full preprocessing
without either option continues to use `results/`.

If you have separate train and validation lists, use this instead:

```bash
python scripts/preprocess_and_analyse.py \
  --dataset-root /datasets/pgohil \
  --project-root "/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection" \
  --metadata-csv /datasets/pgohil/Data_Entry_2017.csv \
  --train-list /datasets/pgohil/train_list.txt \
  --val-list /datasets/pgohil/val_list.txt \
  --test-list /datasets/pgohil/test_list.txt \
  --dry-run \
  --max-images 500 \
  --skip-hashes \
  --num-workers 1 \
  --verify-samples 4
```

This uses the existing validation split instead of creating one.

## 10. Check the dry-run output

Run this inside `/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection`.

```bash
ls -lh results_debug/data
ls -lh results_debug/figures
```

This lists the generated dry-run CSV, JSON, and PNG files.

Check the summary JSON:

```bash
python -m json.tool results_debug/data/dataset_summary.json | less
```

This opens the summary in a readable format.

Exit `less` by pressing:

```text
q
```

## 11. Check common dry-run problems

If you see this:

```text
Could not auto-detect metadata CSV
```

Run again with:

```bash
--metadata-csv /exact/path/from/inspection.csv
```

If you see this:

```text
Training/validation filename list overlaps with test exclusion list
```

Stop and check your split files, because the training data is overlapping the held-out test list.

If you see this:

```text
Patient ID is required for a safe patient-level validation split
```

Use an existing validation split with `--train-list` and `--val-list`, or check that the metadata CSV has a patient ID column.

If you see missing packages such as `torch`, `torchvision`, `pandas`, or `PIL`, install or load them in the `cxr-longtail` environment before continuing.

## 12. Edit the SLURM script if paths are different

Run this inside `/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection`.

```bash
nano slurm/preprocess_nih.sh
```

This opens the SLURM script for editing.

Check these lines:

```bash
PROJECT_ROOT="/home-mscluster/pgohil/cxr-research"
EXP_DIR="${PROJECT_ROOT}/1. Data analysis + inspection"
DATASET_ROOT="/datasets/pgohil"
CONTAINER="/usr/local/containers/anaconda/anaconda-2025-12-2.sif"
ENV_PYTHON="/home-mscluster/pgohil/.conda/envs/cxr-longtail/bin/python"
```

If your metadata or split files must be passed explicitly, edit the Python command near the bottom.

Example for a combined train-validation list:

```bash
apptainer exec --bind /datasets:/datasets "${CONTAINER}" "${ENV_PYTHON}" scripts/preprocess_and_analyse.py \
  --dataset-root "${DATASET_ROOT}" \
  --project-root "${EXP_DIR}" \
  --metadata-csv /datasets/pgohil/Data_Entry_2017.csv \
  --train-val-list /datasets/pgohil/train_val_list.txt \
  --test-list /datasets/pgohil/test_list.txt \
  --num-workers 2
```

Save in `nano` by pressing:

```text
Ctrl + O
Enter
Ctrl + X
```

## 13. Submit the full job

Run this inside `/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection`.

```bash
sbatch slurm/preprocess_nih.sh
```

This submits the full preprocessing and EDA job to SLURM.

Successful submission looks like:

```text
Submitted batch job 123456
```

The number is your job ID.

## 14. Check whether the job is queued or running

Run this on the cluster login node.

```bash
squeue -u pgohil
```

This shows your queued and running jobs.

## 15. Watch the output log

Run this inside `/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection`.

Replace `123456` with your real job ID.

```bash
tail -f logs/preprocess_nih_123456.out
```

This shows progress messages from the job.

Stop watching by pressing:

```text
Ctrl + C
```

## 16. Watch the error log

Run this inside `/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection`.

Replace `123456` with your real job ID.

```bash
tail -f logs/preprocess_nih_123456.err
```

This shows warnings and errors from the job.

Stop watching by pressing:

```text
Ctrl + C
```

## 17. Check detailed job information

Run this on the cluster login node.

Replace `123456` with your real job ID.

```bash
scontrol show job 123456
```

This shows detailed information about the job, including state, requested resources, and runtime.

## 18. Cancel the job if necessary

Run this on the cluster login node.

Replace `123456` with your real job ID.

```bash
scancel 123456
```

This stops the job.

## 19. Check whether the job completed successfully

Run this inside `/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection`.

Replace `123456` with your real job ID.

```bash
tail -n 40 logs/preprocess_nih_123456.out
```

This shows the final lines of the output log.

A successful run should include:

```text
Training-only NIH preprocessing/EDA summary
Train images:
Validation images:
Reports: /home-mscluster/pgohil/cxr-research/1. Data analysis + inspection/results/data
Figures: /home-mscluster/pgohil/cxr-research/1. Data analysis + inspection/results/figures
Finish date:
```

Also check the error log:

```bash
tail -n 40 logs/preprocess_nih_123456.err
```

This shows the final errors or warnings, if any.

## 20. List the generated data outputs

Run this inside `/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection`.

```bash
ls -lh results/data
```

This lists the generated machine-readable files.

Important files include:

```text
train_manifest.csv
val_manifest.csv
class_frequencies.csv
class_groups.csv
class_weights.csv
dataset_summary.json
label_combination_counts.csv
cooccurrence_counts.csv
cooccurrence_percentages.csv
conditional_probabilities.csv
label_jaccard.csv
data_quality_issues.csv
training_image_sampling_weights.csv
training_image_integrity.csv
preprocessing_verification.csv
preprocessing_contact_sheet.png
```

There should be no test manifest.

## 21. List the generated figures

Run this inside `/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection`.

```bash
ls -lh results/figures
```

This lists the generated PNG plots.

Important figures include:

```text
class_frequency_bar.png
class_frequency_log.png
pathology_prevalence.png
head_medium_tail_groups.png
cooccurrence_heatmap.png
label_count_per_image_distribution.png
images_per_patient_distribution.png
original_image_dimensions.png
train_vs_validation_prevalence.png
```

If patient age is available and usable, this file should also appear:

```text
age_distribution.png
```

## 22. Preview CSV files

Run this inside `/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection`.

```bash
head -n 5 results/data/class_frequencies.csv
```

This shows the first few rows of the class frequency table.

```bash
head -n 5 results/data/train_manifest.csv
```

This shows the first few rows of the training manifest.

## 23. Read the JSON summary

Run this inside `/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection`.

```bash
python -m json.tool results/data/dataset_summary.json | less
```

This opens the summary report in a readable format.

Press `q` to exit.

## 24. Copy PNGs to your computer if needed

Run this on your own computer terminal, not inside the cluster session.

```bash
scp 'pgohil@<cluster-login-address>:/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection/results/figures/*.png' .
```

This copies the generated figures from the cluster to your current local folder.

## 25. Rerun safely

Run this inside `/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection`.

```bash
sbatch slurm/preprocess_nih.sh
```

This reruns the job and overwrites only the project outputs under:

```text
/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection/results
```

It does not overwrite or modify the original NIH dataset under:

```text
/datasets/pgohil
```

## 26. Recommended full-job settings

Use these defaults first:

```text
partition: batch
nodes: 1
tasks: 1
CPUs per task: 4
memory: 24G
time: 08:00:00
num workers: 2
GPU: no
container: /usr/local/containers/anaconda/anaconda-2025-12-2.sif
Python: /home-mscluster/pgohil/.conda/envs/cxr-longtail/bin/python
```

Keep `--num-workers 2` at first. Increase it only if the cluster storage is not overloaded and the job is too slow.

## 27. What the pipeline does

The pipeline uses training data only to create:

- training and validation manifests
- class frequency statistics
- long-tailed head/medium/tail grouping
- class weights
- sampling weights
- co-occurrence matrices
- label combination counts
- image integrity statistics
- preprocessing verification
- research-report figures

The pipeline does not create a test manifest and does not open test images.

## 28. What preprocessing is verified

The verification mode loads a few permitted training images and checks this deterministic transform:

```text
open image safely
convert to grayscale
resize to 256 x 256 with bilinear interpolation
centre crop to 224 x 224
replicate grayscale to 3 channels
convert to tensor
normalise with ImageNet mean and standard deviation
```

Expected tensor shape:

```text
[3, 224, 224]
```

The script also checks for NaN and infinite values.

## 29. What not to do

Do not run the full preprocessing job directly on the login node.

Do not create a test manifest.

Do not point `--train-list`, `--val-list`, or `--train-val-list` at a test file.

Do not copy, rename, move, or modify files inside `/datasets/pgohil`.

Do not permanently save randomly augmented images.
