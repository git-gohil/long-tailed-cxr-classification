# NIH ChestX-ray14 preprocessing

Preprocessing builds patient-disjoint training and validation manifests and
training-set summaries. The test list is used only to exclude held-out images;
test images and labels are not analysed during preprocessing.

The cluster jobs expect the dataset at `/datasets/pgohil`, the project at
`/home-mscluster/pgohil/cxr-research`, and the existing `cxr-longtail` environment
with PyTorch, torchvision, NumPy, pandas, Matplotlib, and Pillow. `tqdm` is optional.

Activate the environment and inspect the dataset from the preprocessing folder:

```bash
module load anaconda/anaconda-2025-12-2
eval "$(conda shell.bash hook)"
conda activate cxr-longtail
cd "/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection"
python scripts/inspect_nih_dataset.py --dataset-root /datasets/pgohil
```

Expected inputs are `Data_Entry_2017.csv`, `train_val_list.txt`, and `test_list.txt`
under the dataset root. Separate `--train-list` and `--val-list` arguments can
replace `--train-val-list`; do not supply both forms.

Check a small subset before submitting the full job:

```bash
python scripts/preprocess_and_analyse.py \
  --dataset-root /datasets/pgohil \
  --project-root "/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection" \
  --metadata-csv /datasets/pgohil/Data_Entry_2017.csv \
  --train-val-list /datasets/pgohil/train_val_list.txt \
  --test-list /datasets/pgohil/test_list.txt \
  --dry-run --max-images 500 --skip-hashes --num-workers 1 --verify-samples 4
```

Debug outputs go to `results_debug/`. Full preprocessing verifies every required
training and validation image before replacing outputs; missing, duplicate, or
unreadable inputs stop the run.

Submit the full preprocessing job from the same directory:

```bash
mkdir -p logs
sbatch slurm/preprocess_nih.sh
```

Wait for successful completion before training. Required outputs are
`results/data/train_manifest.csv`, `val_manifest.csv`, and `class_groups.csv`.
Dataset summaries are in `results/data/dataset_summary.json`; figures are in
`results/figures/`. The original dataset images are not modified.
