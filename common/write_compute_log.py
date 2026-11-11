#!/usr/bin/env python3
"""Write per-run compute accounting from a SLURM job."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def parse_elapsed(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    days = 0
    if "-" in value:
        day_text, value = value.split("-", 1)
        days = int(day_text)
    parts = [int(float(p)) for p in value.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    else:
        return None
    return float(((days * 24 + hours) * 60 + minutes) * 60 + seconds)


def sacct(job_id: str | None) -> dict:
    if not job_id:
        return {}
    try:
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=Elapsed,MaxRSS", "--parsable2", "--noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        return {"sacct_error": str(exc)}
    rows = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
    elapsed = next((row[0] for row in rows if row), "")
    max_rss = next((row[1] for row in rows if len(row) > 1 and row[1]), "")
    return {"sacct_elapsed": elapsed, "sacct_max_rss": max_rss, "sacct_returncode": result.returncode}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-time", type=float, required=True)
    parser.add_argument("--gpus", type=float, default=1.0)
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID")
    wall_clock_seconds = max(0.0, time.time() - args.start_time)
    accounting = sacct(job_id)
    elapsed_seconds = parse_elapsed(accounting.get("sacct_elapsed", "")) or wall_clock_seconds
    data = {
        "slurm_job_id": job_id,
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "wall_clock_seconds": wall_clock_seconds,
        "gpu_hours": elapsed_seconds * args.gpus / 3600.0,
        **accounting,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "compute_log.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
