# Agent Instructions

## Git Automation Policy

When meaningful project progress is made, update git automatically unless the user explicitly says not to write or not to commit.

Meaningful progress includes:

- A feature, bug fix, or workflow change is implemented and verified.
- A design decision is recorded in docs or config.
- A deep-learning experiment scaffold, model, target builder, detector, or training/evaluation utility is added or materially changed.
- A reproducibility/hygiene change affects future runs.
- A completed experiment produces a small curated config/manifest/metric summary that should be preserved.

Before committing:

- Run `./scripts/git_ai_status.sh` when available.
- Inspect `git status --short` and stage only files related to the completed work.
- Never stage unrelated user changes.
- Never stage generated heavy artifacts: checkpoints, `.pt`, `.pth`, `.ckpt`, `.tif`, `.nc`, `.db`, `.npy`, `.npz`, raw AIS, or bulk `outputs/` contents.
- Run relevant lightweight checks, for example `python -m py_compile ...`, notebook strip checks, or `git diff --check`.
- If unrelated changes make a clean focused commit impossible, do not force it; report the reason and leave the worktree untouched.

Commit format:

```text
<type>(<scope>): <short imperative summary>
```

Preferred types:

- `feat`: new pipeline/model/data functionality
- `fix`: bug fix or correctness repair
- `docs`: design notes or workflow documentation
- `config`: experiment or runtime configuration
- `chore`: repo hygiene or maintenance
- `exp`: experiment setup, metrics, or reproducibility scaffold

After committing:

- Report the commit hash and one-line summary to the user.
- Mention any remaining uncommitted changes and whether they are unrelated or intentionally deferred.
- Do not push to a remote unless the user explicitly asks for push/sync.

## Deep Learning Artifact Policy

Model code, configs, design docs, and lightweight manifests belong in git. Runtime artifacts belong outside git under `artifacts/` or `outputs/` with metadata that records:

- `git_commit`
- `git_dirty`
- config path and config hash
- scene split manifest
- target generation config
- model config
- checkpoint paths

If a checkpoint is produced from dirty code, save `git diff` into the run directory as `run_git_dirty.patch`.

## Experiment Monitoring Policy

When starting any long-running experiment, training run, preprocessing batch, or detached/background process:

- Report the run directory as a clickable path.
- Report the main log file as a clickable path when one exists.
- Provide the exact command to monitor progress, usually `tail -f <run.log>`.
- Provide the exact command to stop the run safely, for example `screen -S <session> -X quit` or `kill <pid>`.
- If the run uses `screen`, `tmux`, `nohup`, or a similar detached mechanism, report the session name or PID.
- Prefer unbuffered logging for Python long-runs, for example `PYTHONUNBUFFERED=1`, so monitoring commands show live progress.

## Citation And Final Report Policy

When adding, removing, or correcting references for the final EESRL report:

- Update `docs/FINAL_REPORT_CITATIONS_IEEE.md` first.
- Keep only citations tied to the active DNB/AIS density pipeline in the main list.
- Put broader background references in the optional section instead of mixing them into the core method list.
- Do not cite retired GAT, SAR, radar, or unrelated general-ML references unless the final report explicitly discusses a discarded design path.
- Keep citation metadata and rationale in git; do not add local PDF files from `_Readings/`.

When changing final-report structure or figure/table planning:

- Update `docs/FINAL_REPORT_PREP_IEEE.md`.
- Keep the final report narrative aligned with the active PH-assisted U-Net density heatmap pipeline.
- Treat the previous DRUID+GAT midterm direction as historical context only unless the user explicitly asks to revive it.
