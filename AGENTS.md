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
