# DNB Density Model Scaffold

This scaffold provides two supervised 2D density heatmap models that share the same PH-mask data path.

## Models

- Main: `MaskedDensityUNet`, a small U-Net/ResUNet-style image-to-density model.
- Fast: `MaskedDilatedDensityNet`, a CSRNet-inspired dilated CNN that keeps full resolution and is cheaper to run.

Both models consume:

```text
channel 0: normalized DNB brightness crop
channel 1: PH ROI mask
```

Both models output:

```text
1-channel non-negative ship-count density heatmap
```

## Shared Data Path

```text
SceneRaster + GeoJSON GT
-> raw count map
-> DnbCandidateDetector PH contours
-> rectangular crop around PH bbox + padding
-> ROI mask from PH contour pixels
-> sum-preserving Gaussian target inside ROI mask
-> masked loss inside ROI mask only
```

## Smoke Test

From repository root:

```sh
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python -m sub_module.run_density_smoke --model both --steps 2 --max-patches 24
```

Using the main config:

```sh
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python -m sub_module.run_density_smoke --config configs/dnb_density_unet_main.json
```

Using the fast config:

```sh
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python -m sub_module.run_density_smoke --config configs/dnb_density_dilated_fast.json
```

From `[3]_DNB_AIS - (STEP 3)`:

```sh
PYTHONPATH=.. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python -m sub_module.run_density_smoke --model both --steps 2 --max-patches 24
```

Run only the main model:

```sh
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python -m sub_module.run_density_smoke --model main
```

Equivalent main-only runner:

```sh
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python -m sub_module.run_density_main_smoke
```

Run only the fast model:

```sh
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python -m sub_module.run_density_smoke --model fast
```

Equivalent fast-only runner:

```sh
PYTHONPATH=. /Users/jungtaeuk/anaconda3/envs/DNB_AIS/bin/python -m sub_module.run_density_fast_smoke
```

The smoke test does not write checkpoints. It prints a JSON report with detector summary, patch summary, model parameter count, tensor shapes, and short train-step losses.
