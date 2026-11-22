# Frozen Jeveration readouts — few-shot RGBD experiment

A replacement learning method using the **same 95 train / 20 validation
Hypersim cache and 1,200 optimizer updates**. No model redownload, new labels,
image re-encoding, or overwriting of v1/v2/v3 checkpoints.

## Mathematical design

1. Frozen 32×32 Jeveration decoder features are projected using a
   fixed-seed orthogonal projection. Their mean and variance come from
   **training data only**; decoded RGB, low-resolution color/edges and
   spatial coordinates augment the readout feature space at 128×128.
2. Depth starts from a closed-form **weighted ridge-regression log-depth
   predictor**. That solution initializes the weights of 48 log-depth
   probability bins, refined using log-depth Huber and interpolated-bin CE.
3. Normals start from **independent weighted ridge regression** over XYZ
   coordinates, unit normalized, and refined using cosine error.
4. Segmentation starts from **class prototypes** on frozen training features,
   then learns bounded class-balanced CE, ignoring semantic label 255.
5. A deterministic **guided upsampler** maps 128×128 predictions to 512×512.
   This is not AnyUp's learned neural upsampler.
6. Per-pixel original-vs-decoded color agreement downweights unreliable
   *training* labels. Held-out GT metrics are NOT agreement weighted.
7. Three task-specific optimizers and independently selected checkpoints;
   step-zero analytical baseline is also checkpointed. No fusion network.

This is an **original synthesis**, not the released code of any one paper.
It must be validated on the real 20-image holdout before quality claims.

## Relevant research and released code

- [LiDeRe (CVPR 2026)](https://github.com/timojl/lidere) — lightweight dense readouts on frozen features
- [AnyUp (ICLR 2026)](https://github.com/wimmerth/anyup) — universal feature upsampling
- [Probe3D (CVPR 2024)](https://github.com/mbanani/probe3d) — probing frozen features for depth and normals
- [PrAda](https://github.com/FocoosAI/PrAda) — few-shot class prototypes
- [CaBins](https://github.com/EunjinSon1/CaBins) — probabilistic depth bins
- [DSINE (CVPR 2024)](https://github.com/baegwangbin/DSINE) — camera-aware surface-normal estimation
- [Fast Guided Filter](https://arxiv.org/abs/1505.00996) — fast edge-guided image filtering

## Colab training

Mount Drive and install the repo:

~~~python
from google.colab import drive
drive.mount("/content/drive")
!pip -q install "git+https://github.com/Apache0ne/Jeveration.git"
~~~

Train on the **existing cache only**:

~~~bash
python -u -m jeveration.rgbd_frozen_probe_train fit \
  --root /content/drive/MyDrive/Jeveration_GT/pilot \
  --local-cache-dir /content/jeveration_gt_pilot_cache \
  --cache-mode ram \
  --save-dir /content/drive/MyDrive/Jeveration_GT/pilot/checkpoints_frozen_probe \
  --steps 1200 --batch-size 1 --grad-accum 16 \
  --projected 48 --ridge-reg 0.005 --init-stride 4 \
  --lr-depth 0.0002 --lr-normal 0.0003 --lr-seg 0.00025 \
  --eval-every 100 --save-every 200 --save-at 400,700,900
~~~

Add --resume when the same Drive folder contains latest_probe.json.
Both full optimizer snapshots and individual best task probes save to Drive.

Export the report:

~~~bash
python -u -m jeveration.rgbd_frozen_probe_train report \
  --root /content/drive/MyDrive/Jeveration_GT/pilot \
  --local-cache-dir /content/jeveration_gt_pilot_cache \
  --save-dir /content/drive/MyDrive/Jeveration_GT/pilot/checkpoints_frozen_probe \
  --output /content/Jeveration_FrozenProbe_Report
~~~

The report ZIP is /content/Jeveration_RGBD_FrozenProbe_Report.zip with
per-image and aggregate GT metrics, 20 visual comparisons, step-0 analytical
baseline, training curves, and v1/v2/v3 comparison metrics.

## Text-to-image with depth, normals and segmentation

~~~python
from jeveration.rgbd_frozen_probe import generate_rgbd
from jeveration.rgbd_frozen_probe_train import load_best

# pipe is a previously loaded JeverationPipeline with a plain decoder.
width = pipe.codec.decoder.width
probe = load_best(
    "/content/drive/MyDrive/Jeveration_GT/pilot/checkpoints_frozen_probe",
    width, pipe.device
)
result = generate_rgbd(pipe, probe, ["An old stone house"], pipe.generator(42))
# result contains rgb, depth, normal and seg.
~~~

This performs an additional lightweight geometry readout after the original
RGB decode. It is **not yet** Gaussian generation, and improvement on unseen
scenes is unverified until the new holdout report is returned.
