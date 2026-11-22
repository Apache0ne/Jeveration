# RGBD v3: independent depth / normal / segmentation and detached fusion

v1/v2 incorrectly trained a SINGLE 17-channel output network with a shared trainable trunk and a single optimizer. Although the task outputs were separate tensor slices, backpropagation through depth, normal and semantic losses modified the same upstream weights.

## Strict experiment constraints

- Identical 95 accepted Hypersim training NPZ and 20 scene-held-out validation NPZ. No new dataset downloads or cache re-encoding.
- Exactly 1200 outer iterations. EACH independent head gets 1200 optimizer updates. Fusion is trained from step 400 onward: 801 separately optimized fusion updates during the SAME 1200 outer iterations (not an extra training phase).
- All three heads have private conv stems, upsamplers, and output layers. No trainable weights or optimizer states are shared.
- The frozen Jeveration decoder still supplies a common RGB image and 32x32 decoder features. The new three heads process these values, but training one cannot alter another.
- Fusion is a separate small residual network. Predictions are `.detach()`-ed by the fusion module before concatenation; fusion cannot change any independent head gradient. Fusion leaves semantic logits unchanged, refines log-depth and camera normals conservatively.

## Losses: independent by construction

**Depth-only gradient:** metric log-depth Huber, SILog (shape consistency), two-scale valid depth reconstruction and valid-pixel log-depth gradients.

**Normal-only gradient:** cosine angular loss and local orientation edge preservation against the original depth-derived Hypersim normals. No depth-derived normal constraint is allowed to backpropagate into the normal predictor.

**Segmentation-only gradient:** bounded, training-frequency weighted CE normalized by effective pixel weights + soft Dice (ignore label 255). Validation reports mIoU and pixel accuracy.

**Fusion-only gradient:** supervised depth and normal losses, a weak geometric normal-from-depth agreement penalty and regularization against large changes from the three frozen/detached task predictions. Fusion is trained with its own optimizer in global steps 400..1200.

## Google Colab training

Mount the existing Google Drive; select L4 (BF16) or A100.

```python
!pip install -q 'git+https://github.com/Apache0ne/Jeveration.git'
```

Fit from the SAME cache, staging onto `/content` SSD and RAM:

```bash
python -u -m jeveration.rgbd_v3_train fit \
  --root /content/drive/MyDrive/Jeveration_GT/pilot \
  --save-dir /content/drive/MyDrive/Jeveration_GT/pilot/checkpoints_v3 \
  --local-cache-dir /content/jeveration_gt_pilot_cache \
  --cache-mode ram --steps 1200 \
  --batch-size 1 --grad-accum 16 \
  --lr-depth 0.00007 --lr-normal 0.00012 \
  --lr-seg 0.00018 --lr-fusion 0.00006 \
  --fusion-start 400 --eval-every 100 \
  --save-every 200 --save-at 400,700,900
```

`--resume` restores **all four** optimizers and their weights from `latest_v3.json`. Periodic `trainer_v3_step_XXXXXX.pt` snapshots save the three heads, fusion, optimizers and RNG. Head-best `best_depth.pt`, `best_normal.pt`, `best_seg.pt`, `best_fusion.pt` are independently tracked. Do not resume the old v2 checkpoint into v3.

Report with the SAME 20 validation images:

```bash
python -u -m jeveration.rgbd_v3_train report \
  --root /content/drive/MyDrive/Jeveration_GT/pilot \
  --save-dir /content/drive/MyDrive/Jeveration_GT/pilot/checkpoints_v3 \
  --local-cache-dir /content/jeveration_gt_pilot_cache \
  --output /content/Jeveration_V3_Report
```

The report ZIP contains aggregate and per-image metrics for **independent heads** and **fused predictions**, 20 visual comparisons, the original source manifest, validation history and prior v1/v2 baselines. Fusion improvements are **not assumed**; it must improve held-out metrics to justify enabling it in inference.

## Native generation

```python
from jeveration.rgbd_v3_model import make_pipeline
JeverationRGBDv3Pipeline = make_pipeline()
pipe = JeverationRGBDv3Pipeline(
    '/content/weights/jeveration.pt', decoder='/content/weights/decoder512.pt'
)
pipe.load_v3_components('/content/drive/MyDrive/Jeveration_GT/pilot/checkpoints_v3')
maps = pipe.text_to_rgbd_v3(['A stone tower in a grassy field'], pipe.generator(42))
```

Each network is a separate trainable sidecar, but **they reuse the SAME frozen RGB decoding pass**. 3 head passes + optional fusion are required; do not claim it performs all geometry in one GPU kernel.

**Limitations:** the 95-image cache is tiny and has alignment SSIM around 0.623 (median). The experiment isolates optimization interference but cannot create new geometry labels or reconstruct hidden 3D surfaces. 1,200 training iterations with three private models and optional fusion will take longer than v2. Separate best checkpoints selected on 20 validation images are not an independent test of generalization. Accuracy remains unverified until the Colab report returns.