# RGBD v2 — Mathematics-first training on exactly 95/20 images

The existing Hypersim cache is unchanged: 95 training frames, 20 held-out frames, same frozen decoder, same head architecture and exactly 1200 AdamW updates.

V1 best mixed-loss checkpoint at step 400: depth AbsRel 0.660804, normal angle 41.0705 degrees, segmentation accuracy 0.595393. Later training loss improved while held-out loss deteriorated.

## Mathematical changes

1. Depth: valid-pixel metric log-depth Huber + SILog scale-shape + two-scale depth loss + depth log-gradient penalties. Invalid pixels are excluded from every term.
2. Normals: cosine angular loss against ground-truth camera-space unit normals. A differentiable pinhole reconstruction converts predicted Z-depth into tangent-plane normals; ground-truth-normal and head-normal consistency losses couple the two geometry branches.
3. Segmentation: training-only bounded inverse-square-root class-frequency weights, CE + soft Dice and ignore=255. Evaluation adds mIoU.
4. Robustness: physically correct mirrored camera normals, mild RGB photometric jitter, decoder-feature channel dropout and 20% original RGB sampling (the original cached RGB is aligned exactly to its ground-truth geometry).
5. Optimization: cosine LR, stronger weight decay, EMA weights, and late sharpness-aware minimization beginning at step 500. This doubles forward/backward passes for later optimizer updates but keeps the original 1200 update count.
6. Model selection: save separate best joint / depth / normal / semantic mIoU checkpoints. The joint score weights depth 60%, normal angle 30%, mIoU 10%. It is evaluated on the original 20 held-out images.

## Colab fit

Install the repository from GitHub and mount Drive. This v2 experiment keeps all generated files in a NEW checkpoints_v2 directory. The original checkpoints directory is not touched.

    python -u -m jeveration.rgbd_gt_v2_train fit \
      --root /content/drive/MyDrive/Jeveration_GT/pilot \
      --save-dir /content/drive/MyDrive/Jeveration_GT/pilot/checkpoints_v2 \
      --local-cache-dir /content/jeveration_gt_pilot_cache \
      --cache-mode ram --steps 1200 --batch-size 1 --grad-accum 16 \
      --lr 0.0001 --ema-decay 0.995 --sam-start 500 --sam-rho 0.025 \
      --original-prob 0.20 --feature-drop 0.06 \
      --eval-every 100 --save-every 200 --save-at 400,700,900

Resume using the same command with --resume, only if checkpoints_v2/latest_v2.json exists.

## Evaluation

    python -u -m jeveration.rgbd_gt_v2_train report \
      --root /content/drive/MyDrive/Jeveration_GT/pilot \
      --save-dir /content/drive/MyDrive/Jeveration_GT/pilot/checkpoints_v2 \
      --local-cache-dir /content/jeveration_gt_pilot_cache \
      --output /content/Jeveration_V2_Report

Output: /content/Jeveration_RGBD_SmallData_V2_Report.zip, holding 20 visual comparisons, per-image errors, global depth/normal/semantic scores and training logs.

## Limitations

Loss redesign cannot guarantee pixel-perfect alignment between generated decoder RGB and original Hypersim geometry. The cached decoder reconstruction median SSIM is ~0.623. Neither the new math nor EMA/SAM guarantees better held-out metrics. Compare V2 directly with V1 before making claims or attempting Gaussian-splat training.