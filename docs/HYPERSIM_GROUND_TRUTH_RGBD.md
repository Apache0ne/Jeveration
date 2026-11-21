# True Hypersim ground-truth RGBD pilot

This path is different from the previous pseudo-label distillation in rgbd_train.py. It does NOT generate depth targets using Jeveration's pre-trained dense heads.

## Source and limits

Dataset: https://huggingface.co/datasets/aaronyee/hypersim-train-subset (CC BY-SA 3.0). This subset contains approximately 150 paired frames across 15 official Hypersim training scenes: final-preview RGB, true rendered camera-range HDF5, and NYU40 semantic HDF5. The other subset RGB files without depth are not used.

The subset does not contain the original native normal_cam HDF5. We derive normals from actual metric depth and verified standard pinhole intrinsics (60-degree horizontal FOV, 1024x768) with a discontinuity mask. These are geometry-derived normals, not independent native normal annotations. Euclidean camera range is converted to camera-forward Z depth before losses. Uncertain NYU40 classes are ignored rather than given invented NYU13 labels.

A fixed scene-disjoint holdout provides exactly 20 validation examples, with a reserve of candidate frames in case RGB decoder alignment checks reject some. There must be >=40 successful training examples. The pilot dataset is too small for a complete high-quality 3D Gaussian model.

IMPORTANT: The public tokenizer encoder is pretrained at 256px while the fine-tuned Jeveration decoder emits 512px. Every decoded RGB must pass an SSIM >=0.40 alignment gate against the real image before its ground-truth geometry is paired with decoder features. If fewer than 20 validation examples pass, preparation aborts rather than using misaligned geometry.

## Actual supervised geometry path

RGB + true scene depth + semantics -> frozen TA-TiTok image encoder (256px) -> 128 VQ codes -> frozen Jeveration RGB decoder (512px) -> decode features + reconstructed RGB -> alignment gate -> trainable RGBDHead -> masked metric depth, depth-derived normals, mapped NYU13 semantics.

## Colab

Mount Drive:
    from google.colab import drive
    drive.mount("/content/drive")

Install:
    !pip install -q "git+https://github.com/Apache0ne/Jeveration.git"

Prepare:
    !python -m jeveration.rgbd_gt_train prepare --root /content/drive/MyDrive/Jeveration_GT/pilot --weights-dir /content/weights --min-ssim 0.40

Fit with live progress, exact step checkpoints and best GT-validation checkpoint:
    !python -m jeveration.rgbd_gt_train fit --root /content/drive/MyDrive/Jeveration_GT/pilot --save-dir /content/drive/MyDrive/Jeveration_GT/pilot/checkpoints --steps 1200 --batch-size 1 --grad-accum 16 --lr 0.00015 --warmup-steps 100 --seg-weight 0.3 --eval-every 100 --save-every 200 --save-at 500,900 --log-every 10

Resume by adding --resume after mounting the same Drive folder.

Ground-truth validation report:
    !python -m jeveration.rgbd_gt_train report --root /content/drive/MyDrive/Jeveration_GT/pilot --save-dir /content/drive/MyDrive/Jeveration_GT/pilot/checkpoints --output /content/Jeveration_GT_report

Result /content/Jeveration_Hypersim_GT_Report.zip contains metrics.json, 20 image comparisons, individual metrics, run manifest, training metrics and validation logs. Actual weights remain in Drive.

## Scaling up

This is a first full-code, small-data alignment pilot, NOT the final scale. For production-quality geometry use a much larger official Hypersim sample with true distance/depth, normal_cam, semantics and cameras. The old 4,200-image pseudo-labeled dataset should remain separate. Redistribution and downstream licensing must respect Apple's CC BY-SA 3.0 terms.
