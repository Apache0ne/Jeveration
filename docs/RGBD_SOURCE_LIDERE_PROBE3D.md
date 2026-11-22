# Source-faithful LiDeRe + Probe3D RGBD training

The previous frozen-probe experiment was NOT a reproduction of the six research methods. This new experiment uses the **actual unmodified published implementations** for TWO compatible components: LiDeRe dense readout and Probe3D depth/normal objectives. It keeps the original 95 Hypersim train examples, 20 held-out examples, and exactly 1200 updates per independent head.

## Exact upstream source files — MIT

| Source | Vendored path | Git blob SHA |
|---|---|---|
| [LiDeRe](https://github.com/timojl/lidere/blob/master/lidere/models/lidere.py) | jeveration/vendor/lidere.py | 5aafadc4eb21b8c69ad2d8bc863fac35d4528d58 |
| [Probe3D probes](https://github.com/mbanani/probe3d/blob/main/evals/models/probes.py) | jeveration/vendor/probe3d_probes.py | d926f10a746021efa1f904c0772776f2bad1ed4b |
| [Probe3D losses](https://github.com/mbanani/probe3d/blob/main/evals/utils/losses.py) | jeveration/vendor/probe3d_losses.py | df4e9ace89ec6a353746c309a05646175c0d27ae |

The LiDeRe license is reproduced in jeveration/vendor/LICENSE_LIDERE. Probe3D copyright and MIT license text remains inside the original source files. Unit tests verify Git blob hashes exactly.

## Source-faithful architecture

The existing cached 32×32 TA-TiTok feature map is given to the ORIGINAL LiDeRe.forward_feats with shape [B,C,1,32,32]. The original trainable LiDeRe LayerFeatureMLP, implicit learned spatial-attention LayerImplicitAtt and UpsampleCNN are not rewritten or substituted. Three separate LiDeRe models predict 256 depth-bin channels, 3 normal channels and 13 semantic logits. DepthBinPrediction from the original Probe3D implementation converts 256 bins to metric depth. DepthLoss and angular_loss from original Probe3D are used in training. The segmentation task uses standard class cross-entropy.

The experiment uses base_size=64 by default, giving learned LiDeRe 256×256 output and official Probe3D-style interpolation to the 512×512 GT resolution. base_size=128 yields native 512×512 LiDeRe predictions but requires much higher attention memory and compute. No new dataset, teacher pseudo-labels, codebook re-encoding or extra image generation.

## Why we are NOT claiming the other papers are implemented

- AnyUp (2026) is a distinct pretrained feature-upscaling model, not LiDeRe's own learned upsampler. Its official network and weights would require a separate controlled experiment: https://github.com/wimmerth/anyup
- PrAda (2026) requires FC-CLIP, class-mask pooling and fusion with the original frozen text logits. Standalone class prototypes in Jeveration are not PrAda: https://github.com/FocoosAI/PrAda
- CaBins requires its CLIP-based adaptive bins; Probe3D's 256 uniform depth bins are NOT CaBins. The CaBins repo lacks a clear reusable license: https://github.com/EunjinSon1/CaBins
- DSINE full camera-aware normals architecture is NOT this 3-vector readout. Its original repository has restrictive research-use terms: https://github.com/baegwangbin/DSINE

## Reproduce on the same data

Mount Drive and install this GitHub repo. Use the four-cell notebook in notebooks/Jeveration_Source_LiDeRe_95_Colab.ipynb, or run:

~~~bash
python -u -m jeveration.rgbd_source_lidere_train fit \
  --root /content/drive/MyDrive/Jeveration_GT/pilot \
  --save-dir /content/drive/MyDrive/Jeveration_GT/pilot/checkpoints_source_lidere \
  --local-cache-dir /content/jeveration_gt_pilot_cache \
  --cache-mode ram --steps 1200 --batch-size 1 --grad-accum 16 \
  --dim 64 --base-size 64 --max-depth 80 \
  --lr-depth 0.00015 --lr-normal 0.00015 --lr-seg 0.00015 \
  --eval-every 100 --save-every 200 --save-at 400,700,900
~~~

To resume, add --resume after a state file has been saved. Original v1/v2/v3/frozen-probe checkpoints are untouched.

~~~bash
python -u -m jeveration.rgbd_source_lidere_train report \
  --root /content/drive/MyDrive/Jeveration_GT/pilot \
  --save-dir /content/drive/MyDrive/Jeveration_GT/pilot/checkpoints_source_lidere \
  --local-cache-dir /content/jeveration_gt_pilot_cache \
  --output /content/Jeveration_Source_LiDeRe_Report
~~~

Output is /content/Jeveration_Source_LiDeRe_Probe3D_Report.zip; it includes 20 visual comparisons and held-out depth AbsRel, normal angular error, semantic accuracy and mIoU.

**Scientific limitations:** The LiDeRe method is source-faithful but the frozen features are Jeveration decoder features rather than the pretrained DINOv3 features used in the paper. Reconstructed RGB alignment to Hypersim GT is imperfect. CPU regression tests do not prove accuracy improvement; 1200-step L4 metrics have not been measured yet.