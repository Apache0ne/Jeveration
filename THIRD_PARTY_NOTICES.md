# Third-party notices

Jeveration builds on the following third-party code and models. Only the first item is redistributed in this repository;
the models are downloaded at run time from their own repositories, under their own licenses.

## TA-TiTok (1d-tokenizer): code included here

- Source: https://github.com/bytedance/1d-tokenizer, commit `942a96fbdd873780179d1b78d5462911528bf8c8`
- Copyright (2024) Bytedance Ltd. and/or its affiliates. Licensed under the Apache License, Version 2.0; the license text is
  in `jeveration/third_party/tatitok/LICENSE`.
- Included file, unmodified: `jeveration/third_party/tatitok/modeling/modules/blocks.py` (the two `__init__.py` files next
  to it are empty).
- Derived code: `SourceDecoder` in `jeveration/tokenizer.py` subclasses `TATiTokDecoder` of that file and repeats its forward
  pass with an added source-conditioning branch.
- The released decoder weights are fine-tuned from the decoder of the TA-TiTok VQ-128 tokenizer
  (`turkeyju/tokenizer_tatitok_bl128_vq` on the Hugging Face Hub). The tokenizer's codebook and configuration are read
  from that repository at run time.

## Qwen3.5-4B: downloaded at run time

- `Qwen/Qwen3.5-4B` on the Hugging Face Hub, used as the frozen backbone (language model and vision tower). Its weights are
  not redistributed here; they are downloaded under the model's own license (see its model card). The Jeveration checkpoint
  holds only adapters and new modules that are used together with these weights.

## CLIP ViT-L/14-336 text tower: downloaded at run time

- `timm/vit_large_patch14_clip_336.openai` on the Hugging Face Hub (OpenAI CLIP weights in open_clip format), loaded with
  `open_clip`. It provides the text input of the TA-TiTok decoder. See its model card for the license.

## Optional structure-map annotators: downloaded at run time, only with `--from-photo`

- HED: the weights `ControlNetHED.pth` of `lllyasviel/Annotators` on the Hugging Face Hub. The small network definition in
  `jeveration/annotators.py` follows the HED annotator of ControlNet 1.1 (https://github.com/lllyasviel/ControlNet-v1-1-nightly)
  so that these weights load.
- Depth: `depth-anything/Depth-Anything-V2-Small-hf` on the Hugging Face Hub, loaded with `transformers`.
- See that project and the two model cards for the licenses.

## Python packages

PyTorch, Transformers, open_clip, flash-linear-attention, OmegaConf, einops, NumPy, Pillow, safetensors and
huggingface_hub are installed separately (`requirements.txt`) under their own licenses.
