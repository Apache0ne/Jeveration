# TA-TiTok decoder code (vendored)

Source: https://github.com/bytedance/1d-tokenizer, commit `942a96fbdd873780179d1b78d5462911528bf8c8`, Apache License 2.0
(`LICENSE` in this directory). Copyright (2024) Bytedance Ltd. and/or its affiliates.

`modeling/modules/blocks.py` is copied unmodified. The two `__init__.py` files are empty (upstream's import training code
that is not needed here). The package keeps its upstream name `modeling`; `jeveration/tokenizer.py` puts this directory on
`sys.path` and uses `TATiTokDecoder` from it.
