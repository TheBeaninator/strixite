# Third-party material

strixite was written from scratch. Other engines (llama.cpp's Qwen support and its multi-token-prediction branches,
among others) were studied as read-only references; the facts were re-derived and written in my own words and code -
no source, documentation or scripts were copied from them. What this repository does contain from others is listed
here, each with its license. I've made a careful, good-faith effort to respect every license involved; if you
believe something here falls short of a license's terms, please open an issue and I'll correct it promptly.

## cpp-httplib 0.56.0 - MIT

`third_party/cpp-httplib/httplib.h`, by Yuji Hirose, vendored unchanged; its license is in
`third_party/cpp-httplib/LICENSE`. It serves strixite's HTTP API.

## Unicode Character Database - Unicode License v3

`serve/unicode_tables.inc` holds tables derived from the Unicode Character Database (general categories and
character classes from Unicode 16.0.0; canonical combining classes, decompositions and compositions from Unicode
15.0.0), generated from Python's `unicodedata` module. The tokenizer needs them to split and normalize text exactly
as the model's reference tokenizer does. The Unicode License v3 asks for its notice to accompany such data:

```
UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright © 1991-2026 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY
DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.
```

## Model weights - Qwen Community License 1.0

The model weights strixite runs are not in this repository. They are a derivative of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), published separately at
[wemoh/Qwen3.8-Flash-Next-strixw](https://huggingface.co/wemoh/Qwen3.8-Flash-Next-strixw) under the Qwen Community
License 1.0, together with that license's full text and the tokenizer / configuration files copied from the original
checkpoint.

## Build and test dependencies

Not shipped here; each is installed by whoever builds strixite, under its own license: AMD's ROCm (TheRock) toolchain
and HIP runtime, CMake, Ninja, and a GCC C++ standard library.

Optional, for the tensor-parallel tools of this fork only (`-DSTRIX_TP_RDMA`, on by default when found):
libibverbs from rdma-core (dual-licensed GPL-2.0 / BSD-2-Clause), linked dynamically by `tp_exchange_bench`,
`strix_tp` and `tp_ar`. It is not distributed here, and the engine and the server do not link it.
