<p align="center"><img src="assets/strixite-banner.jpg" alt="strixite - LLM inference, from scratch, for AMD Strix Halo" width="100%"></p>

# strixite

An LLM inference engine I wrote from scratch in HIP for one chip - AMD's **Strix Halo** (Ryzen AI MAX+ 395 /
Radeon 8060S, gfx1151) - and one model, **Qwen3.8-Flash-Next**. No llama.cpp, no ggml, no vendor libraries
underneath: hand-written kernels, each checked against a CPU reference derived from the model's spec.

> **~47-50 tokens/s, flat from 0 to 492k tokens of context** - real agentic coding use, with the engine's own
> multi-token prediction, on one Strix Halo with 128 GB.

- Runs the ~180B-parameter Qwen3.8-Flash-Next (512-expert MoE, Gated DeltaNet linear attention, sparse attention,
  a 51B-parameter n-gram embedding table read from SSD) entirely on the iGPU, in ~66 GiB of 4/8-bit weights.
- Up to 512k tokens of context.
- An OpenAI-compatible server: chat completions with streaming, tool calls, reasoning, and a prompt cache that keeps
  long agent conversations from being re-read every turn.

**Source code: coming soon™.** I'm getting it ready to share; until then this repository is a placeholder.

**Weights:** the converted weights it runs are already up -
[wemoh/Qwen3.8-Flash-Next-strixw](https://huggingface.co/wemoh/Qwen3.8-Flash-Next-strixw) on Hugging Face.

**Platform:** strixite runs on Linux with an AMD Strix Halo (gfx1151) and 128 GB of memory, tested on Fedora 43.
It's built and tested by one person on one machine, so that's the only setup I can promise works; Windows and macOS
aren't supported, and I don't plan to add them. I can't take on ports to other platforms here, but forks are very
welcome.

**License:** [AGPL-3.0](LICENSE). The model weights are separate and under the Qwen Community License 1.0.
