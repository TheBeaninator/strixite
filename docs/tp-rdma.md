# Tensor parallelism over RDMA (this fork)

This fork of [strixite](https://github.com/shawnshekari/strixite) runs one Qwen3.8-Flash-Next decode stream across
**2 or 4 Strix Halo machines** with tensor parallelism. The machines exchange partial sums over 100 GbE RDMA
(RoCE v2). Each machine holds only its share of the weights. Rank 0 drives the others and runs the MTP layer; the
draft head's vocabulary is split across all ranks. On one machine the arithmetic is unchanged: a single machine
computes bit-identical results with and without these changes (see [Correctness](#correctness)). The fork does change
one single-machine default: the MTP draft policy (see [MTP draft policy](#mtp-draft-policy)).

## How it works

- **Split loader** (`runtime/strixw_loader`, `TpConfig{world, rank}`). Each rank reads only the rows it keeps from
  the same `.strixw` file. A rank of 2 reads about 47 of 69.5 GiB. The split covers:
  - GDN key/value heads
  - attention query heads (KV heads split 2/N, duplicated at N = 4)
  - the routed and shared experts' intermediate (gate/up rows, down columns)
  - the LM head rows by vocabulary

  The embedding, HC mixes, norms, router, PLE and QSA indexer are replicated. The MTP layer lives whole on rank 0.
- **Exchanges** (`runtime/tp_comm`, `TpComm`):
  - 96 all-reduces per forward, at the mixer and MoE row-parallel outputs, plus an exact merge of the LM-head
    candidates.
  - A communication thread posts RDMA WRITE-with-immediate into 3 rotating windows per peer. It uses libibverbs with
    the stock rdma-core mlx5 provider, not the GPU's peer memory.
  - Every rank sums the partials in FP32 in rank order and rounds once, so all ranks hold bit-identical activations.
  - Large exchanges (prefill, 1 MiB and up) run as a reduce-scatter plus all-gather: each rank sums only its slice,
    in the same rank order with the same single rounding, and sends back BF16. The results are bit-identical to the
    all-to-all path, and each rank sends (N-1) x 6 / N instead of (N-1) x 4 bytes per element (TP4: 90 instead of
    240 MiB per 8,192-token exchange).
  - Any error poisons every wait: a work-completion error, the watchdog, or a peer failure.
  - The TCP mesh used for the handshake stays open as the control channel.
- **Mirror protocol** (`runtime/tp_mirror`):
  - Rank 0 (`TpDriver`) sends every state-changing call to the executors before making it: forwards, MTP verifies,
    their keeps and drops, snapshots and resets.
  - The executors (`tp_executor`) replay those calls.
  - Optional checks hash the replicated activations and session state across ranks after every call.
  - `TpLoopback` runs the same protocol in one process, for the mirror-shadow test.
- **FP32 partials (the TP default).** Under TP the mixer and MoE partials stay FP32 until the exchange. With BF16
  partials, four-node perplexity drifted +0.71% from one node; with FP32 partials it is +0.15%.
- **Split draft head.** The MTP draft scores 65,536 vocabulary rows per call. Under TP each rank scores its own
  quarter or half of them (Q4, quantized identically on every rank). Rank 0 sends the draft input to every rank,
  each rank picks its local top-2, and rank 0 merges the parts exactly the way the whole-head pick does (lowest id
  on a tie). The drafts, and so the outputs, are identical to scoring the whole head on rank 0. Executors load
  their draft rows too (+42 MB at TP4). This changes the mirror protocol to v3.
- **Small-output GEMV occupancy.** At TP4 a rank's shared-expert gate/up has only 320 outputs, which left most of
  gfx1151's 40 compute units idle at 2..8 rows. The Q4/Q8 GEMVs now use fewer rows per block until the grid has 160
  blocks. A row's arithmetic does not change, so outputs are bit-identical.
- **K-tail WMMA kernels.** A TP-4 rank's expert and shared-expert down projections have K = 640 / 4 = 160. The Q4/Q8
  WMMA GEMMs gain a tail path for K % 64 == 32. At world 1 every K is a multiple of 64, so the old kernels run.

## MTP draft policy

`strix_server` and `tp_ar` use upstream's MTP defaults: margin 2.0, up to 5 drafts, and the Q4 draft head (drafts score
a Q4 copy of the draft vocabulary's LM head rows, made once at load; upstream since strixite PR #2). `tp_ar` takes
`--mtp-margin`, `--mtp-draft` and `--mtp-draft-q4 on|off` for comparisons, and `--mtp-sweep` for a grid of them.
`strix_bench` drafts over the Q8 rows unless `--mtp-draft-q4 on`, so its `--hash-run` stays comparable across builds.

The four-node numbers above were measured before the fork moved to upstream's defaults, with margin 1.5 and up to 7
drafts (the history table's first row); they will be re-measured.

## Hardware it was measured on

- Strix Halo mini PCs (gfx1151, 128 GB), EC "balanced" power mode (85 W).
- One Mellanox/NVIDIA ConnectX-6 Dx 100 GbE NIC per machine, through a 100 GbE switch (SN2700), RoCE v2.
- Things these machines needed:
  - `iommu=pt` on the kernel command line, or RDMA traffic stalls.
  - An unlimited locked-memory limit for the process (`LimitMEMLOCK=infinity` under systemd, or `ulimit -l unlimited`).
  - An IPv4 address on each NIC. The device is found by its RoCE v2 GID for that address.

## Weights for TP

TP needs every split dimension to be a multiple of its quantization group. At N = 4 the expert intermediate is 160 per
rank, so the expert down projection and the shared expert's down projection must use group size 32. One conversion
serves both N = 2 and N = 4:

```
build/strix/convert_qwen4exp --src ORIGINAL_MODEL_DIR --out OUTPUT_DIR --git HASH \
    --layout default=q8g64,exp_gu=q4g128,exp_down=q4g32,sh_down=q8g32,gdn_in=q4g128
```

([docs/tools.md](tools.md) describes the converter.) Every machine needs this `.strixw` file (local NVMe is much faster
to load than a network share) and the n-gram table.

## Build

```
sudo apt install libibverbs-dev rdma-core        # or your distribution's rdma-core packages
cmake --preset strix && cmake --build build/strix
```

The build produces:
- `tp_exchange_bench`
- `strix_tp` (the communicator)
- `tp_ar`
- `strix_bench`
- `test_wmma_ktail`

`-DSTRIX_TP_RDMA=OFF` builds without the RDMA library and tools. The engine and `strix_server` never link libibverbs
either way.

## Run

Every rank runs the same binary with the same weights and arguments, except `--tp-rank`. `--tp-peers` lists every
rank's NIC address in rank order. Rank r listens on `--tp-port` + r.

```
# on each of 4 machines, R = 0..3 (rank 0 reports; the others execute rank 0's calls and exit)
tp_ar --weights W.strixw --ngram NGRAM.table --tokenizer tokenizer.json \
      --corpus wiki.train.raw --kl-corpus wiki.test.raw \
      --tp-world 4 --tp-rank R --tp-peers 192.0.2.1,192.0.2.2,192.0.2.3,192.0.2.4 \
      --depths 4096,65536 --mtp 1 --out r.json
```

The device is picked automatically: the mlx5 device with a RoCE v2 GID for the rank's address. `--tp-dev` overrides
it.

Useful checks:

| Command | What it checks |
|---|---|
| `test_wmma_ktail` | WMMA kernels at K = 160 / 320 / 640 against the FP32 kernels and a host reference |
| `test_mtp_split_pick` | The split draft pick (per-rank parts + merge) against the whole-head pick: 418 cases, including ties across rank boundaries, NaN, +-0 and empty parts, for N = 2 and 4 |
| `strix_bench --hash-run 1` | One fixed call sequence, every result hashed. Two builds must print the same three hashes |
| `tp_ar --shadow 1` | Mirror-shadow test: driver and executor sessions in one process over `TpLoopback`, the real MTP loop with injected drops, restores and failures. No RDMA is used. |
| `tp_ar --tp-world 2 --tp-peers IP,IP` (two processes, one machine) | Loopback smoke test through one NIC |
| `tp_ar ... --hash 1 --golden-in FILE` | Quality across machines: replicated-activation hashes, KL, top-1 and perplexity against a one-node golden file (`--golden-out` on one machine) |
| `tp_exchange_bench` | The exchange alone (latency per all-reduce, bandwidth load) |

## Results

All results are one stream on the hardware above. Each A/B ran in alternating order with the same binaries
(A, B, A, B), using a teacher-forced replay of the engine's MTP loop over each configuration's own greedy
continuation. The changes marked bit-identical were also checked hash for hash.

### Current build, four machines

| Context | MTP, 4 nodes | Prefill, 4 nodes |
|---|---|---|
| 4k | ~114 t/s | |
| 64k | ~121 t/s | 37.0 s (one node: ~46 s) |
| 400k | ~108 t/s | 213 s (one node: ~278 s) |

How the four-node numbers moved since the first build below (each step an alternating A/B on four machines):

| Change | 4k | 64k | 400k |
|---|---|---|---|
| MTP policy 1.5 x 7 + Q4 draft head (MTP t/s) | 99.1 -> 107.2 | 111.7 -> 115.4 | 103.2 -> 102.5 |
| Split draft head (MTP t/s, outputs identical) | 106.8 -> 114.4 (+7.1%) | 115.9 -> 121.3 (+4.7%) | 102.7 -> 107.9 (+5.1%) |
| Small-output GEMV occupancy (MTP, bit-identical) | +0.6% | +1.1% | |
| Reduce-scatter prefill exchanges (prefill s, bit-identical) | | 55.2 -> 37.0 | 310 -> 213 |

Quality of the current build vs one node: plain-decode perplexity +0.15%, MTP perplexity -0.34%, 0 activation hash
mismatches across the four ranks.

### First build: four machines vs one

Measured 2026-10-06. Method:
- One binary per session, with one-node and four-node runs alternated.
- The one-node baseline is the faster of two machines at each depth.
- "Plain" means greedy decode, teacher-forced, p50 per token.
- MTP uses the model's own draft head (5 drafts, margin 2.0, draft vocabulary 65,536), teacher-forced over each
  configuration's own greedy continuation. Absolute MTP numbers depend on how predictable the text is; the ratios
  are the fair comparison.

| Context | Plain, 1 node | Plain, 4 nodes | Speed-up | MTP, 1 node | MTP, 4 nodes | Speed-up |
|---|---|---|---|---|---|---|
| 4k | 34.0 t/s | 60.2 t/s | 1.77x | 57.4 t/s | 99.4 t/s | 1.73x |
| 64k | 34.0 t/s | 55.8 t/s | 1.64x | 65.8 t/s | 108.9 t/s | 1.66x |
| 400k | 32.5 t/s | 54.9 t/s | 1.69x | 62.5 t/s | 101.4 t/s | 1.62x |

### Correctness

First build, four nodes with FP32 partials, MTP on (2.0 x 5, Q8 draft head), 5,120 WikiText-2 positions after an 8,192-token prefix, compared against the
one-node result:

| | BF16 partials (before) | FP32 partials (shipped) |
|---|---|---|
| Perplexity vs one node, plain decode | +0.71% | **+0.15%** |
| Perplexity vs one node, MTP | +0.55% | **+0.20%** |
| KL vs one node (gate: 1.5x the one-node noise floor = 0.0170) | 0.0144 | 0.0142 |
| Top-1 agreement | 96.4% | 96.9% |

With FP32 partials:
- Replicated activations were identical on all four ranks (0 hash mismatches).
- Session state matched after every keep and drop (0 state mismatches).
- The LM-head candidate merge was exact (32 / 32).
- Needle retrieval was 16/16 at 64k and beyond.
- A deliberately planted mirroring bug was caught by the state check (the negative control).
- Free-running MTP output diverges from plain greedy decode only at near-ties (the top-2 logit gap is below the
  one-node row-invariance bound).

**Single node is unchanged.** `strix_bench --hash-run 1` at world 1 prints the same
`logits d7fe7b8c12c333f6 drafts c52c86ccc5d0f307 state 1661670be8621634` with and without the TP changes.

## Known limits

- **Prefill scales only modestly.** At 64k a four-node prefill takes about 37 s, vs about 46 s on one node.
- **FP32 partials cost some decode speed** at long context (about 6% of MTP decode at 400k) in exchange for the
  perplexity fix above.
- **The slowest machine paces the group**, including where it loads its weights and n-gram rows from.
- **`strix_server` is single-node.** TP runs only through `tp_ar` so far. The prompt cache's export and import are
  not mirrored, so the backend refuses them under TP.
- World sizes are 2 and 4 only. Device auto-pick assumes the mlx5 driver; pass `--tp-dev` for others.

## Credits

The RDMA transport is written from scratch against libibverbs. The header-free rotating-window idea comes from gufo
(see the comment in `runtime/tp_comm.hpp`).
