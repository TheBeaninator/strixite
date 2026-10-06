#pragma once

// Teacher-forced replay of a captured request (serve/capture.hpp; tools/bench_replay): the backend is fed the
// capture's own tokens - its context, then its generated tokens - while MTP drafts and verifies exactly as the engine
// does (serve/engine.cpp), except that a draft counts as accepted when it equals the *captured* next token. Every
// build and every drafting policy then decodes the same real text, token for token, so a decode A/B on agent traffic
// is like for like (engine changes that alter rounding used to change the token streams, and with them the drafts:
// 9 of 10 streams differed between two builds, 2026-09-28).
//
// Acceptance here is "the draft matches what the model generated in real use", not "matches what this build would
// sample" - equal for the build that served the capture at temperature 0, a close proxy otherwise, and the same for
// both sides of an A/B.

#include <cstdint>
#include <functional>

#include "serve/capture.hpp"
#include "serve/engine.hpp"

namespace strix {

struct ReplayOptions {
    int64_t max_context = 16384;  // the capture's last this-many prompt tokens are the context (its whole prompt if fewer)
    int64_t max_gen = 512;        // generated tokens replayed at most (0 = all)
    int64_t mtp_draft = 4;        // drafts per verify at most (0 = MTP off: plain forwards only)
    float mtp_margin = 2.0f;      // the head's top-1 margin a draft needs
    bool mtp_reject_forward = false;  // pre-PF-1 path: forward a rejected position's token alone (A/B only)
    // Called between the prefill and the decode, if set (bench_replay --ple-after-prefill: empty the PLE row cache, as
    // a turn resumed from the prompt cache after a restart finds it - the prefill that would have filled it is
    // skipped).
    std::function<void()> after_prefill;
    // The host's part of a decode step: off - the logits come back but nothing samples them (the old replay); full /
    // candidates - every row the engine would sample is sampled (Sampler with `sampling`, ids [0, n_valid)), from
    // full rows or from each row's top 20 reduced on the GPU (LmBackend::forward_rows; needs
    // sampling.takes_candidates()). The sampled ids don't steer the replay (it stays teacher-forced); their hash
    // shows that full and candidates sampled the same.
    enum class Sample { Off, Full, Candidates } sample = Sample::Off;
    SamplingParams sampling;
    int64_t n_valid = 0;  // the tokenizer's size; needed unless sample is Off
};

struct ReplayResult {
    int64_t context_n = 0, gen_n = 0;        // tokens prefilled / replayed
    int64_t forwards = 0, drafted = 0, accepted = 0, rollbacks = 0;
    double prefill_ms = 0, decode_ms = 0;    // wall time of each phase
    // The decode phase's PLE rows (BackendStats deltas): read from the table file (gathers + prefetches), forwards that
    // waited for theirs and the seconds waited (the GPU idles meanwhile).
    int64_t ple_rows_from_file = 0, ple_waits = 0;
    double ple_wait_s = 0;
    int64_t sampled = 0;       // rows sampled (sample != Off)
    uint64_t sample_hash = 0;  // of the sampled ids in order
};

// Throws (with the capture id and the step) on anything inconsistent: an empty capture, a backend without MTP when
// drafts are asked for, a context longer than the backend holds.
ReplayResult replay_teacher_forced(LmBackend &be, const CaptureRecord &rec, const ReplayOptions &opt);

}  // namespace strix
