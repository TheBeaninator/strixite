#pragma once

// Token-level capture of finished requests (strix_server --capture-dir): the
// prompt and generated token ids plus what MTP did at every decode step, so drafting policies - prompt-lookup drafts
// where the MTP head holds back, per-turn-type margins - can be simulated offline on real agent traffic before any of
// them is built (measure on logged requests first, build second). Opt-in; nothing is
// captured without the flag. The files hold the user's prompts: they stay on the serving machine.
//
// One file per request, `<unix ms>-req<id>.cap`:
//   line 1: a JSON header {"v":1,"id":..,"prompt_n":..,"tokens_n":..,"steps_n":..,"tools":..,"one_shot":..,
//           "user_turn":..,"finish":"..","mtp_draft":..,"mtp_margin":..}
//   then tokens_n little-endian int32: the prompt, then every generated token that entered the sequence (the final
//   end-of-turn token is not one), then steps_n records of 3 bytes {drafted, accepted, emitted}:
//     drafted  - MTP drafts verified in this step (0 = a plain forward; kThinkingStopStep = the engine fed text
//                instead of sampling: the thinking-stop text, or a thinking nudge - serve/think_nudge.hpp),
//     accepted - drafts accepted (<= drafted),
//     emitted  - tokens this step added to the sequence (sum over steps = tokens_n - prompt_n).

#include <cstdint>
#include <string>
#include <vector>

namespace strix {

constexpr uint8_t kThinkingStopStep = 255;

struct CaptureStep {
    uint8_t drafted = 0, accepted = 0, emitted = 0;
};

struct CaptureRecord {
    int64_t id = 0;
    int64_t prompt_n = 0;
    std::vector<int32_t> tokens;  // prompt then generated
    std::vector<CaptureStep> steps;
    int64_t tools = 0;
    bool one_shot = false, user_turn = false;
    std::string finish;
    int64_t mtp_draft = 0;
    double mtp_margin = 0;
};

// Checks the record (sizes, step sums) and writes it into dir (which must exist) as ".partial", then renames; returns
// the file's path. Throws with the path and the reason on any failure.
std::string write_capture(const std::string &dir, const CaptureRecord &rec, int64_t unix_ms);
// Reads a file written by write_capture; throws on anything malformed.
CaptureRecord read_capture(const std::string &path);

}  // namespace strix
