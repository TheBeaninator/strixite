#pragma once

// Why a request resumed short of what an earlier state shared with it. The model's recurrent (GDN) layers resume only
// on an exact prefix: one changed token ahead of a conversation - a client rebuilding its system prompt, pruning old
// tool output - costs a prefill of everything after it. On 2026-09-30 that was 8 cold prefills of 84k-180k tokens (an
// opencode plugin re-rendering memories into the system prompt), found only by comparing the saved system-prefix sizes
// across turns. This names the token and the message where the prompt left the nearest saved state, so the log says it
// at once.

#include <cstdint>
#include <string>
#include <vector>

namespace strix {

// Lost tokens below this aren't reported: a new turn routinely re-renders the previous answer (thinking dropped), a
// few hundred tokens behind the resume point.
constexpr int64_t kResumeLossReportTokens = 512;
// At or above this the line is a warning (yellow in journalctl): seconds of prefill.
constexpr int64_t kResumeLossWarnTokens = 8192;

// Where token `at` of prompt sits, by message (a message starts at an <|im_start|>, id im_start): "in message 1 of
// 184 (the system prompt, tokens 0-14544)", "in message 57 of 184 (tokens 51234-51899)"; "at the prompt's end" for
// at >= prompt size; "before the first message" for tokens ahead of the first <|im_start|>.
std::string describe_prompt_position(const std::vector<int32_t> &prompt, int64_t at, int32_t im_start);

// The log text for a request that resumed at `start` although `their` - an earlier state holding `their_n` tokens,
// named by `what` ("the live session", "the prompt cache's turn entry (...)") - shared its first `common` tokens with
// the prompt. Empty when fewer than kResumeLossReportTokens were lost. A resume keeps at least the prompt's last token
// to prefill, so at most prompt size - 1 tokens can be lost. Throws on inconsistent arguments.
std::string resume_loss_text(const std::vector<int32_t> &prompt, int64_t start, int64_t common, int64_t their_n,
                             const std::string &what, int32_t im_start);

}  // namespace strix
