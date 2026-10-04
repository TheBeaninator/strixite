#include "serve/resume_loss.hpp"

#include <algorithm>

#include "common/check.hpp"

namespace strix {

std::string describe_prompt_position(const std::vector<int32_t> &prompt, int64_t at, int32_t im_start) {
    STRIX_CHECK(at >= 0, "describe_prompt_position: token ", at, " (want >= 0)");
    const int64_t P = (int64_t)prompt.size();
    if (at >= P) return "at the prompt's end";
    std::vector<int64_t> starts;
    for (int64_t k = 0; k < P; ++k)
        if (prompt[(size_t)k] == im_start) starts.push_back(k);
    if (starts.empty() || at < starts[0]) return "before the first message";
    const size_t m = (size_t)(std::upper_bound(starts.begin(), starts.end(), at) - starts.begin()) - 1;
    const int64_t first = starts[m], last = (m + 1 < starts.size() ? starts[m + 1] : P) - 1;
    return "in message " + std::to_string(m + 1) + " of " + std::to_string(starts.size()) + " (" +
           (m == 0 ? "the system prompt, " : "") + "tokens " + std::to_string(first) + "-" + std::to_string(last) + ")";
}

std::string resume_loss_text(const std::vector<int32_t> &prompt, int64_t start, int64_t common, int64_t their_n,
                             const std::string &what, int32_t im_start) {
    const int64_t P = (int64_t)prompt.size();
    STRIX_CHECK(P >= 1 && start >= 0 && start < P, "resume_loss_text: resumed at ", start, " of a ", P, "-token prompt");
    STRIX_CHECK(common >= 0 && common <= std::min(P, their_n), "resume_loss_text: ", common, " tokens shared by a ",
                P, "-token prompt and a ", their_n, "-token state (", what, ")");
    const int64_t lost = std::min(common, P - 1) - start;
    if (lost < kResumeLossReportTokens) return "";
    const std::string head = "resume lost " + std::to_string(lost) + " tokens: ";
    if (common == P)  // the prompt is a prefix of the state: nothing differs, the state is just further along
        return head + "the prompt ends inside " + what + " (" + std::to_string(their_n) +
               " tokens) - a recurrent state can't step back";
    if (common == their_n)  // the state is a full prefix of the prompt, yet it wasn't resumed from
        return head + what + " covers the first " + std::to_string(their_n) + " tokens but couldn't be resumed from";
    return head + "the prompt differs from " + what + " (" + std::to_string(their_n) + " tokens) at token " +
           std::to_string(common) + ", " + describe_prompt_position(prompt, common, im_start);
}

}  // namespace strix
