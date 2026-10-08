#include "serve/token_mask.hpp"

#include "common/check.hpp"

#include <algorithm>
#include <limits>

namespace strix {

TokenMasker::TokenMasker(const Tokenizer &tokenizer, int32_t mask_rows) : tokenizer_(tokenizer), mask_rows_(mask_rows) {
    STRIX_CHECK(mask_rows >= tokenizer.size(), "TokenMasker: mask_rows ", mask_rows,
                " must cover the tokenizer's ", tokenizer.size(), " ids (the logits row width)");
    for (int32_t id = 0; id < tokenizer.size(); ++id) {
        if (tokenizer.is_added(id)) continue;
        const std::string &bytes = tokenizer.token_bytes(id);
        if (bytes.empty()) continue;
        STRIX_CHECK(bytes.size() < std::numeric_limits<uint16_t>::max(), "TokenMasker: token ", id, " has ",
                    bytes.size(), " bytes, more than a uint16 prefix length holds");
        sorted_ids_.push_back(id);
        longest_token_ = std::max(longest_token_, bytes.size());
    }
    STRIX_CHECK(!sorted_ids_.empty(), "TokenMasker: the tokenizer has no plain (non-added, non-empty) tokens");
    std::sort(sorted_ids_.begin(), sorted_ids_.end(), [&](int32_t left, int32_t right) {
        return tokenizer.token_bytes(left) < tokenizer.token_bytes(right);
    });
    common_prefix_.resize(sorted_ids_.size(), 0);
    for (size_t index = 1; index < sorted_ids_.size(); ++index) {
        const std::string &previous = tokenizer.token_bytes(sorted_ids_[index - 1]);
        const std::string &current = tokenizer.token_bytes(sorted_ids_[index]);
        size_t shared = 0;
        while (shared < previous.size() && shared < current.size() && previous[shared] == current[shared]) ++shared;
        common_prefix_[index] = static_cast<uint16_t>(shared);
    }
}

std::vector<uint32_t> TokenMasker::compute(grammar::GrammarAutomaton &automaton, int32_t state) const {
    STRIX_CHECK(state >= 0 && static_cast<size_t>(state) < automaton.state_count(), "TokenMasker::compute: state ",
                state, " outside the automaton's [0, ", automaton.state_count(), ")");
    std::vector<uint32_t> mask(static_cast<size_t>(mask_words()), 0u);
    // states_by_depth[d]: the automaton state after the first d bytes of the token being looked at; valid for
    // d <= valid_depth (the previous token's bytes that were walked and shared with this one).
    std::vector<int32_t> states_by_depth(longest_token_ + 1);
    states_by_depth[0] = state;
    size_t valid_depth = 0;
    const size_t count = sorted_ids_.size();
    for (size_t index = 0; index < count; ++index) {
        const std::string &bytes = tokenizer_.token_bytes(sorted_ids_[index]);
        valid_depth = std::min<size_t>(valid_depth, common_prefix_[index]);
        size_t depth = valid_depth;
        for (; depth < bytes.size(); ++depth) {
            const int32_t next = automaton.next(states_by_depth[depth], static_cast<uint8_t>(bytes[depth]));
            if (next == grammar::GrammarAutomaton::kDead) break;
            states_by_depth[depth + 1] = next;
        }
        if (depth == bytes.size()) {
            const int32_t id = sorted_ids_[index];
            mask[static_cast<size_t>(id) >> 5] |= 1u << (id & 31);
            valid_depth = depth;
            continue;
        }
        // Byte `depth` was refused: every following token with the same first depth + 1 bytes is refused too.
        valid_depth = depth;
        while (index + 1 < count && common_prefix_[index + 1] > depth) ++index;
    }
    return mask;
}

const std::vector<uint32_t> &TokenMasker::mask(grammar::GrammarAutomaton &automaton, int32_t state) const {
    if (const std::vector<uint32_t> *cached = automaton.cached_mask(state, this)) return *cached;
    return automaton.store_mask(state, this, compute(automaton, state));
}

}  // namespace strix
