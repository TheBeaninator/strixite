#include "serve/structured_output.hpp"

#include "common/check.hpp"

#include <algorithm>
#include <cstring>

namespace strix {

StructuredOutput::StructuredOutput(std::shared_ptr<grammar::GrammarAutomaton> automaton, const TokenMasker &masker,
                                   const Tokenizer &tokenizer, int32_t think_end, std::vector<int32_t> end_ids,
                                   bool in_content)
    : automaton_(std::move(automaton)), masker_(masker), tokenizer_(tokenizer), think_end_(think_end),
      end_ids_(std::move(end_ids)) {
    STRIX_CHECK(automaton_ != nullptr, "StructuredOutput: no grammar automaton (null)");
    STRIX_CHECK(!end_ids_.empty(), "StructuredOutput: no end-of-turn tokens given");
    for (int32_t id : end_ids_)
        STRIX_CHECK(id >= 0 && id < tokenizer.size() && id < masker.mask_rows(), "StructuredOutput: end-of-turn token ",
                    id, " outside the tokenizer's ", tokenizer.size(), " ids");
    STRIX_CHECK(think_end >= 0 && think_end < tokenizer.size(), "StructuredOutput: </think> token ", think_end,
                " outside the tokenizer's ", tokenizer.size(), " ids");
    cursor_ = Cursor{in_content, automaton_->initial()};
}

bool StructuredOutput::allows(const Cursor &at, int32_t token) const {
    const bool is_end = std::find(end_ids_.begin(), end_ids_.end(), token) != end_ids_.end();
    if (!at.content) return !is_end;  // reasoning: anything but the end of the turn
    if (is_end) return automaton_->accepting(at.state);
    if (token < 0 || token >= tokenizer_.size() || tokenizer_.is_added(token)) return false;
    return automaton_->next(at.state, tokenizer_.token_bytes(token)) != grammar::GrammarAutomaton::kDead;
}

StructuredOutput::Cursor StructuredOutput::after(const Cursor &at, int32_t token) const {
    if (!at.content) {
        if (token == think_end_) return Cursor{true, automaton_->initial()};
        return at;
    }
    STRIX_CHECK(token >= 0 && token < tokenizer_.size() && !tokenizer_.is_added(token), "StructuredOutput: token ",
                token, " in the answer is not grammar text (a special token the mask should have refused)");
    const int32_t next = automaton_->next(at.state, tokenizer_.token_bytes(token));
    STRIX_CHECK(next != grammar::GrammarAutomaton::kDead, "StructuredOutput: token ", token, " ('",
                tokenizer_.token_bytes(token), "') is not allowed by the response format here (",
                automaton_->state_count(), " grammar states) - the mask should have refused it");
    return Cursor{true, next};
}

bool StructuredOutput::complete() const { return cursor_.content && automaton_->accepting(cursor_.state); }

void StructuredOutput::row_mask(const Cursor &at, uint32_t *out) const {
    const int64_t words = masker_.mask_words();
    if (!at.content) {
        std::fill(out, out + words, 0xFFFFFFFFu);
        for (int32_t id : end_ids_) out[id >> 5] &= ~(1u << (id & 31));
        return;
    }
    const std::vector<uint32_t> &allowed = masker_.mask(*automaton_, at.state);
    std::memcpy(out, allowed.data(), (size_t)words * 4);
    if (automaton_->accepting(at.state))
        for (int32_t id : end_ids_) out[id >> 5] |= 1u << (id & 31);
}

LogitMasks StructuredOutput::masks_for(const std::vector<int32_t> &pending) const {
    // Every row is masked, reasoning ones too (they refuse the end of the turn: the answer must still come).
    std::vector<Cursor> rows{cursor_};
    for (int32_t token : pending) rows.push_back(after(rows.back(), token));
    LogitMasks masks;
    masks.rows = (int64_t)rows.size(), masks.words = masker_.mask_words();
    masks.bits.resize((size_t)(masks.rows * masks.words));
    for (size_t r = 0; r < rows.size(); ++r) row_mask(rows[r], masks.bits.data() + r * (size_t)masks.words);
    return masks;
}

}  // namespace strix
