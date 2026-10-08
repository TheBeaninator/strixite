#pragma once

// Which tokens a grammar allows next - the mask structured output applies to the logits (1 bit a token, applied in logits_topk's first level before the top-k).
//
// A token is allowed in a state when the grammar accepts all of its bytes from there (serve/grammar's
// GrammarAutomaton). Checking 248k tokens one by one would repeat the work their shared prefixes have in common,
// so the vocabulary is kept sorted by bytes, with each token's common-prefix length to the one before it: the walk
// reuses the automaton states of the shared prefix, and when a byte is refused every following token that starts
// with the same refused prefix is skipped without looking at it. Masks are cached per automaton state.
//
// Added (special) tokens are never allowed by the mask: they aren't grammar text. The caller decides about the end
// of the turn separately (allowed only where the automaton is accepting). Rows past the tokenizer's size (the
// logits row's padding) are never allowed.

#include "serve/grammar.hpp"
#include "serve/tokenizer.hpp"

#include <cstdint>
#include <vector>

namespace strix {

class TokenMasker {
public:
    // mask_rows: the logits row's width (>= tokenizer.size(); the target pads 248,077 ids to 248,320).
    TokenMasker(const Tokenizer &tokenizer, int32_t mask_rows);

    int32_t mask_rows() const { return mask_rows_; }
    int32_t mask_words() const { return (mask_rows_ + 31) / 32; }

    // The allowed tokens in `state`: bit (id % 32) of word (id / 32) set = allowed. Cached in the automaton; the
    // reference stays valid until the automaton's next store_mask (the next uncached mask).
    const std::vector<uint32_t> &mask(grammar::GrammarAutomaton &automaton, int32_t state) const;

    // The same, computed without the cache (for tests and benches).
    std::vector<uint32_t> compute(grammar::GrammarAutomaton &automaton, int32_t state) const;

    static bool allowed(const std::vector<uint32_t> &mask, int32_t id) {
        return (mask[static_cast<size_t>(id) >> 5] >> (id & 31)) & 1u;
    }

private:
    const Tokenizer &tokenizer_;
    int32_t mask_rows_;
    std::vector<int32_t> sorted_ids_;         // grammar-eligible tokens (not added, not empty), sorted by bytes
    std::vector<uint16_t> common_prefix_;     // [i]: bytes token i shares with token i - 1 (0 for the first)
    size_t longest_token_ = 0;
};

}  // namespace strix
