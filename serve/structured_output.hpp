#pragma once

// One generation's structured output (response_format): which tokens each logits row may
// produce. Thinking stays as the request asks; the grammar applies from the first token after </think> (from the
// first token when the prompt has thinking off). So a generation is in one of two phases:
//   - reasoning: any token, except the end of the turn (the answer hasn't started - it must) ;
//   - content: the grammar's tokens (serve/token_mask), plus the end of the turn only where the grammar is complete
//     (accepting); where it can't continue, the end of the turn is the only token.
// The engine feeds every generated or fed token (feed); a forward's rows are masked by the phase and grammar state
// each row follows (masks_for: row r after the tokens fed so far plus the first r of `pending`, an MTP verify's
// drafts). A draft the grammar would refuse is never verified (Cursor::allows).

#include "serve/grammar.hpp"
#include "serve/sampler.hpp"
#include "serve/token_mask.hpp"
#include "serve/tokenizer.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace strix {

class StructuredOutput {
public:
    // automaton: the request's compiled grammar (shared across requests with the same schema; used on the engine
    // thread only). masker: the engine's (its mask_rows() = the logits row). think_end: the </think> token;
    // end_ids: the end-of-turn tokens (allowed where the grammar is complete). in_content: the prompt has thinking
    // off - the grammar applies from the first token.
    StructuredOutput(std::shared_ptr<grammar::GrammarAutomaton> automaton, const TokenMasker &masker,
                     const Tokenizer &tokenizer, int32_t think_end, std::vector<int32_t> end_ids, bool in_content);

    // Where a generation stands: the phase and, in content, the grammar state.
    struct Cursor {
        bool content = false;
        int32_t state = 0;
    };
    Cursor cursor() const { return cursor_; }
    // Whether `token` may come next from `at` (the end of the turn included).
    bool allows(const Cursor &at, int32_t token) const;
    // `token` appended to `at`; throws if the grammar refuses it (a token the mask didn't allow - a bug).
    Cursor after(const Cursor &at, int32_t token) const;
    // A generated or fed token: the generation's cursor moves on.
    void feed(int32_t token) { cursor_ = after(cursor_, token); }

    // Masks for a forward's rows: row r follows the tokens fed so far plus pending[0, r) - so 1 + pending.size()
    // rows (a plain step or a prefill: pending empty, one row; a verify: its drafts).
    LogitMasks masks_for(const std::vector<int32_t> &pending) const;

    bool in_content() const { return cursor_.content; }
    // Content phase and the grammar complete: the output so far is a whole document.
    bool complete() const;

private:
    void row_mask(const Cursor &at, uint32_t *out) const;

    std::shared_ptr<grammar::GrammarAutomaton> automaton_;
    const TokenMasker &masker_;
    const Tokenizer &tokenizer_;
    int32_t think_end_;
    std::vector<int32_t> end_ids_;
    Cursor cursor_;
};

}  // namespace strix
