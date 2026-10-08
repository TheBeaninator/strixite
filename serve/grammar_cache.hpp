#pragma once

// Compiled response formats kept across requests (cached by schema - an agent's title schema, for example, repeats
// every session). An entry is a GrammarAutomaton: the compiled grammar
// with its states and token masks, which fill in as generations use them, so a repeated schema costs neither the
// compile nor the masks again. Keyed by the schema's compact JSON; the least recently used entry is dropped past
// kMaxEntries (a request still running keeps its automaton alive through its shared_ptr).
//
// get() is called from HTTP threads (mutex inside); the automaton it returns is then used by the engine thread
// alone (GrammarAutomaton isn't thread-safe; the engine runs one request at a time).

#include "serve/grammar.hpp"
#include "serve/json.hpp"

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace strix {

class GrammarCache {
public:
    // Up to 62 MiB of cached masks an automaton (GrammarAutomaton::kMaxCachedMasks), so the bound keeps the worst
    // case near 1 GiB; real schemas use a few masks (Hermes' title: 6, ~190 KB).
    static constexpr size_t kMaxEntries = 16;

    // response_format json_object.
    std::shared_ptr<grammar::GrammarAutomaton> any_object();
    // response_format json_schema: `schema` named `path` in errors; throws strix::Error naming the keyword / path
    // the compiler refuses (serve/json_schema).
    std::shared_ptr<grammar::GrammarAutomaton> schema(const json::Value &schema, const std::string &path);

    size_t size() const;
    int64_t hits() const;
    int64_t misses() const;

private:
    template <typename Compile>
    std::shared_ptr<grammar::GrammarAutomaton> get(const std::string &key, Compile compile);

    mutable std::mutex mu_;
    std::list<std::pair<std::string, std::shared_ptr<grammar::GrammarAutomaton>>> entries_;  // most recent first
    std::unordered_map<std::string, decltype(entries_)::iterator> by_key_;
    int64_t hits_ = 0, misses_ = 0;
};

}  // namespace strix
