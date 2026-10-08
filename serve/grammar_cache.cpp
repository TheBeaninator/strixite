#include "serve/grammar_cache.hpp"

#include "common/check.hpp"
#include "serve/json_schema.hpp"

namespace strix {

template <typename Compile>
std::shared_ptr<grammar::GrammarAutomaton> GrammarCache::get(const std::string &key, Compile compile) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        const auto found = by_key_.find(key);
        if (found != by_key_.end()) {
            entries_.splice(entries_.begin(), entries_, found->second);  // most recently used
            ++hits_;
            return found->second->second;
        }
    }
    // Compiled outside the lock (a large schema takes a while; a refused one throws) - two requests racing on the
    // same new schema both compile, and the second insert keeps the first entry.
    auto automaton = std::make_shared<grammar::GrammarAutomaton>(compile());
    std::lock_guard<std::mutex> lock(mu_);
    ++misses_;
    const auto found = by_key_.find(key);
    if (found != by_key_.end()) return found->second->second;
    entries_.emplace_front(key, automaton);
    by_key_[key] = entries_.begin();
    while (entries_.size() > kMaxEntries) {
        by_key_.erase(entries_.back().first);
        entries_.pop_back();
    }
    return automaton;
}

std::shared_ptr<grammar::GrammarAutomaton> GrammarCache::any_object() {
    return get("json_object", [] { return json_schema::compile_any_object(); });
}

std::shared_ptr<grammar::GrammarAutomaton> GrammarCache::schema(const json::Value &schema, const std::string &path) {
    STRIX_CHECK(!path.empty(), "GrammarCache::schema needs the schema's path for its error messages");
    return get("json_schema " + schema.dump(), [&] { return json_schema::compile(schema, path); });
}

size_t GrammarCache::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return entries_.size();
}

int64_t GrammarCache::hits() const {
    std::lock_guard<std::mutex> lock(mu_);
    return hits_;
}

int64_t GrammarCache::misses() const {
    std::lock_guard<std::mutex> lock(mu_);
    return misses_;
}

}  // namespace strix
