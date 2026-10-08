#include "serve/grammar.hpp"

#include "common/check.hpp"

#include <algorithm>
#include <functional>

namespace strix::grammar {

// ---- Grammar ----

int Grammar::add_rule(const std::string &name) {
    STRIX_CHECK(!name.empty(), "a grammar rule needs a name (it labels errors)");
    STRIX_CHECK(find_rule(name) < 0, "grammar rule '", name, "' already exists (rule names must be unique)");
    rules_.push_back(Rule{name, {}});
    validated_ = false;
    return static_cast<int>(rules_.size()) - 1;
}

void Grammar::add_alternative(int rule_index, Alternative alternative) {
    STRIX_CHECK(rule_index >= 0 && rule_index < static_cast<int>(rules_.size()), "rule index ", rule_index,
                " outside [0, ", rules_.size(), ")");
    rules_[rule_index].alternatives.push_back(std::move(alternative));
    validated_ = false;
}

Alternative Grammar::literal(std::string_view bytes) {
    Alternative elements;
    elements.reserve(bytes.size());
    for (char character : bytes) elements.push_back(Element::byte(static_cast<uint8_t>(character)));
    return elements;
}

void Grammar::set_root(int rule_index) {
    STRIX_CHECK(rule_index >= 0 && rule_index < static_cast<int>(rules_.size()), "root rule index ", rule_index,
                " outside [0, ", rules_.size(), ")");
    root_ = rule_index;
    validated_ = false;
}

int Grammar::find_rule(const std::string &name) const {
    for (size_t index = 0; index < rules_.size(); ++index)
        if (rules_[index].name == name) return static_cast<int>(index);
    return -1;
}

void Grammar::validate() const {
    STRIX_CHECK(root_ >= 0, "the grammar has no root rule (set_root was never called; ", rules_.size(), " rules)");
    const int rule_count = static_cast<int>(rules_.size());
    for (const Rule &rule : rules_) {
        STRIX_CHECK(!rule.alternatives.empty(), "grammar rule '", rule.name,
                    "' has no alternatives (it could never match; an empty alternative matches the empty string)");
        for (size_t alternative = 0; alternative < rule.alternatives.size(); ++alternative) {
            for (size_t position = 0; position < rule.alternatives[alternative].size(); ++position) {
                const Element &element = rule.alternatives[alternative][position];
                const std::string where = cat("rule '", rule.name, "' alternative ", alternative, " element ", position);
                if (element.kind == Element::Kind::Rule) {
                    STRIX_CHECK(element.rule >= 0 && element.rule < rule_count, where, " references rule ",
                                element.rule, ", outside [0, ", rule_count, ")");
                } else {
                    STRIX_CHECK(!element.ranges.empty(), where, " is a byte set with no ranges (matches nothing)");
                    for (const ByteRange &range : element.ranges)
                        STRIX_CHECK(range.low <= range.high, where, " has a byte range with low ",
                                    static_cast<int>(range.low), " > high ", static_cast<int>(range.high));
                }
            }
        }
    }

    // Nullable rules (can match the empty string), to a fixpoint.
    std::vector<bool> nullable(rule_count, false);
    for (bool changed = true; changed;) {
        changed = false;
        for (int index = 0; index < rule_count; ++index) {
            if (nullable[index]) continue;
            for (const Alternative &alternative : rules_[index].alternatives) {
                const bool all_nullable = std::all_of(alternative.begin(), alternative.end(), [&](const Element &e) {
                    return e.kind == Element::Kind::Rule && nullable[e.rule];
                });
                if (all_nullable) {
                    nullable[index] = true;
                    changed = true;
                    break;
                }
            }
        }
    }
    // Left edges: rule -> a rule it can reach before consuming a byte. A cycle among them is left recursion.
    std::vector<std::vector<int>> left_edges(rule_count);
    for (int index = 0; index < rule_count; ++index) {
        for (const Alternative &alternative : rules_[index].alternatives) {
            for (const Element &element : alternative) {
                if (element.kind != Element::Kind::Rule) break;
                left_edges[index].push_back(element.rule);
                if (!nullable[element.rule]) break;
            }
        }
    }
    enum Mark : uint8_t { Unvisited, OnPath, Done };
    std::vector<uint8_t> marks(rule_count, Unvisited);
    std::vector<int> path;
    std::function<void(int)> visit = [&](int index) {
        marks[index] = OnPath;
        path.push_back(index);
        for (int next : left_edges[index]) {
            if (marks[next] == OnPath) {
                std::string cycle;
                for (auto it = std::find(path.begin(), path.end(), next); it != path.end(); ++it)
                    cycle += rules_[*it].name + " -> ";
                fail_at(__FILE__, __LINE__, __func__,
                        cat("grammar rule '", rules_[next].name, "' is left-recursive (", cycle, rules_[next].name,
                            " without consuming a byte) - the matcher would expand it forever"));
            }
            if (marks[next] == Unvisited) visit(next);
        }
        path.pop_back();
        marks[index] = Done;
    };
    for (int index = 0; index < rule_count; ++index)
        if (marks[index] == Unvisited) visit(index);
    validated_ = true;
}

namespace {
std::string byte_text(uint8_t value) {
    if (value >= 0x20 && value < 0x7f && value != '"' && value != '\\' && value != ']' && value != '-')
        return std::string(1, static_cast<char>(value));
    static const char kHex[] = "0123456789abcdef";
    return std::string("\\x") + kHex[value >> 4] + kHex[value & 15];
}
}  // namespace

std::string Grammar::dump() const {
    std::string out;
    for (size_t index = 0; index < rules_.size(); ++index) {
        out += rules_[index].name + (static_cast<int>(index) == root_ ? " (root)" : "") + " :=";
        for (size_t alternative = 0; alternative < rules_[index].alternatives.size(); ++alternative) {
            if (alternative) out += " |";
            const Alternative &elements = rules_[index].alternatives[alternative];
            if (elements.empty()) out += " \"\"";
            std::string pending_literal;  // single-byte elements run together as one quoted string
            auto flush = [&] {
                if (!pending_literal.empty()) out += " \"" + pending_literal + "\"";
                pending_literal.clear();
            };
            for (const Element &element : elements) {
                if (element.kind == Element::Kind::Rule) {
                    flush();
                    out += " " + (element.rule >= 0 && element.rule < static_cast<int>(rules_.size())
                                      ? rules_[element.rule].name
                                      : cat("<bad rule ", element.rule, ">"));
                } else if (element.ranges.size() == 1 && element.ranges[0].low == element.ranges[0].high) {
                    pending_literal += byte_text(element.ranges[0].low);
                } else {
                    flush();
                    out += " [";
                    for (const ByteRange &range : element.ranges) {
                        out += byte_text(range.low);
                        if (range.high != range.low) out += "-" + byte_text(range.high);
                    }
                    out += "]";
                }
            }
            flush();
        }
        out += "\n";
    }
    return out;
}

// ---- GrammarMatcher ----

GrammarMatcher::GrammarMatcher(std::shared_ptr<const Grammar> grammar) : grammar_(std::move(grammar)) {
    STRIX_CHECK(grammar_ != nullptr, "GrammarMatcher needs a grammar (got a null pointer)");
    STRIX_CHECK(grammar_->validated(), "GrammarMatcher needs a validated grammar - call Grammar::validate() first "
                "(it refuses left recursion, which would make the matcher loop)");
    const int root = grammar_->root();
    const auto &alternatives = grammar_->rules()[root].alternatives;
    for (size_t alternative = 0; alternative < alternatives.size(); ++alternative)
        expand(Stack{Frame{root, static_cast<int32_t>(alternative), 0}}, parses_);
    normalize(parses_);
}

const Element &GrammarMatcher::top_element(const Stack &stack) const {
    const Frame &top = stack.back();
    return grammar_->rules()[top.rule].alternatives[top.alternative][top.position];
}

void GrammarMatcher::expand(Stack stack, std::vector<Stack> &out) const {
    const auto &rules = grammar_->rules();
    while (true) {
        if (stack.empty()) {
            out.push_back(std::move(stack));
            return;
        }
        Frame &top = stack.back();
        const Alternative &elements = rules[top.rule].alternatives[top.alternative];
        if (top.position == static_cast<int32_t>(elements.size())) {
            stack.pop_back();  // this alternative is done: return to the caller's frame
            continue;
        }
        const Element &element = elements[top.position];
        if (element.kind == Element::Kind::Bytes) {
            out.push_back(std::move(stack));
            return;
        }
        // A rule reference: step the caller past it, then descend. A reference in last position replaces the
        // caller's frame (tail call), so right-recursive repetition keeps the stack flat.
        const int callee = element.rule;
        ++top.position;
        if (top.position == static_cast<int32_t>(elements.size())) stack.pop_back();
        if (static_cast<int>(stack.size()) >= kMaxStackDepth) return;  // too deep: this parse is dropped
        const auto &callee_alternatives = rules[callee].alternatives;
        for (size_t alternative = 0; alternative + 1 < callee_alternatives.size(); ++alternative) {
            Stack branch = stack;
            branch.push_back(Frame{callee, static_cast<int32_t>(alternative), 0});
            expand(std::move(branch), out);
        }
        stack.push_back(Frame{callee, static_cast<int32_t>(callee_alternatives.size() - 1), 0});
    }
}

void GrammarMatcher::normalize(std::vector<Stack> &parses) const {
    std::sort(parses.begin(), parses.end());
    parses.erase(std::unique(parses.begin(), parses.end()), parses.end());
    STRIX_CHECK(parses.size() <= kMaxParses, "grammar matcher: ", parses.size(), " live parses after ",
                bytes_consumed_, " bytes, more than the limit ", kMaxParses,
                " - the grammar is too ambiguous (a schema compiler bug; root rule '",
                grammar_->rules()[grammar_->root()].name, "')");
}

bool GrammarMatcher::advance(uint8_t byte) {
    std::vector<Stack> next;
    for (const Stack &stack : parses_) {
        if (stack.empty()) continue;  // a complete parse takes no more bytes
        const Element &element = top_element(stack);
        const bool allowed = std::any_of(element.ranges.begin(), element.ranges.end(),
                                         [&](const ByteRange &range) { return byte >= range.low && byte <= range.high; });
        if (!allowed) continue;
        Stack stepped = stack;
        ++stepped.back().position;
        expand(std::move(stepped), next);
    }
    if (next.empty()) return false;
    ++bytes_consumed_;
    normalize(next);
    parses_ = std::move(next);
    return true;
}

bool GrammarMatcher::advance(std::string_view bytes) {
    GrammarMatcher trial = *this;
    for (char character : bytes)
        if (!trial.advance(static_cast<uint8_t>(character))) return false;
    *this = std::move(trial);
    return true;
}

bool GrammarMatcher::allows(std::string_view bytes) const {
    GrammarMatcher trial = *this;
    return trial.advance(bytes);
}

bool GrammarMatcher::accepting() const {
    return std::any_of(parses_.begin(), parses_.end(), [](const Stack &stack) { return stack.empty(); });
}

bool GrammarMatcher::can_continue() const {
    return std::any_of(parses_.begin(), parses_.end(), [](const Stack &stack) { return !stack.empty(); });
}

std::vector<uint8_t> GrammarMatcher::allowed_bytes() const {
    std::vector<bool> seen(256, false);
    for (const Stack &stack : parses_) {
        if (stack.empty()) continue;
        for (const ByteRange &range : top_element(stack).ranges)
            for (int value = range.low; value <= range.high; ++value) seen[value] = true;
    }
    std::vector<uint8_t> out;
    for (int value = 0; value < 256; ++value)
        if (seen[value]) out.push_back(static_cast<uint8_t>(value));
    return out;
}

// ---- GrammarAutomaton ----

namespace {
uint64_t hash_parses(const std::vector<GrammarMatcher::Stack> &parses) {
    uint64_t hash = 1469598103934665603ull;  // FNV-1a over the frames, with a separator per stack
    auto mix = [&](uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ull;
    };
    for (const GrammarMatcher::Stack &stack : parses) {
        for (const GrammarMatcher::Frame &frame : stack)
            mix((static_cast<uint64_t>(static_cast<uint32_t>(frame.rule)) << 32) ^
                (static_cast<uint64_t>(static_cast<uint32_t>(frame.alternative)) << 16) ^
                static_cast<uint32_t>(frame.position));
        mix(0x9e3779b97f4a7c15ull);
    }
    return hash;
}
}  // namespace

GrammarAutomaton::GrammarAutomaton(std::shared_ptr<const Grammar> grammar, bool colliding_hash_for_tests)
    : grammar_(std::move(grammar)), colliding_hash_(colliding_hash_for_tests) {
    STRIX_CHECK(grammar_ != nullptr, "GrammarAutomaton needs a grammar (got a null pointer)");
    intern(GrammarMatcher(grammar_));  // state 0: the start
}

void GrammarAutomaton::check_state(int32_t state, const char *what) const {
    STRIX_CHECK(state >= 0 && static_cast<size_t>(state) < states_.size(), what, ": state ", state, " outside [0, ",
                states_.size(), ") (kDead is not a state; check next()'s result first)");
}

int32_t GrammarAutomaton::intern(GrammarMatcher matcher) {
    const uint64_t hash = colliding_hash_ ? 0 : hash_parses(matcher.parses());
    std::vector<int32_t> &bucket = by_hash_[hash];
    for (int32_t existing : bucket)
        if (states_[existing].parses() == matcher.parses()) return existing;
    STRIX_CHECK(states_.size() < kMaxStates, "grammar automaton: more than ", kMaxStates, " states for root rule '",
                grammar_->rules()[grammar_->root()].name, "' - the grammar is too large to serve");
    const int32_t id = static_cast<int32_t>(states_.size());
    accepting_.push_back(matcher.accepting());
    can_continue_.push_back(matcher.can_continue());
    states_.push_back(std::move(matcher));
    bucket.push_back(id);
    return id;
}

int32_t GrammarAutomaton::next(int32_t state, uint8_t byte) {
    const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(state)) << 8) | byte;
    const auto found = transitions_.find(key);
    if (found != transitions_.end()) return found->second;
    check_state(state, "GrammarAutomaton::next");
    GrammarMatcher matcher = states_[state];
    const int32_t result = matcher.advance(byte) ? intern(std::move(matcher)) : kDead;
    transitions_.emplace(key, result);
    return result;
}

int32_t GrammarAutomaton::next(int32_t state, std::string_view bytes) {
    for (char character : bytes) {
        state = next(state, static_cast<uint8_t>(character));
        if (state == kDead) return kDead;
    }
    return state;
}

bool GrammarAutomaton::accepting(int32_t state) const {
    check_state(state, "GrammarAutomaton::accepting");
    return accepting_[state] != 0;
}

bool GrammarAutomaton::can_continue(int32_t state) const {
    check_state(state, "GrammarAutomaton::can_continue");
    return can_continue_[state] != 0;
}

const std::vector<uint32_t> *GrammarAutomaton::cached_mask(int32_t state, const void *owner) const {
    STRIX_CHECK(mask_owner_ == nullptr || mask_owner_ == owner, "grammar automaton: masks were cached for another "
                "vocabulary (one automaton serves one tokenizer)");
    const auto found = masks_.find(state);
    return found == masks_.end() ? nullptr : &found->second;
}

const std::vector<uint32_t> &GrammarAutomaton::store_mask(int32_t state, const void *owner, std::vector<uint32_t> mask) {
    check_state(state, "GrammarAutomaton::store_mask");
    STRIX_CHECK(owner != nullptr, "grammar automaton: a cached mask needs its vocabulary's identity (got null)");
    STRIX_CHECK(mask_owner_ == nullptr || mask_owner_ == owner, "grammar automaton: masks were cached for another "
                "vocabulary (one automaton serves one tokenizer)");
    mask_owner_ = owner;
    if (masks_.size() >= kMaxCachedMasks) masks_.clear();
    return masks_[state] = std::move(mask);
}

}  // namespace strix::grammar
