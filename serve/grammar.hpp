#pragma once

// Byte-level context-free grammars and their matcher - the core of structured output (response_format).
// serve/json_schema compiles a request's JSON schema into a Grammar; the engine
// keeps one GrammarMatcher per generation and asks it which bytes (later: which tokens) may come next.
//
// A grammar is a list of rules; each rule is a list of alternatives; each alternative is a sequence of elements.
// An element is either one byte drawn from a set of byte ranges, or a reference to another rule. Working on bytes,
// not characters, lets one grammar judge any token's bytes, including a token that ends inside a UTF-8 character.
//
// The matcher is a nondeterministic pushdown automaton: its state is the set of every parse still possible after the
// bytes seen so far, each parse a stack of frames (rule, alternative, position in it). A byte keeps the parses whose
// next byte element allows it. The parses are expanded until each one's top frame waits on a byte element (or the
// parse is complete), so checking a byte is a range lookup per parse. Two rules keep this finite:
//   - no left recursion (a rule reaching itself before any byte is consumed) - Grammar::validate refuses it;
//   - a reference in the last position of an alternative replaces its frame instead of stacking on it (a tail
//     call), so repetition written as right recursion (`items := item items | empty`) doesn't grow the stack.
// Nesting that is real (an object inside an object) does grow it; kMaxStackDepth bounds it, and a parse that would
// go deeper is dropped - output from a recursive schema can't nest without limit.
//
// Not thread-safe; a matcher is used by one generation at a time. Copying a matcher copies its state (the grammar
// is shared, immutable).

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace strix::grammar {

// A parse deeper than this many frames is dropped (bounds the output's nesting; see the header comment).
constexpr int kMaxStackDepth = 256;
// More live parses than this after a byte throws: a grammar that ambiguous is a compiler bug, not an input to serve.
constexpr size_t kMaxParses = 4096;

struct ByteRange {
    uint8_t low = 0, high = 0;  // inclusive
};

struct Element {
    enum class Kind { Bytes, Rule };
    Kind kind = Kind::Bytes;
    std::vector<ByteRange> ranges;  // Bytes: matches one byte inside any of these ranges
    int rule = -1;                  // Rule: the referenced rule's index

    static Element byte(uint8_t value) { return Element{Kind::Bytes, {{value, value}}, -1}; }
    static Element byte_range(uint8_t low, uint8_t high) { return Element{Kind::Bytes, {{low, high}}, -1}; }
    static Element byte_set(std::vector<ByteRange> ranges) { return Element{Kind::Bytes, std::move(ranges), -1}; }
    static Element reference(int rule_index) { return Element{Kind::Rule, {}, rule_index}; }
};

using Alternative = std::vector<Element>;

struct Rule {
    std::string name;                       // for error messages and dumps
    std::vector<Alternative> alternatives;  // an empty Alternative matches the empty string
};

class Grammar {
public:
    // Adds a rule with no alternatives yet; returns its index. Names must be unique (they label errors).
    int add_rule(const std::string &name);
    // Appends an alternative to rule `rule_index`.
    void add_alternative(int rule_index, Alternative alternative);
    // A literal byte string as an alternative's elements (one Bytes element per byte).
    static Alternative literal(std::string_view bytes);
    void set_root(int rule_index);

    // Throws strix::Error naming the rule when: there is no root, a rule has no alternatives, a reference points
    // outside the rule list, a byte set is empty or has low > high, or a rule is left-recursive (can reach itself
    // without consuming a byte). Matchers only accept validated grammars.
    void validate() const;
    bool validated() const { return validated_; }

    int root() const { return root_; }
    const std::vector<Rule> &rules() const { return rules_; }
    int find_rule(const std::string &name) const;  // -1 if absent

    // One line per rule, `name := a b | c`, bytes as quoted strings or [ranges] - for tests and debugging.
    std::string dump() const;

private:
    std::vector<Rule> rules_;
    int root_ = -1;
    mutable bool validated_ = false;
};

class GrammarMatcher {
public:
    // The grammar must be validated (Grammar::validate) - checked here.
    explicit GrammarMatcher(std::shared_ptr<const Grammar> grammar);

    // Consumes one byte if some parse allows it and returns true; returns false (state unchanged) otherwise.
    bool advance(uint8_t byte);
    // All of `bytes` or nothing: returns false and leaves the state unchanged if any byte is refused.
    bool advance(std::string_view bytes);
    // Whether `bytes` would be accepted, without consuming them.
    bool allows(std::string_view bytes) const;

    // Some parse is complete: the text so far is a full sentence of the grammar (an end of output is allowed).
    bool accepting() const;
    // Some parse can take another byte.
    bool can_continue() const;
    // Every byte value some parse allows next (for tests and error messages; sorted, unique).
    std::vector<uint8_t> allowed_bytes() const;

    size_t parse_count() const { return parses_.size(); }
    size_t bytes_consumed() const { return bytes_consumed_; }
    const Grammar &grammar() const { return *grammar_; }

    struct Frame {
        int32_t rule, alternative, position;
        bool operator==(const Frame &other) const {
            return rule == other.rule && alternative == other.alternative && position == other.position;
        }
        bool operator<(const Frame &other) const {
            if (rule != other.rule) return rule < other.rule;
            if (alternative != other.alternative) return alternative < other.alternative;
            return position < other.position;
        }
    };
    using Stack = std::vector<Frame>;  // back() is the top; an empty stack is a complete parse
    // The live parses, sorted and unique: two matchers with equal parses accept exactly the same continuations.
    const std::vector<Stack> &parses() const { return parses_; }

private:
    // Expands `stack` until its top waits on a Bytes element (or it is empty) and appends the results to `out`.
    void expand(Stack stack, std::vector<Stack> &out) const;
    void normalize(std::vector<Stack> &parses) const;  // sort + unique, check kMaxParses
    const Element &top_element(const Stack &stack) const;

    std::shared_ptr<const Grammar> grammar_;
    std::vector<Stack> parses_;  // sorted, unique; every non-empty stack's top waits on a Bytes element
    size_t bytes_consumed_ = 0;
};

// The matcher's reachable states, numbered and their byte transitions cached - a lazily built DFA over the
// pushdown automaton. Token masks (serve/token_mask) walk the whole vocabulary from one state, byte by byte; with
// the transitions cached that walk is a hash lookup per byte instead of a parse-set copy, and states repeat a lot
// (inside a JSON string every character leads back to the same state). One automaton per compiled grammar, shared
// by every generation that uses the grammar; states are never forgotten (ids stay valid), bounded by kMaxStates.
// Not thread-safe (the engine runs one generation at a time).
class GrammarAutomaton {
public:
    // More states than this throws: a grammar whose matcher reaches that many distinct parse sets is too large to
    // serve (recursive schemas nest at most kMaxStackDepth deep, so this is a schema of that shape).
    static constexpr size_t kMaxStates = 1 << 18;
    static constexpr int32_t kDead = -1;

    // colliding_hash_for_tests: every state hashes alike, so finding a state relies on comparing parse sets alone
    // (tests check that path; a real hash collision is too rare to reach it otherwise).
    explicit GrammarAutomaton(std::shared_ptr<const Grammar> grammar, bool colliding_hash_for_tests = false);

    int32_t initial() const { return 0; }
    // The state after `byte`, or kDead if no parse allows it. Computed once per (state, byte), then cached.
    int32_t next(int32_t state, uint8_t byte);
    // The state after all of `bytes`, or kDead.
    int32_t next(int32_t state, std::string_view bytes);
    bool accepting(int32_t state) const;
    bool can_continue(int32_t state) const;
    size_t state_count() const { return states_.size(); }
    size_t transition_count() const { return transitions_.size(); }
    const Grammar &grammar() const { return *grammar_; }

    // Token masks cached per state (serve/token_mask fills them; `owner` identifies the vocabulary they were made
    // for - one automaton serves one vocabulary). nullptr if not cached yet.
    const std::vector<uint32_t> *cached_mask(int32_t state, const void *owner) const;
    // Stores a mask (dropping every cached mask first once kMaxCachedMasks are held) and returns the stored copy.
    // Pointers / references to cached masks stay valid until the next store_mask.
    const std::vector<uint32_t> &store_mask(int32_t state, const void *owner, std::vector<uint32_t> mask);
    static constexpr size_t kMaxCachedMasks = 2048;  // 31 KB each at the target's 248,320 rows: <= 62 MiB
    size_t cached_mask_count() const { return masks_.size(); }

private:
    int32_t intern(GrammarMatcher matcher);
    void check_state(int32_t state, const char *what) const;

    std::shared_ptr<const Grammar> grammar_;
    bool colliding_hash_ = false;
    std::vector<GrammarMatcher> states_;
    std::vector<uint8_t> accepting_, can_continue_;
    std::unordered_map<uint64_t, std::vector<int32_t>> by_hash_;  // parse-set hash -> states with that hash
    std::unordered_map<uint64_t, int32_t> transitions_;            // state << 8 | byte -> next state (or kDead)
    std::unordered_map<int32_t, std::vector<uint32_t>> masks_;
    const void *mask_owner_ = nullptr;
};

}  // namespace strix::grammar
