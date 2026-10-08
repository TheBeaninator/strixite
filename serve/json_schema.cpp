#include "serve/json_schema.hpp"

#include "common/check.hpp"

#include <algorithm>
#include <deque>
#include <map>
#include <set>

namespace strix::json_schema {

namespace {

using grammar::Alternative;
using grammar::ByteRange;
using grammar::Element;
using grammar::Grammar;

// Keywords that carry no constraint: skipped.
const std::set<std::string> kAnnotations = {"description", "title",    "default",  "examples", "$comment",
                                            "$schema",     "$id",      "deprecated", "readOnly", "writeOnly"};
// Keywords the compiler enforces.
const std::set<std::string> kEnforced = {"type",      "properties", "required", "additionalProperties", "items",
                                         "minItems",  "maxItems",   "minLength", "maxLength",           "enum",
                                         "const",     "anyOf",      "oneOf",     "$ref",                "$defs",
                                         "definitions"};
const std::set<std::string> kObjectKeywords = {"properties", "required", "additionalProperties"};
const std::set<std::string> kArrayKeywords = {"items", "minItems", "maxItems"};
const std::set<std::string> kStringKeywords = {"minLength", "maxLength"};
const std::set<std::string> kTypeNames = {"object", "array", "string", "integer", "number", "boolean", "null"};

const char *kEnforcedList =
    "type, properties, required, additionalProperties, items, minItems, maxItems, minLength, maxLength, enum, const, "
    "anyOf, oneOf, $ref, $defs";

void append(Alternative &alternative, const Alternative &more) {
    alternative.insert(alternative.end(), more.begin(), more.end());
}

Alternative sequence(std::initializer_list<Alternative> parts) {
    Alternative out;
    for (const Alternative &part : parts) append(out, part);
    return out;
}

Alternative ref(int rule_index) { return Alternative{Element::reference(rule_index)}; }
Alternative literal(std::string_view bytes) { return Grammar::literal(bytes); }
Alternative bytes(std::vector<ByteRange> ranges) { return Alternative{Element::byte_set(std::move(ranges))}; }

class Compiler {
public:
    explicit Compiler(std::string root_path) : root_path_(std::move(root_path)) {}

    std::shared_ptr<const Grammar> finish_with_root(int value_rule) {
        const int root = grammar_.add_rule("root");
        grammar_.add_alternative(root, sequence({ref(whitespace()), ref(value_rule), ref(whitespace())}));
        grammar_.set_root(root);
        try {
            grammar_.validate();
        } catch (const Error &error) {
            fail_at(__FILE__, __LINE__, __func__,
                    cat(root_path_, ": the schema refers to itself without any output in between (a $ref / anyOf "
                        "cycle that never reaches a value) - ", error.what()));
        }
        return std::make_shared<const Grammar>(std::move(grammar_));
    }

    int any_object() {
        return shared("object", [&](int rule) {
            const int member = grammar_.add_rule("object.member");
            grammar_.add_alternative(member, sequence({ref(string_rule()), ref(whitespace()), literal(":"),
                                                      ref(whitespace()), ref(any_value())}));
            fill_delimited_list(rule, "{", "}", member, "object");
        });
    }

    int any_value() {
        return shared("value", [&](int rule) {
            grammar_.add_alternative(rule, ref(any_object()));
            grammar_.add_alternative(rule, ref(array_of(-1, "array")));
            grammar_.add_alternative(rule, ref(string_rule()));
            grammar_.add_alternative(rule, ref(number_rule()));
            grammar_.add_alternative(rule, literal("true"));
            grammar_.add_alternative(rule, literal("false"));
            grammar_.add_alternative(rule, literal("null"));
        });
    }

    // Compiles `schema` (at `path`) into a new rule; returns its index.
    int node(const json::Value &schema, const std::string &path, int depth) {
        const int rule = grammar_.add_rule(cat("n", next_name_++, " ", path));
        fill(rule, schema, path, depth);
        return rule;
    }

    void compile_root(const json::Value &schema, int root_rule) {
        root_schema_ = &schema;
        root_rule_ = root_rule;
        fill(root_rule, schema, root_path_, 0);
        while (!pending_definitions_.empty()) {
            const PendingDefinition definition = pending_definitions_.front();
            pending_definitions_.pop_front();
            fill(definition.rule, *definition.schema, definition.path, 1);
        }
    }

    int add_rule(const std::string &name) { return grammar_.add_rule(name); }

private:
    struct PendingDefinition {
        int rule;
        const json::Value *schema;
        std::string path;
    };

    // A rule built once and shared by every use (white space, strings, numbers, ...).
    template <typename Builder>
    int shared(const std::string &name, Builder build) {
        const auto found = shared_rules_.find(name);
        if (found != shared_rules_.end()) return found->second;
        const int rule = grammar_.add_rule(name);
        shared_rules_[name] = rule;  // before building: the builder may refer to the rule itself
        build(rule);
        return rule;
    }

    // Matches 0..count bytes, each from `ranges`: chain[k] := "" | byte chain[k + 1], chain[count] := "".
    int bounded_run(const std::string &name, std::vector<ByteRange> ranges, int count) {
        return shared(cat(name, "{0,", count, "}"), [&](int first) {
            int current = first;
            for (int index = 0; index < count; ++index) {
                const int next = index + 1 < count ? grammar_.add_rule(cat(name, "{0,", count, "}.", index + 1))
                                                   : grammar_.add_rule(cat(name, "{0,", count, "}.end"));
                grammar_.add_alternative(current, Alternative{});
                grammar_.add_alternative(current, sequence({bytes(ranges), ref(next)}));
                current = next;
            }
            grammar_.add_alternative(current, Alternative{});
        });
    }

    int whitespace() {
        return bounded_run("ws", {{' ', ' '}, {'\t', '\t'}, {'\n', '\n'}, {'\r', '\r'}}, kMaxWhitespaceRun);
    }

    int digits(int count) { return bounded_run("digits", {{'0', '9'}}, count); }

    int integer_rule() {
        return shared("integer", [&](int rule) {
            const int magnitude = grammar_.add_rule("integer.magnitude");
            grammar_.add_alternative(magnitude, literal("0"));
            grammar_.add_alternative(magnitude, sequence({bytes({{'1', '9'}}), ref(digits(kMaxIntegerDigits - 1))}));
            grammar_.add_alternative(rule, ref(magnitude));
            grammar_.add_alternative(rule, sequence({literal("-"), ref(magnitude)}));
        });
    }

    int number_rule() {
        return shared("number", [&](int rule) {
            const int fraction = grammar_.add_rule("number.fraction");
            grammar_.add_alternative(fraction, Alternative{});
            grammar_.add_alternative(fraction,
                                     sequence({literal("."), bytes({{'0', '9'}}), ref(digits(kMaxFractionDigits - 1))}));
            const int sign = grammar_.add_rule("number.exponent_sign");
            grammar_.add_alternative(sign, Alternative{});
            grammar_.add_alternative(sign, bytes({{'+', '+'}, {'-', '-'}}));
            const int exponent = grammar_.add_rule("number.exponent");
            grammar_.add_alternative(exponent, Alternative{});
            grammar_.add_alternative(exponent, sequence({bytes({{'E', 'E'}, {'e', 'e'}}), ref(sign), bytes({{'0', '9'}}),
                                                         ref(digits(kMaxExponentDigits - 1))}));
            grammar_.add_alternative(rule, sequence({ref(integer_rule()), ref(fraction), ref(exponent)}));
        });
    }

    // A JSON string: printable ASCII except '"' and '\', escapes, and well-formed UTF-8 (RFC 3629's table).
    int string_rule() {
        return shared("string", [&](int rule) {
            const int character = grammar_.add_rule("string.character");
            grammar_.add_alternative(character, bytes({{0x20, 0x21}, {0x23, 0x5b}, {0x5d, 0x7f}}));
            const int escape = grammar_.add_rule("string.escape");
            grammar_.add_alternative(escape, bytes({{'"', '"'}, {'\\', '\\'}, {'/', '/'}, {'b', 'b'}, {'f', 'f'},
                                                    {'n', 'n'}, {'r', 'r'}, {'t', 't'}}));
            const std::vector<ByteRange> hex = {{'0', '9'}, {'A', 'F'}, {'a', 'f'}};
            // \u escapes: a code point outside the surrogates, or a high + low surrogate pair - one character either
            // way (what minLength / maxLength count), and never a lone surrogate (not valid text; json::Value refuses it).
            grammar_.add_alternative(escape, sequence({literal("u"), bytes({{'0', '9'}, {'A', 'C'}, {'a', 'c'}}), bytes(hex),
                                                       bytes(hex), bytes(hex)}));
            grammar_.add_alternative(escape, sequence({literal("u"), bytes({{'D', 'D'}, {'d', 'd'}}), bytes({{'0', '7'}}),
                                                       bytes(hex), bytes(hex)}));
            grammar_.add_alternative(escape, sequence({literal("u"), bytes({{'E', 'F'}, {'e', 'f'}}), bytes(hex), bytes(hex),
                                                       bytes(hex)}));
            grammar_.add_alternative(escape, sequence({literal("u"), bytes({{'D', 'D'}, {'d', 'd'}}),
                                                       bytes({{'8', '9'}, {'A', 'B'}, {'a', 'b'}}), bytes(hex), bytes(hex),
                                                       literal("\\u"), bytes({{'D', 'D'}, {'d', 'd'}}),
                                                       bytes({{'C', 'F'}, {'c', 'f'}}), bytes(hex), bytes(hex)}));
            grammar_.add_alternative(character, sequence({literal("\\"), ref(escape)}));
            const std::vector<ByteRange> continuation = {{0x80, 0xbf}};
            grammar_.add_alternative(character, sequence({bytes({{0xc2, 0xdf}}), bytes(continuation)}));
            grammar_.add_alternative(character, sequence({literal("\xe0"), bytes({{0xa0, 0xbf}}), bytes(continuation)}));
            grammar_.add_alternative(character,
                                     sequence({bytes({{0xe1, 0xec}, {0xee, 0xef}}), bytes(continuation), bytes(continuation)}));
            grammar_.add_alternative(character, sequence({literal("\xed"), bytes({{0x80, 0x9f}}), bytes(continuation)}));
            grammar_.add_alternative(character, sequence({literal("\xf0"), bytes({{0x90, 0xbf}}), bytes(continuation),
                                                          bytes(continuation)}));
            grammar_.add_alternative(character, sequence({bytes({{0xf1, 0xf3}}), bytes(continuation),
                                                          bytes(continuation), bytes(continuation)}));
            grammar_.add_alternative(character, sequence({literal("\xf4"), bytes({{0x80, 0x8f}}), bytes(continuation),
                                                          bytes(continuation)}));
            const int rest = grammar_.add_rule("string.rest");
            grammar_.add_alternative(rest, literal("\""));
            grammar_.add_alternative(rest, sequence({ref(character), ref(rest)}));
            grammar_.add_alternative(rule, sequence({literal("\""), ref(rest)}));
            string_character_ = character, string_rest_ = rest;
        });
    }

    // A string of min..max characters (max -1: no upper bound): rule := '"' count[0]; count[i] := '"' (i >= min) |
    // character count[i + 1] (i < max); unbounded, count[min] is the plain string's rest. Each count a rule, so a
    // bounded string's every position is its own grammar state (its own token mask, so long bounded strings cost masks).
    int bounded_string(int64_t min, int64_t max) {
        if (min == 0 && max < 0) return string_rule();
        string_rule();  // the character and rest rules
        return shared(cat("string{", min, ",", max < 0 ? std::string() : std::to_string(max), "}"), [&](int rule) {
            const int64_t last = max < 0 ? min : max;
            std::vector<int> count((size_t)last + 1);
            for (int64_t index = 0; index <= last; ++index)
                count[(size_t)index] = grammar_.add_rule(cat("string{", min, ",", max, "}.", index));
            for (int64_t index = 0; index <= last; ++index) {
                const int here = count[(size_t)index];
                if (max < 0 && index == min) {
                    grammar_.add_alternative(here, ref(string_rest_));
                    continue;
                }
                if (index >= min) grammar_.add_alternative(here, literal("\""));
                if (index < last) grammar_.add_alternative(here, sequence({ref(string_character_), ref(count[(size_t)index + 1])}));
            }
            grammar_.add_alternative(rule, sequence({literal("\""), ref(count[0])}));
        });
    }

    // rule := open ws body; body := close | item ws tail; tail := close | "," ws item ws tail.
    void fill_delimited_list(int rule, const char *open, const char *close, int item, const std::string &name) {
        const int body = grammar_.add_rule(cat(name, ".body#", next_name_));
        const int tail = grammar_.add_rule(cat(name, ".tail#", next_name_++));
        grammar_.add_alternative(rule, sequence({literal(open), ref(whitespace()), ref(body)}));
        grammar_.add_alternative(body, literal(close));
        grammar_.add_alternative(body, sequence({ref(item), ref(whitespace()), ref(tail)}));
        grammar_.add_alternative(tail, literal(close));
        grammar_.add_alternative(tail, sequence({literal(","), ref(whitespace()), ref(item), ref(whitespace()), ref(tail)}));
    }

    // An array of `item_rule` (-1: any value) with min..max items (max -1: no upper bound).
    // Bounded: rule := '[' ws after[0]; after[0] := ']' (min 0) | item ws after[1] (max > 0); after[i >= 1] := ']'
    // (i >= min) | ',' ws item ws after[i + 1] (i < max); unbounded, after[last] repeats itself.
    int array_of(int item_rule, const std::string &name, int64_t min = 0, int64_t max = -1) {
        if (min == 0 && max < 0) {
            if (item_rule < 0) {
                return shared("array", [&](int rule) { fill_delimited_list(rule, "[", "]", any_value(), "array"); });
            }
            const int rule = grammar_.add_rule(cat(name, "#", next_name_++));
            fill_delimited_list(rule, "[", "]", item_rule, name);
            return rule;
        }
        const int item = item_rule < 0 ? any_value() : item_rule;
        const int rule = grammar_.add_rule(cat(name, "{", min, ",", max, "}#", next_name_));
        const int64_t last = max < 0 ? std::max<int64_t>(min, 1) : max;
        std::vector<int> after((size_t)last + 1);
        for (int64_t index = 0; index <= last; ++index)
            after[(size_t)index] = grammar_.add_rule(cat(name, "{", min, ",", max, "}#", next_name_, ".", index));
        ++next_name_;
        for (int64_t index = 0; index <= last; ++index) {
            const int here = after[(size_t)index];
            if (index >= min) grammar_.add_alternative(here, literal("]"));
            const Alternative next_item =
                index == 0 ? sequence({ref(item), ref(whitespace())})
                           : sequence({literal(","), ref(whitespace()), ref(item), ref(whitespace())});
            if (index < last) grammar_.add_alternative(here, sequence({next_item, ref(after[(size_t)index + 1])}));
            else if (max < 0) grammar_.add_alternative(here, sequence({next_item, ref(here)}));  // unbounded: repeat
        }
        grammar_.add_alternative(rule, sequence({literal("["), ref(whitespace()), ref(after[0])}));
        return rule;
    }

    // A length keyword's value: absent -1, else an integer in 0..kMaxLengthBound.
    static int64_t length_bound(const json::Value &schema, const char *keyword, const std::string &path) {
        const json::Value *value = schema.find(keyword);
        if (value == nullptr) return -1;
        STRIX_CHECK(value->is_int(), path, ".", keyword, ": expected a non-negative integer, found ",
                    json::type_name(value->type()));
        const int64_t bound = value->as_int(cat(path, ".", keyword));
        STRIX_CHECK(bound >= 0 && bound <= kMaxLengthBound, path, ".", keyword, ": ", bound, " is outside 0..",
                    kMaxLengthBound, " (structured output enforces a length by counting it out in the grammar, up to ",
                    kMaxLengthBound, ")");
        return bound;
    }

    int resolve_reference(const json::Value &reference, const std::string &path) {
        const std::string &target = reference.as_string(path + ".$ref");
        if (target == "#") return root_rule_;
        std::string container, name;
        for (const char *prefix : {"#/$defs/", "#/definitions/"}) {
            const std::string prefix_text = prefix;
            if (target.compare(0, prefix_text.size(), prefix_text) == 0) {
                container = prefix_text.substr(2, prefix_text.size() - 3);
                name = target.substr(prefix_text.size());
            }
        }
        STRIX_CHECK(!container.empty() && !name.empty() && name.find('/') == std::string::npos, path,
                    ".$ref: '", target, "' - supported references are \"#\", \"#/$defs/NAME\" and "
                    "\"#/definitions/NAME\" (definitions at the schema's root)");
        std::string unescaped;  // JSON pointer escapes: ~1 is '/', ~0 is '~'
        for (size_t index = 0; index < name.size(); ++index) {
            if (name[index] == '~' && index + 1 < name.size() && (name[index + 1] == '0' || name[index + 1] == '1')) {
                unescaped += name[index + 1] == '0' ? '~' : '/';
                ++index;
            } else {
                unescaped += name[index];
            }
        }
        const std::string key = container + "/" + unescaped;
        const auto known = definition_rules_.find(key);
        if (known != definition_rules_.end()) return known->second;
        const json::Value *definitions = root_schema_->is_object() ? root_schema_->find(container) : nullptr;
        STRIX_CHECK(definitions != nullptr && definitions->is_object(), path, ".$ref: '", target, "' - the schema's root has no '",
                    container, "' object");
        const json::Value *definition = definitions->find(unescaped);
        STRIX_CHECK(definition != nullptr, path, ".$ref: '", target, "' - no definition '", unescaped, "' in the root's '",
                    container, "'");
        const std::string definition_path = cat(root_path_, ".", container, ".", unescaped);
        const int rule = grammar_.add_rule(cat("n", next_name_++, " ", definition_path));
        definition_rules_[key] = rule;
        pending_definitions_.push_back(PendingDefinition{rule, definition, definition_path});
        return rule;
    }

    void fill_object(int rule, const json::Value &schema, const std::string &path, int depth) {
        const json::Value *properties = schema.find("properties");
        const json::Value *required = schema.find("required");
        const json::Value *additional = schema.find("additionalProperties");
        if (additional != nullptr) {
            STRIX_CHECK(additional->is_bool(), path, ".additionalProperties: a schema there is not supported (only "
                        "true or false; found ", json::type_name(additional->type()), ") - its keys couldn't be told "
                        "apart from the listed properties'");
        }
        std::set<std::string> required_names;
        if (required != nullptr) {
            for (size_t index = 0; index < required->as_array(path + ".required").size(); ++index)
                required_names.insert(required->as_array(path + ".required")[index].as_string(
                    cat(path, ".required[", index, "]")));
        }
        if (properties == nullptr) {
            STRIX_CHECK(required_names.empty(), path, ".required lists '", *required_names.begin(),
                        "' but the schema has no 'properties' to say what it holds");
            if (additional != nullptr && !additional->as_bool(path + ".additionalProperties")) {
                grammar_.add_alternative(rule, sequence({literal("{"), ref(whitespace()), literal("}")}));
            } else {
                grammar_.add_alternative(rule, ref(any_object()));
            }
            return;
        }
        const auto &members = properties->as_object(path + ".properties");
        for (const std::string &name : required_names) {
            const bool listed = std::any_of(members.begin(), members.end(), [&](const auto &m) { return m.first == name; });
            STRIX_CHECK(listed, path, ".required lists '", name, "', which is not in 'properties' (",
                        members.size(), " listed)");
        }
        // first[i]: no member written yet, deciding property i onward; more[i]: at least one written.
        // first[i] := kv[i] more[i+1] | first[i+1] (optional only); more[i] := "," ws kv[i] more[i+1] | more[i+1]
        // (optional only); first[n] := more[n] := "". kv[i] := "name" ws ":" ws value ws.
        const size_t count = members.size();
        std::vector<int> first(count + 1), more(count + 1);
        for (size_t index = 0; index <= count; ++index) {
            first[index] = grammar_.add_rule(cat("n", next_name_, " ", path, ".first[", index, "]"));
            more[index] = grammar_.add_rule(cat("n", next_name_, " ", path, ".more[", index, "]"));
        }
        ++next_name_;
        grammar_.add_alternative(first[count], Alternative{});
        grammar_.add_alternative(more[count], Alternative{});
        for (size_t index = 0; index < count; ++index) {
            const auto &[name, property_schema] = members[index];
            std::string quoted;
            json::append_quoted(quoted, name);
            const int value = node(property_schema, cat(path, ".properties.", name), depth + 1);
            const Alternative key_value =
                sequence({literal(quoted), ref(whitespace()), literal(":"), ref(whitespace()), ref(value), ref(whitespace())});
            grammar_.add_alternative(first[index], sequence({key_value, ref(more[index + 1])}));
            grammar_.add_alternative(more[index],
                                     sequence({literal(","), ref(whitespace()), key_value, ref(more[index + 1])}));
            if (required_names.count(name) == 0) {
                grammar_.add_alternative(first[index], ref(first[index + 1]));
                grammar_.add_alternative(more[index], ref(more[index + 1]));
            }
        }
        grammar_.add_alternative(rule, sequence({literal("{"), ref(whitespace()), ref(first[0]), literal("}")}));
    }

    void fill(int rule, const json::Value &schema, const std::string &path, int depth) {
        STRIX_CHECK(depth <= kMaxSchemaDepth, path, ": the schema nests deeper than ", kMaxSchemaDepth,
                    " levels (the limit for structured output)");
        if (schema.is_bool()) {
            STRIX_CHECK(schema.as_bool(path), path, ": the schema is `false` - no value could match it");
            grammar_.add_alternative(rule, ref(any_value()));
            return;
        }
        STRIX_CHECK(schema.is_object(), path, ": a schema must be an object or a boolean, found ",
                    json::type_name(schema.type()));
        std::vector<std::string> constraints;  // enforced keywords present, in the schema's order
        for (const auto &[keyword, value] : schema.as_object(path)) {
            if (kAnnotations.count(keyword)) continue;
            STRIX_CHECK(kEnforced.count(keyword), path, ": keyword '", keyword, "' is not supported by structured "
                        "output - strix-server can't enforce it while generating (enforced: ", kEnforcedList,
                        "; annotations like description are ignored). Remove it, or check it client-side.");
            if (keyword == "$defs" || keyword == "definitions") {
                STRIX_CHECK(depth == 0 && &schema == root_schema_, path, ": '", keyword,
                            "' is supported at the schema's root only");
                STRIX_CHECK(value.is_object(), path, ".", keyword, ": expected an object of named schemas, found ",
                            json::type_name(value.type()));
                continue;
            }
            constraints.push_back(keyword);
        }
        auto only = [&](const std::set<std::string> &allowed, const std::string &keyword) {
            for (const std::string &other : constraints)
                STRIX_CHECK(other == keyword || allowed.count(other), path, ": '", keyword, "' together with '", other,
                            "' is not supported by structured output (both would have to hold at once) - put the "
                            "constraint inside the '", keyword, "' branches instead");
        };
        if (schema.find("$ref") != nullptr) {
            only({}, "$ref");
            grammar_.add_alternative(rule, ref(resolve_reference(*schema.find("$ref"), path)));
            return;
        }
        for (const char *combinator : {"anyOf", "oneOf"}) {
            const json::Value *options = schema.find(combinator);
            if (options == nullptr) continue;
            only({}, combinator);
            const auto &list = options->as_array(path + "." + combinator);
            STRIX_CHECK(!list.empty(), path, ".", combinator, ": expected at least one schema, found an empty array");
            for (size_t index = 0; index < list.size(); ++index)
                grammar_.add_alternative(rule, ref(node(list[index], cat(path, ".", combinator, "[", index, "]"), depth + 1)));
            return;
        }

        // Types: listed, or inferred from the keywords present (properties -> object, items -> array), else any.
        std::vector<std::string> types;
        if (const json::Value *type = schema.find("type")) {
            if (type->is_string()) {
                types.push_back(type->as_string(path + ".type"));
            } else {
                const auto &list = type->as_array(path + ".type");
                STRIX_CHECK(!list.empty(), path, ".type: expected a type name or a non-empty array of them");
                for (size_t index = 0; index < list.size(); ++index)
                    types.push_back(list[index].as_string(cat(path, ".type[", index, "]")));
            }
            for (const std::string &name : types)
                STRIX_CHECK(kTypeNames.count(name), path, ".type: '", name,
                            "' is not a JSON schema type (object, array, string, integer, number, boolean, null)");
        }

        const json::Value *enumeration = schema.find("enum");
        const json::Value *constant = schema.find("const");
        if (enumeration != nullptr || constant != nullptr) {
            const std::string keyword = enumeration != nullptr ? "enum" : "const";
            only({"type"}, keyword);
            std::vector<json::Value> values;
            if (enumeration != nullptr) {
                values = enumeration->as_array(path + ".enum");
                STRIX_CHECK(!values.empty(), path, ".enum: expected at least one value, found an empty array");
            } else {
                values.push_back(*constant);
            }
            for (size_t index = 0; index < values.size(); ++index) {
                if (!types.empty()) {
                    const json::Type found = values[index].type();
                    const bool fits = std::any_of(types.begin(), types.end(), [&](const std::string &name) {
                        return (name == "object" && found == json::Type::Object) ||
                               (name == "array" && found == json::Type::Array) ||
                               (name == "string" && found == json::Type::String) ||
                               (name == "integer" && found == json::Type::Int) ||
                               (name == "number" && (found == json::Type::Int || found == json::Type::Float)) ||
                               (name == "boolean" && found == json::Type::Bool) ||
                               (name == "null" && found == json::Type::Null);
                    });
                    STRIX_CHECK(fits, path, ".", keyword, (enumeration != nullptr ? cat("[", index, "]") : ""), ": ",
                                values[index].dump(), " is ", json::type_name(found), ", not one of the schema's types ",
                                list_str(types));
                }
                grammar_.add_alternative(rule, literal(values[index].dump()));
            }
            return;
        }

        const bool has_object_keywords = std::any_of(constraints.begin(), constraints.end(),
                                                     [](const std::string &k) { return kObjectKeywords.count(k) > 0; });
        const bool has_items = std::any_of(constraints.begin(), constraints.end(),
                                           [](const std::string &k) { return kArrayKeywords.count(k) > 0; });
        const bool has_lengths = std::any_of(constraints.begin(), constraints.end(),
                                             [](const std::string &k) { return kStringKeywords.count(k) > 0; });
        if (types.empty()) {
            if (has_object_keywords) types.push_back("object");
            if (has_items) types.push_back("array");
            if (has_lengths) types.push_back("string");
        }
        if (types.empty()) {
            grammar_.add_alternative(rule, ref(any_value()));
            return;
        }
        const bool any_object_type = std::count(types.begin(), types.end(), "object") > 0;
        const bool any_array_type = std::count(types.begin(), types.end(), "array") > 0;
        STRIX_CHECK(!has_object_keywords || any_object_type, path, ": object keywords (properties / required / "
                    "additionalProperties) with a type that isn't 'object' (", list_str(types), ")");
        STRIX_CHECK(!has_items || any_array_type, path, ": array keywords (items / minItems / maxItems) with a type that "
                    "isn't 'array' (", list_str(types), ")");
        STRIX_CHECK(!has_lengths || std::count(types.begin(), types.end(), "string") > 0, path,
                    ": minLength / maxLength with a type that isn't 'string' (", list_str(types), ")");
        const int64_t min_items = length_bound(schema, "minItems", path), max_items = length_bound(schema, "maxItems", path);
        const int64_t min_length = length_bound(schema, "minLength", path), max_length = length_bound(schema, "maxLength", path);
        STRIX_CHECK(max_items < 0 || min_items <= max_items, path, ": minItems ", min_items, " > maxItems ", max_items,
                    " - no array could match");
        STRIX_CHECK(max_length < 0 || min_length <= max_length, path, ": minLength ", min_length, " > maxLength ",
                    max_length, " - no string could match");
        const bool has_number = std::count(types.begin(), types.end(), "number") > 0;
        std::set<std::string> done;
        for (const std::string &name : types) {
            if (!done.insert(name).second) continue;  // a type listed twice
            if (name == "object") {
                fill_object(rule, schema, path, depth);
            } else if (name == "array") {
                const json::Value *items = schema.find("items");
                if (items != nullptr) {
                    STRIX_CHECK(!items->is_array(), path, ".items: an array of schemas (tuple form) is not supported - "
                                "give one schema for every item");
                }
                grammar_.add_alternative(
                    rule, ref(array_of(items != nullptr ? node(*items, path + ".items", depth + 1) : -1, path + ".array",
                                       std::max<int64_t>(min_items, 0), max_items)));
            } else if (name == "string") {
                grammar_.add_alternative(rule, ref(bounded_string(std::max<int64_t>(min_length, 0), max_length)));
            } else if (name == "integer") {
                if (!has_number) grammar_.add_alternative(rule, ref(integer_rule()));  // number already covers it
            } else if (name == "number") {
                grammar_.add_alternative(rule, ref(number_rule()));
            } else if (name == "boolean") {
                grammar_.add_alternative(rule, literal("true"));
                grammar_.add_alternative(rule, literal("false"));
            } else {
                grammar_.add_alternative(rule, literal("null"));
            }
        }
    }

    Grammar grammar_;
    std::string root_path_;
    std::map<std::string, int> shared_rules_;
    std::map<std::string, int> definition_rules_;
    std::deque<PendingDefinition> pending_definitions_;
    const json::Value *root_schema_ = nullptr;
    int string_character_ = -1, string_rest_ = -1;  // string_rule()'s parts, for bounded strings
    int root_rule_ = -1;
    int next_name_ = 0;
};

}  // namespace

std::shared_ptr<const Grammar> compile_any_object() {
    Compiler compiler("response_format");
    return compiler.finish_with_root(compiler.any_object());
}

std::shared_ptr<const Grammar> compile(const json::Value &schema, const std::string &path) {
    STRIX_CHECK(!path.empty(), "json_schema::compile needs the schema's path for its error messages");
    Compiler compiler(path);
    const int value_rule = compiler.add_rule("schema");
    compiler.compile_root(schema, value_rule);
    return compiler.finish_with_root(value_rule);
}

}  // namespace strix::json_schema
