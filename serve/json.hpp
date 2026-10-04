#pragma once

// Hand-rolled JSON (the schema parsed and emitted here is small and fixed, so no vendored library). Strict RFC 8259
// parsing of UTF-8 text; objects keep their key order (the chat template's `tojson` prints tools in the order the
// client sent them) and a duplicate key keeps its first position with the last value, as Python's json.loads does.
// Integers keep their literal digits (Python ints are unbounded, and `tojson` prints them back unchanged); as_int()
// range-checks.
//
// Two printers: dump() - compact, for the server's responses (floats as Python repr, so they re-parse as floats);
// dump_python() - exactly what Python's json.dumps(x, ensure_ascii=False) prints (", " / ": " separators, repr()
// floats), which transformers' chat template `tojson` filter uses. Parse errors and type mismatches throw strix::Error
// naming the path ("messages[2].content") and what was expected vs found.

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::json {

enum class Type { Null, Bool, Int, Float, String, Array, Object };
const char *type_name(Type t);

class Value {
public:
    Value() = default;  // null
    static Value null() { return Value(); }
    static Value boolean(bool b);
    static Value integer(int64_t v);
    static Value number(double v);  // must be finite
    static Value string(std::string s);
    static Value array();
    static Value object();

    // Throws with the byte offset, line:column and what was expected. max_depth bounds nesting.
    static Value parse(std::string_view text, int max_depth = 128);

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Bool; }
    bool is_int() const { return type_ == Type::Int; }
    bool is_number() const { return type_ == Type::Int || type_ == Type::Float; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    // Typed access; `path` names the value in the error ("request.max_tokens").
    bool as_bool(const std::string &path) const;
    int64_t as_int(const std::string &path) const;   // Int within int64 range (1.0 is a Float: refused)
    double as_double(const std::string &path) const;  // Int or Float
    const std::string &as_string(const std::string &path) const;
    const std::vector<Value> &as_array(const std::string &path) const;
    const std::vector<std::pair<std::string, Value>> &as_object(const std::string &path) const;

    // Object lookup: nullptr if absent (or if this isn't an object).
    const Value *find(std::string_view key) const;
    // Array / object building. push/set on the wrong type throw.
    Value &push(Value v);
    Value &set(std::string key, Value v);  // replaces an existing key in place

    std::string dump() const;
    std::string dump_python() const;
    const std::string &int_literal() const { return s_; }  // Int only: the literal digits

    bool operator==(const Value &o) const;

private:
    Type type_ = Type::Null;
    bool b_ = false;
    double d_ = 0;
    std::string s_;  // String text, or an Int's literal
    std::vector<Value> arr_;
    std::vector<std::pair<std::string, Value>> obj_;
    friend class Parser;
    void dump_to(std::string &out, bool python) const;
};

// Appends s as a JSON string literal (quotes included), escaping ", \, and control characters (\n \r \t \b \f,
// others \u00xx); non-ASCII passes through (ensure_ascii=False).
void append_quoted(std::string &out, std::string_view s);
// Python's repr() of a finite double (shortest round-trip digits; exponent form below 1e-4 and from 1e16).
std::string python_float_repr(double v);
// True if s is well-formed UTF-8 (no overlongs, surrogates or code points past U+10FFFF).
bool valid_utf8(std::string_view s);

}  // namespace strix::json
