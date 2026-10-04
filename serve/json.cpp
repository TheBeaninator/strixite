#include "serve/json.hpp"

#include "common/check.hpp"

#include <charconv>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace strix::json {

const char *type_name(Type t) {
    switch (t) {
    case Type::Null: return "null";
    case Type::Bool: return "a boolean";
    case Type::Int: return "an integer";
    case Type::Float: return "a number";
    case Type::String: return "a string";
    case Type::Array: return "an array";
    case Type::Object: return "an object";
    }
    return "?";
}

Value Value::boolean(bool b) {
    Value v;
    v.type_ = Type::Bool, v.b_ = b;
    return v;
}
Value Value::integer(int64_t i) {
    Value v;
    v.type_ = Type::Int, v.s_ = std::to_string(i), v.d_ = (double)i;
    return v;
}
Value Value::number(double d) {
    STRIX_CHECK(std::isfinite(d), "json::Value::number: ", d, " is not finite (JSON has no NaN / inf)");
    Value v;
    v.type_ = Type::Float, v.d_ = d;
    return v;
}
Value Value::string(std::string s) {
    Value v;
    v.type_ = Type::String, v.s_ = std::move(s);
    return v;
}
Value Value::array() {
    Value v;
    v.type_ = Type::Array;
    return v;
}
Value Value::object() {
    Value v;
    v.type_ = Type::Object;
    return v;
}

namespace {
[[noreturn]] void type_fail(const std::string &path, const char *want, Type got) {
    STRIX_FAIL("'", path, "' must be ", want, ", got ", type_name(got));
}
}  // namespace

bool Value::as_bool(const std::string &path) const {
    if (type_ != Type::Bool) type_fail(path, "a boolean", type_);
    return b_;
}
int64_t Value::as_int(const std::string &path) const {
    if (type_ != Type::Int) type_fail(path, "an integer", type_);
    int64_t v = 0;
    const auto [end, ec] = std::from_chars(s_.data(), s_.data() + s_.size(), v);
    STRIX_CHECK(ec == std::errc() && end == s_.data() + s_.size(), "'", path, "' = ", s_,
                " is out of the 64-bit integer range");
    return v;
}
double Value::as_double(const std::string &path) const {
    if (type_ != Type::Int && type_ != Type::Float) type_fail(path, "a number", type_);
    return d_;
}
const std::string &Value::as_string(const std::string &path) const {
    if (type_ != Type::String) type_fail(path, "a string", type_);
    return s_;
}
const std::vector<Value> &Value::as_array(const std::string &path) const {
    if (type_ != Type::Array) type_fail(path, "an array", type_);
    return arr_;
}
const std::vector<std::pair<std::string, Value>> &Value::as_object(const std::string &path) const {
    if (type_ != Type::Object) type_fail(path, "an object", type_);
    return obj_;
}

const Value *Value::find(std::string_view key) const {
    if (type_ != Type::Object) return nullptr;
    for (const auto &[k, v] : obj_)
        if (k == key) return &v;
    return nullptr;
}

Value &Value::push(Value v) {
    STRIX_CHECK(type_ == Type::Array, "json::Value::push on ", type_name(type_), ", expected an array");
    arr_.push_back(std::move(v));
    return arr_.back();
}

Value &Value::set(std::string key, Value v) {
    STRIX_CHECK(type_ == Type::Object, "json::Value::set('", key, "') on ", type_name(type_), ", expected an object");
    for (auto &[k, old] : obj_)
        if (k == key) return old = std::move(v);
    obj_.emplace_back(std::move(key), std::move(v));
    return obj_.back().second;
}

bool Value::operator==(const Value &o) const {
    if (type_ != o.type_) return false;
    switch (type_) {
    case Type::Null: return true;
    case Type::Bool: return b_ == o.b_;
    case Type::Int: return s_ == o.s_;
    case Type::Float: return d_ == o.d_;
    case Type::String: return s_ == o.s_;
    case Type::Array: return arr_ == o.arr_;
    case Type::Object: return obj_ == o.obj_;
    }
    return false;
}

// ---- UTF-8 ----

namespace {
// Decodes one code point at s[i]; returns its length, 0 if malformed.
int utf8_len(std::string_view s, size_t i) {
    const auto b = [&](size_t k) { return (unsigned char)s[k]; };
    const unsigned c = b(i);
    if (c < 0x80) return 1;
    int n;
    uint32_t cp;
    if ((c & 0xE0) == 0xC0) n = 2, cp = c & 0x1F;
    else if ((c & 0xF0) == 0xE0) n = 3, cp = c & 0x0F;
    else if ((c & 0xF8) == 0xF0) n = 4, cp = c & 0x07;
    else return 0;
    if (i + (size_t)n > s.size()) return 0;
    for (int k = 1; k < n; ++k) {
        if ((b(i + (size_t)k) & 0xC0) != 0x80) return 0;
        cp = (cp << 6) | (b(i + (size_t)k) & 0x3F);
    }
    static const uint32_t kMin[5] = {0, 0, 0x80, 0x800, 0x10000};
    if (cp < kMin[n] || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return 0;
    return n;
}

void append_utf8(std::string &out, uint32_t cp) {
    if (cp < 0x80) out += (char)cp;
    else if (cp < 0x800) out += (char)(0xC0 | (cp >> 6)), out += (char)(0x80 | (cp & 0x3F));
    else if (cp < 0x10000)
        out += (char)(0xE0 | (cp >> 12)), out += (char)(0x80 | ((cp >> 6) & 0x3F)), out += (char)(0x80 | (cp & 0x3F));
    else
        out += (char)(0xF0 | (cp >> 18)), out += (char)(0x80 | ((cp >> 12) & 0x3F)),
            out += (char)(0x80 | ((cp >> 6) & 0x3F)), out += (char)(0x80 | (cp & 0x3F));
}
}  // namespace

bool valid_utf8(std::string_view s) {
    for (size_t i = 0; i < s.size();) {
        const int n = utf8_len(s, i);
        if (n == 0) return false;
        i += (size_t)n;
    }
    return true;
}

// ---- parser ----

class Parser {
public:
    Parser(std::string_view t, int max_depth) : t_(t), max_depth_(max_depth) {}

    Value run() {
        STRIX_CHECK(max_depth_ >= 1, "json::Value::parse: max_depth ", max_depth_, " must be >= 1");
        ws();
        Value v = value(0);
        ws();
        if (i_ != t_.size()) fail("end of input after the top-level value");
        return v;
    }

private:
    std::string_view t_;
    size_t i_ = 0;
    int max_depth_;

    [[noreturn]] void fail(const std::string &expected) const {
        size_t line = 1, col = 1;
        for (size_t k = 0; k < i_ && k < t_.size(); ++k) (t_[k] == '\n') ? (++line, col = 1) : ++col;
        std::string found = i_ >= t_.size() ? "end of input" : "'" + std::string(t_.substr(i_, 12)) + "'";
        STRIX_FAIL("JSON parse error at byte ", i_, " (line ", line, ", column ", col, "): expected ", expected,
                   ", found ", found);
    }
    void ws() {
        while (i_ < t_.size() && (t_[i_] == ' ' || t_[i_] == '\t' || t_[i_] == '\n' || t_[i_] == '\r')) ++i_;
    }
    bool lit(const char *w) {
        const size_t n = std::strlen(w);
        if (t_.substr(i_, n) != w) return false;
        i_ += n;
        return true;
    }

    Value value(int depth) {
        if (i_ >= t_.size()) fail("a value");
        const char c = t_[i_];
        if (c == '{' || c == '[') {
            if (depth + 1 > max_depth_) fail("nesting at most " + std::to_string(max_depth_) + " deep");
            return c == '{' ? object(depth + 1) : array(depth + 1);
        }
        if (c == '"') return Value::string(string());
        if (c == '-' || (c >= '0' && c <= '9')) return number();
        if (lit("true")) return Value::boolean(true);
        if (lit("false")) return Value::boolean(false);
        if (lit("null")) return Value();
        fail("a value");
    }

    Value object(int depth) {
        Value v = Value::object();
        std::unordered_map<std::string, size_t> index;  // key -> position, once the object is big (a vocabulary)
        constexpr size_t kIndexFrom = 16;
        ++i_;
        ws();
        if (i_ < t_.size() && t_[i_] == '}') return ++i_, v;
        for (;;) {
            ws();
            if (i_ >= t_.size() || t_[i_] != '"') fail("a string key");
            std::string k = string();
            ws();
            if (i_ >= t_.size() || t_[i_] != ':') fail("':'");
            ++i_;
            ws();
            // A duplicate keeps its first position with the last value (Python's json.loads).
            if (v.obj_.size() < kIndexFrom) {
                v.set(std::move(k), value(depth));
                if (v.obj_.size() == kIndexFrom)
                    for (size_t n = 0; n < v.obj_.size(); ++n) index.emplace(v.obj_[n].first, n);
            } else {
                Value x = value(depth);
                const auto [it, fresh] = index.emplace(k, v.obj_.size());
                if (fresh) v.obj_.emplace_back(std::move(k), std::move(x));
                else v.obj_[it->second].second = std::move(x);
            }
            ws();
            if (i_ < t_.size() && t_[i_] == ',') { ++i_; continue; }
            if (i_ < t_.size() && t_[i_] == '}') return ++i_, v;
            fail("',' or '}'");
        }
    }

    Value array(int depth) {
        Value v = Value::array();
        ++i_;
        ws();
        if (i_ < t_.size() && t_[i_] == ']') return ++i_, v;
        for (;;) {
            ws();
            v.arr_.push_back(value(depth));
            ws();
            if (i_ < t_.size() && t_[i_] == ',') { ++i_; continue; }
            if (i_ < t_.size() && t_[i_] == ']') return ++i_, v;
            fail("',' or ']'");
        }
    }

    uint32_t hex4() {
        if (i_ + 4 > t_.size()) fail("4 hex digits after \\u");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = t_[i_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
            else { --i_; fail("a hex digit in \\u escape"); }
        }
        return v;
    }

    std::string string() {
        ++i_;  // opening quote
        std::string out;
        for (;;) {
            if (i_ >= t_.size()) fail("closing '\"'");
            const unsigned char c = (unsigned char)t_[i_];
            if (c == '"') return ++i_, out;
            if (c < 0x20) fail("no raw control character in a string (escape it)");
            if (c == '\\') {
                if (++i_ >= t_.size()) fail("an escape character");
                const char e = t_[i_++];
                switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xDC00 && cp <= 0xDFFF) { i_ -= 6; fail("no lone low surrogate"); }
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (!(i_ + 2 <= t_.size() && t_[i_] == '\\' && t_[i_ + 1] == 'u'))
                            fail("a \\u low surrogate after a high surrogate");
                        i_ += 2;
                        const uint32_t lo = hex4();
                        if (lo < 0xDC00 || lo > 0xDFFF) { i_ -= 6; fail("a low surrogate (DC00-DFFF)"); }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: --i_; fail("one of \" \\ / b f n r t u after '\\'");
                }
                continue;
            }
            const int n = utf8_len(t_, i_);
            if (n == 0) fail("valid UTF-8");
            out.append(t_.substr(i_, (size_t)n));
            i_ += (size_t)n;
        }
    }

    Value number() {
        const size_t start = i_;
        bool is_float = false;
        if (t_[i_] == '-') ++i_;
        const auto digits = [&] {
            const size_t s = i_;
            while (i_ < t_.size() && t_[i_] >= '0' && t_[i_] <= '9') ++i_;
            return i_ - s;
        };
        if (i_ < t_.size() && t_[i_] == '0') ++i_;
        else if (digits() == 0) fail("a digit");
        if (i_ < t_.size() && t_[i_] == '.') {
            ++i_, is_float = true;
            if (digits() == 0) fail("a digit after '.'");
        }
        if (i_ < t_.size() && (t_[i_] == 'e' || t_[i_] == 'E')) {
            ++i_, is_float = true;
            if (i_ < t_.size() && (t_[i_] == '+' || t_[i_] == '-')) ++i_;
            if (digits() == 0) fail("a digit in the exponent");
        }
        const std::string_view lit = t_.substr(start, i_ - start);
        Value v;
        double d = 0;
        const auto [end, ec] = std::from_chars(lit.data(), lit.data() + lit.size(), d);
        if (ec == std::errc::result_out_of_range) {
            i_ = start;
            fail("a number within the double range");
        }
        STRIX_CHECK(ec == std::errc() && end == lit.data() + lit.size(), "json: from_chars refused '", lit, "'");
        if (is_float) {
            v.type_ = Type::Float, v.d_ = d;
        } else {
            v.type_ = Type::Int, v.d_ = d;
            v.s_ = lit == "-0" ? "0" : std::string(lit);  // Python: int("-0") prints 0
        }
        return v;
    }
};

Value Value::parse(std::string_view text, int max_depth) { return Parser(text, max_depth).run(); }

// ---- printers ----

void append_quoted(std::string &out, std::string_view s) {
    static const char *hex = "0123456789abcdef";
    out += '"';
    for (const char ch : s) {
        const unsigned char c = (unsigned char)ch;
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        default:
            if (c < 0x20) out += "\\u00", out += hex[c >> 4], out += hex[c & 15];
            else out += ch;
        }
    }
    out += '"';
}

std::string python_float_repr(double v) {
    STRIX_CHECK(std::isfinite(v), "python_float_repr: ", v, " is not finite");
    if (v == 0) return std::signbit(v) ? "-0.0" : "0.0";
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);  // shortest round trip
    STRIX_CHECK(r.ec == std::errc(), "python_float_repr: to_chars failed");
    const std::string sci(buf, r.ptr);  // "-d.ddde+XX"
    const size_t epos = sci.find('e');
    const bool neg = sci[0] == '-';
    std::string digits;
    for (size_t k = neg ? 1 : 0; k < epos; ++k)
        if (sci[k] != '.') digits += sci[k];
    const int e = std::stoi(sci.substr(epos + 1));
    std::string out = neg ? "-" : "";
    if (e >= -4 && e < 16) {  // Python: fixed when -4 <= exponent < 16
        if (e < 0) {
            out += "0." + std::string((size_t)(-e - 1), '0') + digits;
        } else if ((size_t)e + 1 >= digits.size()) {
            out += digits + std::string((size_t)e + 1 - digits.size(), '0') + ".0";
        } else {
            out += digits.substr(0, (size_t)e + 1) + "." + digits.substr((size_t)e + 1);
        }
    } else {
        out += digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        char eb[16];
        std::snprintf(eb, sizeof eb, "e%c%02d", e < 0 ? '-' : '+', e < 0 ? -e : e);
        out += eb;
    }
    return out;
}

void Value::dump_to(std::string &out, bool python) const {
    switch (type_) {
    case Type::Null: out += "null"; break;
    case Type::Bool: out += b_ ? "true" : "false"; break;
    case Type::Int: out += s_; break;
    case Type::Float: out += python_float_repr(d_); break;  // keeps a '.' or exponent: re-parses as a Float
    case Type::String: append_quoted(out, s_); break;
    case Type::Array:
        out += '[';
        for (size_t k = 0; k < arr_.size(); ++k) {
            if (k) out += python ? ", " : ",";
            arr_[k].dump_to(out, python);
        }
        out += ']';
        break;
    case Type::Object:
        out += '{';
        for (size_t k = 0; k < obj_.size(); ++k) {
            if (k) out += python ? ", " : ",";
            append_quoted(out, obj_[k].first);
            out += python ? ": " : ":";
            obj_[k].second.dump_to(out, python);
        }
        out += '}';
        break;
    }
}

std::string Value::dump() const {
    std::string out;
    dump_to(out, false);
    return out;
}

std::string Value::dump_python() const {
    std::string out;
    dump_to(out, true);
    return out;
}

}  // namespace strix::json
