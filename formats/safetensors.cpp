#include "formats/safetensors.hpp"

#include "common/check.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

namespace strix {

const char *dtype_name(Dtype d) {
    switch (d) {
        case Dtype::F64: return "F64";
        case Dtype::F32: return "F32";
        case Dtype::F16: return "F16";
        case Dtype::BF16: return "BF16";
        case Dtype::I64: return "I64";
        case Dtype::I32: return "I32";
        case Dtype::I16: return "I16";
        case Dtype::I8: return "I8";
        case Dtype::U8: return "U8";
        case Dtype::Bool: return "BOOL";
        default: return "UNKNOWN";
    }
}

size_t dtype_size(Dtype d) {
    switch (d) {
        case Dtype::F64: return 8;
        case Dtype::F32: return 4;
        case Dtype::F16: return 2;
        case Dtype::BF16: return 2;
        case Dtype::I64: return 8;
        case Dtype::I32: return 4;
        case Dtype::I16: return 2;
        case Dtype::I8: return 1;
        case Dtype::U8: return 1;
        case Dtype::Bool: return 1;
        default: return 0;
    }
}

static Dtype dtype_from_string(const std::string &s) {
    static const std::pair<const char *, Dtype> table[] = {
        {"F64", Dtype::F64}, {"F32", Dtype::F32}, {"F16", Dtype::F16}, {"BF16", Dtype::BF16},
        {"I64", Dtype::I64}, {"I32", Dtype::I32}, {"I16", Dtype::I16}, {"I8", Dtype::I8},
        {"U8", Dtype::U8},   {"BOOL", Dtype::Bool},
    };
    for (auto &[name, d] : table)
        if (s == name) return d;
    return Dtype::Unknown;
}

int64_t TensorInfo::numel() const {
    int64_t n = 1;
    for (int64_t d : shape) {
        STRIX_CHECK(d >= 0, "tensor '", name, "' has negative dimension ", d, " in shape ", list_str(shape));
        STRIX_CHECK(d == 0 || n <= std::numeric_limits<int64_t>::max() / d, "tensor '", name,
                    "' element count overflows int64 for shape ", list_str(shape));
        n *= d;
    }
    return n;
}

// --- minimal JSON, scoped to exactly the safetensors header grammar:
// a top-level object whose values are either a tensor descriptor object
// ("dtype": string, "shape": [int...], "data_offsets": [int, int]) or, for
// the optional "__metadata__" key, a flat object of string -> string. Not a
// general-purpose JSON library; deliberately just enough to parse this.
namespace {

struct JsonValue {
    // Object entries are JsonValues too, carrying their own key - not
    // std::pair<string, JsonValue>: libstdc++'s pair uses a
    // conditionally-explicit constructor whose SFINAE needs JsonValue
    // complete earlier than this self-referential definition allows.
    // vector<JsonValue> alone is fine (P0307); folding the key in avoids
    // needing a second incomplete-type-holding template at all.
    enum class Kind { Null, Bool, Number, String, Array, Object } kind = Kind::Null;
    std::string key;  // set only for elements of another JsonValue's `obj`
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<JsonValue> arr;
    std::vector<JsonValue> obj;

    const JsonValue *get(const std::string &k) const {
        for (auto &v : obj)
            if (v.key == k) return &v;
        return nullptr;
    }
};

const char *kind_name(JsonValue::Kind k) {
    switch (k) {
        case JsonValue::Kind::Null: return "null";
        case JsonValue::Kind::Bool: return "bool";
        case JsonValue::Kind::Number: return "number";
        case JsonValue::Kind::String: return "string";
        case JsonValue::Kind::Array: return "array";
        case JsonValue::Kind::Object: return "object";
    }
    return "?";
}

class JsonParser {
public:
    JsonParser(const char *data, size_t len, const std::string &path)
        : begin_(data), p_(data), end_(data + len), path_(path) {}

    JsonValue parse() {
        skip_ws();
        JsonValue v = parse_value(0);
        skip_ws();
        // The spec allows the header to be padded with trailing spaces; anything else is corruption.
        if (p_ != end_) parse_error("unexpected trailing content after the JSON header");
        return v;
    }

private:
    static constexpr int kMaxDepth = 8;  // real headers nest 3 deep; bounds recursion on corrupt input
    const char *begin_;
    const char *p_;
    const char *end_;
    const std::string &path_;

    [[noreturn]] void parse_error(const std::string &what) const {
        size_t off = (size_t)(p_ - begin_);
        std::string near(p_, (size_t)std::min<ptrdiff_t>(end_ - p_, 24));
        STRIX_FAIL("safetensors header of '", path_, "': ", what, " at header byte ", off, " (file byte ", off + 8,
                   "), near \"", near, "\"");
    }

    void skip_ws() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_;
    }
    char peek() const { return p_ < end_ ? *p_ : '\0'; }
    char get() {
        if (p_ >= end_) parse_error("unexpected end of header");
        return *p_++;
    }
    void expect(char c) {
        if (peek() != c) parse_error(std::string("expected '") + c + "'");
        ++p_;
    }
    void expect_lit(const char *lit) {
        for (const char *q = lit; *q; ++q)
            if (get() != *q) parse_error(std::string("malformed literal, expected '") + lit + "'");
    }

    JsonValue parse_value(int depth) {
        if (depth > kMaxDepth) parse_error("nesting deeper than " + std::to_string(kMaxDepth));
        skip_ws();
        char c = peek();
        if (c == '{') return parse_object(depth);
        if (c == '[') return parse_array(depth);
        if (c == '"') {
            JsonValue v;
            v.kind = JsonValue::Kind::String;
            v.str = parse_string();
            return v;
        }
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') {
            expect_lit("null");
            return JsonValue{};
        }
        if (c == '-' || std::isdigit((unsigned char)c)) return parse_number();
        parse_error("expected a JSON value");
    }

    JsonValue parse_bool() {
        JsonValue v;
        v.kind = JsonValue::Kind::Bool;
        if (peek() == 't') {
            expect_lit("true");
            v.b = true;
        } else {
            expect_lit("false");
            v.b = false;
        }
        return v;
    }

    static void append_utf8(std::string &s, unsigned cp) {
        if (cp < 0x80) {
            s += (char)cp;
        } else if (cp < 0x800) {
            s += (char)(0xC0 | (cp >> 6));
            s += (char)(0x80 | (cp & 0x3F));
        } else {
            s += (char)(0xE0 | (cp >> 12));
            s += (char)(0x80 | ((cp >> 6) & 0x3F));
            s += (char)(0x80 | (cp & 0x3F));
        }
    }

    unsigned parse_hex4() {
        unsigned v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = get();
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
            else parse_error("bad \\u escape (expected 4 hex digits)");
        }
        return v;
    }

    std::string parse_string() {
        expect('"');
        std::string s;
        while (true) {
            char c = get();
            if (c == '"') break;
            if ((unsigned char)c < 0x20) parse_error("unescaped control character in string");
            if (c == '\\') {
                char e = get();
                switch (e) {
                    case '"': s += '"'; break;
                    case '\\': s += '\\'; break;
                    case '/': s += '/'; break;
                    case 'b': s += '\b'; break;
                    case 'f': s += '\f'; break;
                    case 'n': s += '\n'; break;
                    case 'r': s += '\r'; break;
                    case 't': s += '\t'; break;
                    case 'u': append_utf8(s, parse_hex4()); break;
                    default: parse_error(std::string("bad escape '\\") + e + "'");
                }
            } else {
                s += c;
            }
        }
        return s;
    }

    JsonValue parse_number() {
        const char *start = p_;
        while (p_ < end_ && (std::isdigit((unsigned char)peek()) || peek() == '.' || peek() == 'e' ||
                             peek() == 'E' || peek() == '+' || peek() == '-'))
            ++p_;
        std::string tok(start, p_);
        char *endp = nullptr;
        errno = 0;
        double v = std::strtod(tok.c_str(), &endp);
        if (tok.empty() || endp != tok.c_str() + tok.size() || errno == ERANGE || !std::isfinite(v)) {
            p_ = start;
            parse_error("malformed number '" + tok + "'");
        }
        JsonValue out;
        out.kind = JsonValue::Kind::Number;
        out.num = v;
        return out;
    }

    JsonValue parse_array(int depth) {
        expect('[');
        JsonValue v;
        v.kind = JsonValue::Kind::Array;
        skip_ws();
        if (peek() == ']') {
            ++p_;
            return v;
        }
        while (true) {
            v.arr.push_back(parse_value(depth + 1));
            skip_ws();
            char c = get();
            if (c == ']') break;
            if (c != ',') {
                --p_;
                parse_error("expected ',' or ']' in array");
            }
        }
        return v;
    }

    JsonValue parse_object(int depth) {
        expect('{');
        JsonValue v;
        v.kind = JsonValue::Kind::Object;
        skip_ws();
        if (peek() == '}') {
            ++p_;
            return v;
        }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            if (v.get(key)) parse_error("duplicate key '" + key + "'");
            skip_ws();
            expect(':');
            JsonValue val = parse_value(depth + 1);
            val.key = std::move(key);
            v.obj.push_back(std::move(val));
            skip_ws();
            char c = get();
            if (c == '}') break;
            if (c != ',') {
                --p_;
                parse_error("expected ',' or '}' in object");
            }
        }
        return v;
    }
};

// Unmaps on scope exit unless released - keeps every early throw below leak-free.
struct MappingGuard {
    void *p;
    size_t n;
    ~MappingGuard() {
        if (p) munmap(p, n);
    }
};

// A JSON number that must be an exact non-negative integer (shape dims, offsets).
uint64_t as_index(const JsonValue &v, const std::string &what) {
    STRIX_CHECK(v.kind == JsonValue::Kind::Number, what, " must be a number, got ", kind_name(v.kind));
    STRIX_CHECK(v.num >= 0 && v.num == std::floor(v.num), what, " must be a non-negative integer, got ", v.num);
    STRIX_CHECK(v.num <= 9007199254740992.0, what, " = ", v.num, " exceeds 2^53 and can't be represented exactly");
    return (uint64_t)v.num;
}

}  // namespace

SafetensorsFile::SafetensorsFile(const std::string &path) : path_(path) {
    STRIX_CHECK(!path.empty(), "safetensors path is empty");
    int fd = open(path.c_str(), O_RDONLY);
    STRIX_CHECK(fd >= 0, "cannot open safetensors file '", path, "': ", std::strerror(errno));

    struct stat st{};
    int stat_rc = fstat(fd, &st);
    int stat_errno = errno;
    if (stat_rc != 0) close(fd);
    STRIX_CHECK(stat_rc == 0, "fstat failed on '", path, "': ", std::strerror(stat_errno));
    if (!S_ISREG(st.st_mode)) close(fd);
    STRIX_CHECK(S_ISREG(st.st_mode), "'", path, "' is not a regular file");
    file_size_ = (size_t)st.st_size;
    if (file_size_ < 8) close(fd);
    STRIX_CHECK(file_size_ >= 8, "'", path, "' is ", file_size_,
                " bytes; a safetensors file needs at least the 8-byte header length");

    map_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    int map_errno = errno;
    close(fd);  // the mapping keeps the file referenced
    if (map_ == MAP_FAILED) map_ = nullptr;
    STRIX_CHECK(map_ != nullptr, "mmap of '", path, "' (", file_size_, " bytes) failed: ", std::strerror(map_errno));
    MappingGuard guard{map_, file_size_};

    const auto *base = static_cast<const unsigned char *>(map_);
    uint64_t header_len;
    std::memcpy(&header_len, base, 8);  // little-endian per spec; this host is x86-64
    // The format caps the header at 100 MB; a larger value means a corrupt or non-safetensors file.
    constexpr uint64_t kMaxHeader = 100u * 1000 * 1000;
    STRIX_CHECK(header_len >= 2 && header_len <= kMaxHeader, "'", path, "': header length ", header_len,
                " is outside [2, ", kMaxHeader, "] - not a safetensors file, or corrupt");
    STRIX_CHECK(8 + header_len <= file_size_, "'", path, "': header length ", header_len,
                " runs past the end of the ", file_size_, "-byte file (truncated download?)");
    data_start_ = 8 + (size_t)header_len;

    JsonParser parser(reinterpret_cast<const char *>(base + 8), (size_t)header_len, path);
    JsonValue header = parser.parse();
    STRIX_CHECK(header.kind == JsonValue::Kind::Object, "'", path, "': header must be a JSON object, got ",
                kind_name(header.kind));

    size_t data_len = file_size_ - data_start_;
    tensors_.reserve(header.obj.size());
    for (auto &val : header.obj) {
        const std::string &name = val.key;
        if (name == "__metadata__") {
            STRIX_CHECK(val.kind == JsonValue::Kind::Object, "'", path, "': __metadata__ must be an object, got ",
                        kind_name(val.kind));
            continue;
        }
        std::string where = cat("tensor '", name, "' in '", path, "'");
        STRIX_CHECK(val.kind == JsonValue::Kind::Object, where, " must be an object, got ", kind_name(val.kind));
        const JsonValue *dtype_v = val.get("dtype");
        const JsonValue *shape_v = val.get("shape");
        const JsonValue *offsets_v = val.get("data_offsets");
        STRIX_CHECK(dtype_v && shape_v && offsets_v, where, " is missing ",
                    !dtype_v ? "\"dtype\"" : !shape_v ? "\"shape\"" : "\"data_offsets\"");
        STRIX_CHECK(dtype_v->kind == JsonValue::Kind::String, where, ": dtype must be a string, got ",
                    kind_name(dtype_v->kind));
        STRIX_CHECK(shape_v->kind == JsonValue::Kind::Array, where, ": shape must be an array, got ",
                    kind_name(shape_v->kind));
        STRIX_CHECK(offsets_v->kind == JsonValue::Kind::Array && offsets_v->arr.size() == 2, where,
                    ": data_offsets must be a 2-element array [start, end]");

        TensorInfo t;
        t.name = name;
        t.dtype = dtype_from_string(dtype_v->str);
        STRIX_CHECK(t.dtype != Dtype::Unknown, where, " has unsupported dtype \"", dtype_v->str,
                    "\"; supported: F64 F32 F16 BF16 I64 I32 I16 I8 U8 BOOL");
        for (size_t i = 0; i < shape_v->arr.size(); ++i)
            t.shape.push_back((int64_t)as_index(shape_v->arr[i], cat(where, " shape[", i, "]")));
        uint64_t rel_start = as_index(offsets_v->arr[0], where + " data_offsets[0]");
        uint64_t rel_end = as_index(offsets_v->arr[1], where + " data_offsets[1]");
        STRIX_CHECK(rel_start <= rel_end, where, ": data_offsets start ", rel_start, " > end ", rel_end);
        STRIX_CHECK(rel_end <= data_len, where, ": data ends at data-section byte ", rel_end, " but the data section is only ",
                    data_len, " bytes (truncated file?)");
        t.byte_offset = data_start_ + (size_t)rel_start;
        t.byte_length = (size_t)(rel_end - rel_start);
        size_t expected = (size_t)t.numel() * dtype_size(t.dtype);
        STRIX_CHECK(expected == t.byte_length, where, ": shape ", list_str(t.shape), " x ", dtype_name(t.dtype), " (",
                    dtype_size(t.dtype), " B/elem) needs ", expected, " bytes but data_offsets span ", t.byte_length);
        tensors_.push_back(std::move(t));
    }

    // Tensors must not overlap (a corrupt or hand-edited header could alias two tensors' bytes).
    std::vector<const TensorInfo *> by_offset;
    for (auto &t : tensors_) by_offset.push_back(&t);
    std::sort(by_offset.begin(), by_offset.end(),
              [](const TensorInfo *a, const TensorInfo *b) { return a->byte_offset < b->byte_offset; });
    for (size_t i = 1; i < by_offset.size(); ++i) {
        const TensorInfo *prev = by_offset[i - 1], *cur = by_offset[i];
        STRIX_CHECK(prev->byte_offset + prev->byte_length <= cur->byte_offset, "'", path, "': tensor '", cur->name,
                    "' (file bytes ", cur->byte_offset, "..) overlaps tensor '", prev->name, "' (file bytes ",
                    prev->byte_offset, "..", prev->byte_offset + prev->byte_length, ")");
    }

    guard.p = nullptr;  // success: the object owns the mapping now
}

SafetensorsFile::~SafetensorsFile() {
    if (map_) munmap(map_, file_size_);
}

const TensorInfo *SafetensorsFile::find(const std::string &name) const {
    for (auto &t : tensors_)
        if (t.name == name) return &t;
    return nullptr;
}

const TensorInfo &SafetensorsFile::get(const std::string &name) const {
    const TensorInfo *t = find(name);
    STRIX_CHECK(t != nullptr, "tensor '", name, "' not found in '", path_, "' (", tensors_.size(), " tensors)");
    return *t;
}

const void *SafetensorsFile::data(const TensorInfo &t) const {
    STRIX_CHECK(map_ != nullptr, "'", path_, "' is not mapped");
    STRIX_CHECK(t.byte_offset >= data_start_ && t.byte_offset + t.byte_length <= file_size_, "tensor '", t.name,
                "' (file bytes ", t.byte_offset, "..", t.byte_offset + t.byte_length, ") is outside '", path_,
                "' (data section ", data_start_, "..", file_size_, ") - TensorInfo from a different file?");
    return static_cast<const unsigned char *>(map_) + t.byte_offset;
}

}  // namespace strix
