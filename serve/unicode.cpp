#include "serve/unicode.hpp"

#include "common/check.hpp"

#include <algorithm>

namespace strix::unicode {

namespace {
struct CpRange {
    uint32_t first, last;
    uint8_t value;
};
struct Decomp {
    uint32_t cp, offset, length;
};
struct Compose {
    uint32_t first, second, composite;
};
#include "serve/unicode_tables.inc"

template <size_t N>
uint8_t range_lookup(const CpRange (&table)[N], uint32_t cp) {
    const auto it = std::upper_bound(std::begin(table), std::end(table), cp,
                                     [](uint32_t c, const CpRange &r) { return c < r.first; });
    if (it == std::begin(table)) return 0;
    const CpRange &r = *(it - 1);
    return cp <= r.last ? r.value : 0;
}

// Hangul syllables (Unicode ch. 3.12): algorithmic decomposition / composition.
constexpr uint32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr uint32_t kLCount = 19, kVCount = 21, kTCount = 28, kNCount = kVCount * kTCount, kSCount = kLCount * kNCount;

void decompose(uint32_t cp, std::vector<uint32_t> &out) {
    if (cp >= kSBase && cp < kSBase + kSCount) {
        const uint32_t s = cp - kSBase;
        out.push_back(kLBase + s / kNCount);
        out.push_back(kVBase + (s % kNCount) / kTCount);
        if (s % kTCount) out.push_back(kTBase + s % kTCount);
        return;
    }
    const auto it = std::lower_bound(std::begin(kDecomp), std::end(kDecomp), cp,
                                     [](const Decomp &d, uint32_t c) { return d.cp < c; });
    if (it != std::end(kDecomp) && it->cp == cp) out.insert(out.end(), kDecompData + it->offset, kDecompData + it->offset + it->length);
    else out.push_back(cp);
}

// 0 if (a, b) has no primary composite.
uint32_t compose_pair(uint32_t a, uint32_t b) {
    if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount)
        return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
    if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 && b > kTBase && b < kTBase + kTCount)
        return a + (b - kTBase);
    const auto it = std::lower_bound(std::begin(kCompose), std::end(kCompose), std::pair{a, b},
                                     [](const Compose &c, const std::pair<uint32_t, uint32_t> &k) {
                                         return c.first != k.first ? c.first < k.first : c.second < k.second;
                                     });
    return it != std::end(kCompose) && it->first == a && it->second == b ? it->composite : 0;
}
}  // namespace

const char *classes_version() { return kClassesVersion; }
const char *nfc_version() { return kNfcVersion; }

std::vector<uint32_t> decode_utf8(std::string_view s, const char *what) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    static const uint32_t kMin[5] = {0, 0, 0x80, 0x800, 0x10000};
    for (size_t i = 0; i < s.size();) {
        const unsigned c = (unsigned char)s[i];
        int n;
        uint32_t cp;
        if (c < 0x80) n = 1, cp = c;
        else if ((c & 0xE0) == 0xC0) n = 2, cp = c & 0x1F;
        else if ((c & 0xF0) == 0xE0) n = 3, cp = c & 0x0F;
        else if ((c & 0xF8) == 0xF0) n = 4, cp = c & 0x07;
        else STRIX_FAIL(what, ": invalid UTF-8 lead byte 0x", std::hex, c, " at byte ", std::dec, i);
        STRIX_CHECK(i + (size_t)n <= s.size(), what, ": UTF-8 sequence cut off at byte ", i);
        for (int k = 1; k < n; ++k) {
            const unsigned b = (unsigned char)s[i + (size_t)k];
            STRIX_CHECK((b & 0xC0) == 0x80, what, ": invalid UTF-8 continuation byte at ", i + (size_t)k);
            cp = (cp << 6) | (b & 0x3F);
        }
        STRIX_CHECK(cp >= kMin[n] && cp <= 0x10FFFF && !(cp >= 0xD800 && cp <= 0xDFFF), what,
                    ": overlong / surrogate / out-of-range UTF-8 (U+", std::hex, cp, std::dec, ") at byte ", i);
        out.push_back(cp);
        i += (size_t)n;
    }
    return out;
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

std::string encode_utf8(const std::vector<uint32_t> &cps, size_t begin, size_t end) {
    STRIX_CHECK(begin <= end && end <= cps.size(), "encode_utf8: range [", begin, ", ", end, ") of ", cps.size());
    std::string out;
    out.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) append_utf8(out, cps[i]);
    return out;
}

uint8_t char_class(uint32_t cp) { return range_lookup(kCatRanges, cp); }
uint8_t combining_class(uint32_t cp) { return range_lookup(kCccRanges, cp); }

bool is_white_space(uint32_t cp) {
    return (cp >= 0x9 && cp <= 0xD) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

bool python_isspace(uint32_t cp) { return is_white_space(cp) || (cp >= 0x1C && cp <= 0x1F); }

std::vector<uint32_t> nfc(const std::vector<uint32_t> &cps) {
    // Fast path: ASCII (and Latin-1 without combining marks) is already NFC.
    if (std::all_of(cps.begin(), cps.end(), [](uint32_t c) { return c < 0x300; })) return cps;
    // 1. Full canonical decomposition, 2. canonical ordering of each run of non-starters (stable by class).
    std::vector<uint32_t> d;
    d.reserve(cps.size() + 8);
    for (uint32_t c : cps) decompose(c, d);
    std::vector<uint8_t> cc(d.size());
    for (size_t i = 0; i < d.size(); ++i) cc[i] = combining_class(d[i]);
    for (size_t i = 0; i < d.size();) {
        if (cc[i] == 0) { ++i; continue; }
        size_t j = i;
        while (j < d.size() && cc[j] != 0) ++j;
        std::vector<std::pair<uint8_t, uint32_t>> run;
        for (size_t k = i; k < j; ++k) run.emplace_back(cc[k], d[k]);
        std::stable_sort(run.begin(), run.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
        for (size_t k = i; k < j; ++k) cc[k] = run[k - i].first, d[k] = run[k - i].second;
        i = j;
    }
    // 3. Canonical composition: each character after the last starter combines with it unless blocked (a
    // character in between with class 0 or >= its own).
    std::vector<uint32_t> out;
    std::vector<uint8_t> occ;
    out.reserve(d.size());
    long starter = -1;
    for (size_t i = 0; i < d.size(); ++i) {
        const uint32_t c = d[i];
        const uint8_t k = cc[i];
        if (starter >= 0) {
            const uint8_t last = out.size() - 1 == (size_t)starter ? 0 : occ.back();
            const bool blocked = out.size() - 1 != (size_t)starter && (last == 0 || last >= k);
            if (!blocked) {
                const uint32_t comp = compose_pair(out[(size_t)starter], c);
                if (comp) {
                    out[(size_t)starter] = comp;
                    continue;
                }
            }
        }
        out.push_back(c);
        occ.push_back(k);
        if (k == 0) starter = (long)out.size() - 1;
    }
    return out;
}

std::string python_strip(std::string_view s) {
    const std::vector<uint32_t> cps = decode_utf8(s, "python_strip");
    size_t b = 0, e = cps.size();
    while (b < e && python_isspace(cps[b])) ++b;
    while (e > b && python_isspace(cps[e - 1])) --e;
    return encode_utf8(cps, b, e);
}

}  // namespace strix::unicode
