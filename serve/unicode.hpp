#pragma once

// The Unicode the tokenizer and the chat template need, from tables generated off Python's unicodedata
// (reference/gen_unicode_tables.py -> serve/unicode_tables.inc): UTF-8 <-> code points, the pre-tokenizer's
// letter / mark / number classes and white space, NFC, and Python's str.isspace (Jinja's `trim`). The classes and
// NFC come from different Unicode versions on purpose - the ones the reference tokenizer uses (see the script).

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace strix::unicode {

const char *classes_version();  // Unicode version of char_class()
const char *nfc_version();      // Unicode version of nfc() / combining_class()

// Throws (naming `what` and the byte offset) on malformed UTF-8.
std::vector<uint32_t> decode_utf8(std::string_view s, const char *what);
void append_utf8(std::string &out, uint32_t cp);
std::string encode_utf8(const std::vector<uint32_t> &cps, size_t begin, size_t end);

constexpr uint8_t kLetter = 1, kMark = 2, kNumber = 4;
uint8_t char_class(uint32_t cp);  // kLetter | kMark | kNumber bits (general category L*, M*, N*), 0 otherwise
// Regex \s: the White_Space property (tab..CR, space, NEL, NBSP, U+1680, U+2000-200A, U+2028/9, U+202F, U+205F,
// U+3000).
bool is_white_space(uint32_t cp);
// Python's str.isspace (what str.strip / Jinja's trim remove): White_Space plus U+001C-001F.
bool python_isspace(uint32_t cp);
uint8_t combining_class(uint32_t cp);

// Canonical composition (NFC) of a code point sequence.
std::vector<uint32_t> nfc(const std::vector<uint32_t> &cps);

// Python's str.strip() on UTF-8 text (input must be valid UTF-8).
std::string python_strip(std::string_view s);

}  // namespace strix::unicode
