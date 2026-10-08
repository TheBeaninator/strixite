#pragma once

// JSON schema -> byte grammar (serve/grammar), for structured output (`response_format` json_object /
// json_schema). The grammar matches exactly the JSON texts the compiler can promise
// are valid against the schema; a schema keyword whose constraint it can't enforce is an error naming the keyword
// and its path - never silently unenforced.
//
// Enforced: type (one or a list), properties, required, additionalProperties (false, true or absent - see below),
// items (one schema), minItems / maxItems, minLength / maxLength (characters = code points; an escaped surrogate pair
// is one; bounds up to kMaxLengthBound - 2026-10-05, Hermes Agent's own tests asked for maxLength), enum, const, anyOf, oneOf (as anyOf: exclusivity not checked), $ref to "#", "#/$defs/NAME"
// or "#/definitions/NAME" ($defs / definitions at the root only), and `true` / `{}` (any value).
// Ignored (annotations, no constraint): description, title, default, examples, $comment, $schema, $id, deprecated,
// readOnly, writeOnly.
// Refused (an error naming the keyword): every other keyword - pattern, format, minimum, maximum, multipleOf,
// uniqueItems, allOf, not, if / then / else, prefixItems, patternProperties, ... - and a schema of `false`.
//
// Choices the output can't tell from the schema, each stricter than the schema requires (the output still
// validates):
//   - an object with `properties` emits them in the schema's order, required ones always, optional ones or not,
//     and never a key outside `properties` (additionalProperties absent or true is treated as false; a schema
//     value for it is refused). An object schema without `properties` allows any members.
//   - enum / const values are emitted as their compact JSON (serve/json dump()), byte for byte.
//   - numbers: at most kMaxIntegerDigits digits before the point, kMaxFractionDigits after it,
//     kMaxExponentDigits in the exponent; no leading zeros, "-0" allowed.
//   - white space: at most kMaxWhitespaceRun bytes (space, tab, newline, carriage return) at each place JSON allows
//     it, including before and after the top-level value - room for pretty printing, never a runaway of blanks.
//   - nesting: the matcher's stack bound (grammar::kMaxStackDepth) limits how deep output from a recursive schema
//     can nest; the schema itself may nest at most kMaxSchemaDepth levels.

#include "serve/grammar.hpp"
#include "serve/json.hpp"

#include <memory>
#include <string>

namespace strix::json_schema {

constexpr int kMaxWhitespaceRun = 32;
constexpr int kMaxIntegerDigits = 19;
constexpr int kMaxFractionDigits = 20;
constexpr int kMaxExponentDigits = 2;  // e99: 19 + 20 digits stay inside the double range (3 digits: 1e999 = inf; json::Value::parse refuses it)
constexpr int kMaxSchemaDepth = 32;
// The largest minLength / maxLength / minItems / maxItems enforced: a bound is counted out in grammar rules (one per
// position), and every position inside a bounded string is its own automaton state with its own token mask.
constexpr int64_t kMaxLengthBound = 4096;

// `response_format: {"type": "json_object"}`: any JSON object.
std::shared_ptr<const grammar::Grammar> compile_any_object();

// `response_format: {"type": "json_schema", "json_schema": {"schema": ...}}`. `path` names the schema in errors
// ("response_format.json_schema.schema"). Throws strix::Error on a schema it can't enforce or that is invalid.
// The returned grammar is validated.
std::shared_ptr<const grammar::Grammar> compile(const json::Value &schema, const std::string &path);

}  // namespace strix::json_schema
