#pragma once

// Generated tokens -> what the API returns, incrementally for streaming.
// The prompt ends inside the think block ("<think>\n"), so generation starts as reasoning until the </think>
// token, then content; a <tool_call> token opens a call that runs to </tool_call> and is parsed from the model's
// XML form (<function=NAME> <parameter=P> value </parameter> ... </function>) into a name and JSON arguments,
// typed from the tool's JSON schema. Markers count only as the single special tokens, never as text the model
// spelled out, and only in their phase (a <tool_call> inside reasoning is reasoning text).
//
// Text is released in whole UTF-8 characters (a token can end inside one). Like the chat template, which trims
// reasoning and content when the turn is re-rendered, leading and trailing white space of each channel is
// dropped: trailing white space is held back until more text follows. Stop strings apply to content; text that
// could still become a stop string is held back until it can't.

#include "serve/json.hpp"
#include "serve/tokenizer.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace strix {

struct ToolCallOut {
    std::string id, name;
    json::Value arguments;  // an object
};

struct OutputEvent {
    enum class Kind { Reasoning, Content, ToolCall } kind;
    std::string text;  // Reasoning / Content
    ToolCallOut call;  // ToolCall
};

// Parses the text between <tool_call> and </tool_call>. tools: the request's tools (or nullptr), for typing
// parameter values: a parameter whose schema type is "string" (or that isn't in the schema) stays the raw
// string; any other is parsed as JSON, and kept as the raw string if it isn't JSON. Returns false (out
// untouched) if the text isn't a well-formed call.
bool parse_tool_call(const std::string &body, const json::Value *tools, ToolCallOut &out);

class OutputParser {
public:
    struct Config {
        bool thinking = true;           // the prompt left generation inside the think block
        const json::Value *tools = nullptr;
        std::vector<std::string> stop;  // non-empty strings
        uint64_t id_seed = 0;           // tool call ids
        // Trim each channel's leading / trailing white space (chat, like the template's re-render). Off for raw
        // /v1/completions, where it's part of the text (code indentation, a continuation's leading space).
        bool trim_ws = true;
    };
    OutputParser(const Tokenizer &tok, Config cfg);

    // One generated token (never an end-of-sequence token: the caller stops on those). Returns true when a stop
    // string completed - generation should end; this token's text past the stop string is dropped.
    bool feed(int32_t id, std::vector<OutputEvent> &out);
    // Generation over: releases held text (less trailing white space). A tool call still open is dropped
    // (dropped_partial_call() says so).
    void finish(std::vector<OutputEvent> &out);

    int64_t reasoning_tokens() const { return reasoning_tokens_; }
    bool in_reasoning() const { return phase_ == Phase::Reasoning; }
    int64_t tool_calls() const { return tool_calls_; }
    bool dropped_partial_call() const { return dropped_call_; }

private:
    // One text channel: bytes in, whole characters out, leading / trailing white space trimmed, stop strings.
    struct Channel {
        OutputEvent::Kind kind;
        std::string bytes;    // not yet whole characters
        std::string pending;  // whole characters held back (trailing white space, possible stop-string start)
        bool started = false; // a non-white-space character went out
    };
    void add(Channel &c, const std::string &bytes, std::vector<OutputEvent> &out, bool *stopped);
    void release(Channel &c, std::vector<OutputEvent> &out, bool drop_trailing_ws);

    const Tokenizer &tok_;
    Config cfg_;
    int32_t think_end_, call_begin_, call_end_;
    enum class Phase { Reasoning, Content, Call } phase_;
    Channel reasoning_{OutputEvent::Kind::Reasoning, {}, {}, false}, content_{OutputEvent::Kind::Content, {}, {}, false};
    std::string call_body_;
    int64_t reasoning_tokens_ = 0, tool_calls_ = 0;
    bool dropped_call_ = false, stopped_ = false;
    size_t max_stop_ = 0;
    std::mt19937_64 rng_;
};

}  // namespace strix
