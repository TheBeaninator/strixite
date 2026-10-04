#include "serve/output_parser.hpp"

#include "common/check.hpp"
#include "serve/unicode.hpp"

#include <algorithm>

namespace strix {

namespace {

// Length of the longest prefix of s made of whole UTF-8 characters (s is otherwise well-formed).
size_t whole_chars(const std::string &s) {
    const size_t n = s.size();
    for (size_t back = 1; back <= std::min<size_t>(3, n); ++back) {
        const unsigned char c = (unsigned char)s[n - back];
        if ((c & 0xC0) == 0x80) continue;  // continuation byte: keep looking for the lead
        const size_t need = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
        return need > back ? n - back : n;
    }
    return n;  // 4+ trailing continuation bytes can't be one character: leave them to fail at decode
}

// Byte offset where s's trailing white space starts (Python isspace), s whole characters.
size_t trailing_ws_start(const std::string &s) {
    const std::vector<uint32_t> cps = unicode::decode_utf8(s, "output text");
    size_t end = 0, at = 0;
    for (uint32_t c : cps) {
        std::string one;
        unicode::append_utf8(one, c);
        at += one.size();
        if (!unicode::python_isspace(c)) end = at;
    }
    return end;
}

void skip_ws(const std::string &s, size_t &i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i;
}

const json::Value *param_schema(const json::Value *tools, const std::string &fn, const std::string &param) {
    if (!tools || !tools->is_array()) return nullptr;
    for (const json::Value &t : tools->as_array("tools")) {
        const json::Value *f = t.find("function");
        const json::Value *name = f ? f->find("name") : nullptr;
        if (!name || !name->is_string() || name->as_string("name") != fn) continue;
        const json::Value *params = f->find("parameters");
        const json::Value *props = params ? params->find("properties") : nullptr;
        return props ? props->find(param) : nullptr;
    }
    return nullptr;
}

bool is_string_typed(const json::Value *schema) {
    if (!schema) return true;  // not in the schema: keep what the model wrote
    const json::Value *type = schema->find("type");
    if (!type) return false;
    if (type->is_string()) return type->as_string("type") == "string";
    if (!type->is_array()) return false;
    // ["string"] or ["string", "null"]: a string; anything else in the list could be JSON.
    bool str = false;
    for (const json::Value &v : type->as_array("type")) {
        if (!v.is_string()) return false;
        const std::string &t = v.as_string("type");
        if (t == "string") str = true;
        else if (t != "null") return false;
    }
    return str;
}

}  // namespace

bool parse_tool_call(const std::string &body, const json::Value *tools, ToolCallOut &out) {
    size_t i = 0;
    skip_ws(body, i);
    static const std::string kFn = "<function=", kParam = "<parameter=", kParamEnd = "</parameter>", kFnEnd = "</function>";
    if (body.compare(i, kFn.size(), kFn) != 0) return false;
    i += kFn.size();
    const size_t name_end = body.find('>', i);
    if (name_end == std::string::npos || name_end == i || body.find('\n', i) < name_end) return false;
    ToolCallOut call;
    call.name = body.substr(i, name_end - i);
    call.arguments = json::Value::object();
    i = name_end + 1;
    for (;;) {
        skip_ws(body, i);
        if (body.compare(i, kFnEnd.size(), kFnEnd) == 0) {
            i += kFnEnd.size();
            skip_ws(body, i);
            if (i != body.size()) return false;
            break;
        }
        if (body.compare(i, kParam.size(), kParam) != 0) return false;
        i += kParam.size();
        const size_t pn_end = body.find('>', i);
        if (pn_end == std::string::npos || pn_end == i || body.find('\n', i) < pn_end) return false;
        const std::string pname = body.substr(i, pn_end - i);
        i = pn_end + 1;
        const size_t v_end = body.find(kParamEnd, i);
        if (v_end == std::string::npos) return false;
        std::string value = body.substr(i, v_end - i);
        if (!value.empty() && value.front() == '\n') value.erase(0, 1);  // the format's newline after the tag
        if (!value.empty() && value.back() == '\n') value.pop_back();    // ... and before the closing tag
        i = v_end + kParamEnd.size();
        json::Value v = json::Value::string(value);
        if (!is_string_typed(param_schema(tools, call.name, pname))) {
            try {
                v = json::Value::parse(value);
            } catch (const Error &) {
                v = json::Value::string(value);
            }
        }
        call.arguments.set(pname, std::move(v));
    }
    out = std::move(call);
    return true;
}

OutputParser::OutputParser(const Tokenizer &tok, Config cfg)
    : tok_(tok), cfg_(std::move(cfg)), think_end_(tok.id_of("</think>")), call_begin_(tok.id_of("<tool_call>")),
      call_end_(tok.id_of("</tool_call>")), phase_(cfg_.thinking ? Phase::Reasoning : Phase::Content), rng_(cfg_.id_seed) {
    for (const std::string &s : cfg_.stop) {
        STRIX_CHECK(!s.empty() && json::valid_utf8(s), "OutputParser: stop strings must be non-empty UTF-8");
        max_stop_ = std::max(max_stop_, s.size());
    }
}

void OutputParser::add(Channel &c, const std::string &bytes, std::vector<OutputEvent> &out, bool *stopped) {
    c.bytes += bytes;
    const size_t n = whole_chars(c.bytes);
    std::string whole = c.bytes.substr(0, n);
    c.bytes.erase(0, n);
    if (!c.started && cfg_.trim_ws) {
        const std::vector<uint32_t> cps = unicode::decode_utf8(whole, "output text");
        size_t k = 0;
        while (k < cps.size() && unicode::python_isspace(cps[k])) ++k;
        whole = unicode::encode_utf8(cps, k, cps.size());
        if (whole.empty()) return;
        c.started = true;
    }
    c.pending += whole;
    if (stopped && !cfg_.stop.empty()) {
        size_t hit = std::string::npos;
        for (const std::string &s : cfg_.stop) hit = std::min(hit, c.pending.find(s));
        if (hit != std::string::npos) {
            c.pending.erase(hit);
            release(c, out, true);
            *stopped = true;
            return;
        }
    }
    // Hold back the last max_stop - 1 bytes (a stop string may still complete there), and the white space
    // before them (it's trailing if the text ends there).
    size_t cut = c.pending.size();
    if (stopped && max_stop_ > 1) cut -= std::min(cut, max_stop_ - 1);
    while (cut > 0 && cut < c.pending.size() && ((unsigned char)c.pending[cut] & 0xC0) == 0x80) --cut;
    const size_t keep = cfg_.trim_ws ? trailing_ws_start(c.pending.substr(0, cut)) : cut;
    if (keep == 0) return;
    out.push_back({c.kind, c.pending.substr(0, keep), {}});
    c.pending.erase(0, keep);
}

void OutputParser::release(Channel &c, std::vector<OutputEvent> &out, bool drop_trailing_ws) {
    c.bytes.clear();  // an incomplete character at the end can't be shown
    if (drop_trailing_ws && cfg_.trim_ws) c.pending.erase(trailing_ws_start(c.pending));
    if (!c.pending.empty()) out.push_back({c.kind, c.pending, {}});
    c.pending.clear();
}

bool OutputParser::feed(int32_t id, std::vector<OutputEvent> &out) {
    STRIX_CHECK(!stopped_, "OutputParser::feed after a stop string ended generation");
    const std::string &bytes = tok_.token_bytes(id);
    switch (phase_) {
    case Phase::Reasoning:
        ++reasoning_tokens_;
        if (id == think_end_) {
            release(reasoning_, out, true);
            phase_ = Phase::Content;
        } else {
            add(reasoning_, bytes, out, nullptr);
        }
        return false;
    case Phase::Content:
        if (id == call_begin_) {  // held content stays held: trimmed if the call parses, kept if it doesn't
            phase_ = Phase::Call;
            call_body_.clear();
            return false;
        }
        add(content_, bytes, out, &stopped_);
        return stopped_;
    case Phase::Call:
        if (id != call_end_) {
            call_body_ += bytes;
            return false;
        }
        phase_ = Phase::Content;
        if (ToolCallOut call; json::valid_utf8(call_body_) && parse_tool_call(call_body_, cfg_.tools, call)) {
            static const char *kAlnum = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
            call.id = "call_";
            for (int k = 0; k < 24; ++k) call.id += kAlnum[rng_() % 62];
            release(content_, out, true);
            out.push_back({OutputEvent::Kind::ToolCall, {}, std::move(call)});
            ++tool_calls_;
        } else {  // not a call the model formed correctly: show it as the text it is
            add(content_, "<tool_call>" + call_body_ + "</tool_call>", out, &stopped_);
            return stopped_;
        }
        return false;
    }
    return false;
}

void OutputParser::finish(std::vector<OutputEvent> &out) {
    if (phase_ == Phase::Call) dropped_call_ = true;
    release(reasoning_, out, true);
    release(content_, out, true);
}

}  // namespace strix
