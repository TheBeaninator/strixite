#include "serve/capture.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "common/check.hpp"
#include "serve/json.hpp"

namespace strix {

namespace {
void check_record(const CaptureRecord &rec, const std::string &where) {
    STRIX_CHECK(rec.prompt_n >= 1 && rec.prompt_n <= (int64_t)rec.tokens.size(), where, ": prompt_n ", rec.prompt_n,
                " of ", rec.tokens.size(), " tokens");
    int64_t emitted = 0;
    for (size_t i = 0; i < rec.steps.size(); ++i) {
        const CaptureStep &s = rec.steps[i];
        STRIX_CHECK(s.drafted == kThinkingStopStep || s.accepted <= s.drafted, where, ": step ", i, " accepted ",
                    (int)s.accepted, " of ", (int)s.drafted, " drafts");
        emitted += s.emitted;
    }
    STRIX_CHECK(emitted == (int64_t)rec.tokens.size() - rec.prompt_n, where, ": steps emit ", emitted,
                " tokens, the record holds ", (int64_t)rec.tokens.size() - rec.prompt_n, " generated");
}
}  // namespace

std::string write_capture(const std::string &dir, const CaptureRecord &rec, int64_t unix_ms) {
    const std::string name = std::to_string(unix_ms) + "-req" + std::to_string(rec.id) + ".cap";
    const std::string path = (std::filesystem::path(dir) / name).string();
    check_record(rec, "write_capture " + path);
    json::Value h = json::Value::object();
    h.set("v", json::Value::integer(1));
    h.set("id", json::Value::integer(rec.id));
    h.set("prompt_n", json::Value::integer(rec.prompt_n));
    h.set("tokens_n", json::Value::integer((int64_t)rec.tokens.size()));
    h.set("steps_n", json::Value::integer((int64_t)rec.steps.size()));
    h.set("tools", json::Value::integer(rec.tools));
    h.set("one_shot", json::Value::boolean(rec.one_shot));
    h.set("user_turn", json::Value::boolean(rec.user_turn));
    h.set("finish", json::Value::string(rec.finish));
    h.set("mtp_draft", json::Value::integer(rec.mtp_draft));
    h.set("mtp_margin", json::Value::number(rec.mtp_margin));
    const std::string tmp = path + ".partial";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        STRIX_CHECK(f.good(), "write_capture: can't create ", tmp);
        const std::string line = h.dump() + "\n";
        f.write(line.data(), (std::streamsize)line.size());
        f.write(reinterpret_cast<const char *>(rec.tokens.data()), (std::streamsize)(rec.tokens.size() * 4));
        for (const CaptureStep &s : rec.steps) {
            const char b[3] = {(char)s.drafted, (char)s.accepted, (char)s.emitted};
            f.write(b, 3);
        }
        f.flush();
        STRIX_CHECK(f.good(), "write_capture: writing ", tmp, " failed (disk full?)");
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    STRIX_CHECK(!ec, "write_capture: renaming ", tmp, " to ", path, ": ", ec.message());
    return path;
}

CaptureRecord read_capture(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    STRIX_CHECK(f.good(), "read_capture: can't open ", path);
    std::string line;
    STRIX_CHECK((bool)std::getline(f, line), "read_capture: ", path, " has no header line");
    const json::Value h = json::Value::parse(line);
    STRIX_CHECK(h.find("v") && h.find("v")->as_int("v") == 1, "read_capture: ", path, ": version, expected 1");
    CaptureRecord r;
    r.id = h.find("id")->as_int("id");
    r.prompt_n = h.find("prompt_n")->as_int("prompt_n");
    const int64_t tn = h.find("tokens_n")->as_int("tokens_n"), sn = h.find("steps_n")->as_int("steps_n");
    STRIX_CHECK(tn >= 0 && tn <= (int64_t{1} << 26) && sn >= 0 && sn <= tn, "read_capture: ", path, ": tokens_n ", tn,
                ", steps_n ", sn);
    r.tools = h.find("tools")->as_int("tools");
    r.one_shot = h.find("one_shot")->as_bool("one_shot");
    r.user_turn = h.find("user_turn")->as_bool("user_turn");
    r.finish = h.find("finish")->as_string("finish");
    r.mtp_draft = h.find("mtp_draft")->as_int("mtp_draft");
    r.mtp_margin = h.find("mtp_margin")->as_double("mtp_margin");
    r.tokens.resize((size_t)tn);
    f.read(reinterpret_cast<char *>(r.tokens.data()), (std::streamsize)(tn * 4));
    r.steps.resize((size_t)sn);
    for (CaptureStep &s : r.steps) {
        char b[3];
        f.read(b, 3);
        s.drafted = (uint8_t)b[0], s.accepted = (uint8_t)b[1], s.emitted = (uint8_t)b[2];
    }
    STRIX_CHECK(f.good(), "read_capture: ", path, " is shorter than its header says");
    f.peek();
    STRIX_CHECK(f.eof(), "read_capture: ", path, " has bytes past its steps");
    check_record(r, "read_capture " + path);
    return r;
}

}  // namespace strix
