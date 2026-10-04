#include "formats/convert_qwen4exp.hpp"

#include "common/check.hpp"
#include "formats/merge_plan.hpp"
#include "formats/q4.hpp"
#include "formats/q5.hpp"
#include "formats/q6.hpp"
#include "formats/q8.hpp"
#include "formats/safetensors.hpp"

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <thread>
#include <type_traits>

namespace strix {

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr size_t kChunkBytes = 256ull << 20;  // source bytes per read (and per parallel quantize)
constexpr int kErrorSampleRows = 32;          // rows per Q4 tensor used for the logged error

// ---------------------------------------------------------------- small helpers

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

std::string fmt_bytes(double b) {
    char s[32];
    if (b >= 1024.0 * 1024 * 1024) std::snprintf(s, sizeof s, "%.2f GiB", b / (1024.0 * 1024 * 1024));
    else if (b >= 1024.0 * 1024) std::snprintf(s, sizeof s, "%.1f MiB", b / (1024.0 * 1024));
    else if (b >= 1024.0) std::snprintf(s, sizeof s, "%.1f KiB", b / 1024.0);
    else std::snprintf(s, sizeof s, "%.0f B", b);
    return s;
}

std::string fmt_dur(double sec) {
    char s[32];
    if (sec < 60) std::snprintf(s, sizeof s, "%.1fs", sec);
    else if (sec < 3600) std::snprintf(s, sizeof s, "%dm%02ds", (int)sec / 60, (int)sec % 60);
    else std::snprintf(s, sizeof s, "%dh%02dm", (int)sec / 3600, ((int)sec % 3600) / 60);
    return s;
}

std::string shape_str(const std::vector<int64_t> &s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? " x " : "") + std::to_string(s[i]);
    return r + "]";
}

std::string hex64(uint64_t v) {
    char s[20];
    std::snprintf(s, sizeof s, "%016llx", (unsigned long long)v);
    return s;
}

// Everything goes to the console (if any) and to convert.log, each line stamped with the elapsed time.
class Log {
public:
    Log(FILE *console, Clock::time_point t0) : console_(console), t0_(t0) {}
    ~Log() {
        if (file_) std::fclose(file_);
    }
    void open_file(const std::string &path) {
        file_ = std::fopen(path.c_str(), "w");
        STRIX_CHECK(file_ != nullptr, "convert: can't open log file '", path, "': ", std::strerror(errno));
    }
    double elapsed() const { return std::chrono::duration<double>(Clock::now() - t0_).count(); }
    __attribute__((format(printf, 2, 3))) void operator()(const char *fmt, ...) {
        char body[2048];
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(body, sizeof body, fmt, ap);
        va_end(ap);
        char line[2100];
        std::snprintf(line, sizeof line, "[%8.1fs] %s\n", elapsed(), body);
        put(line);
    }
    // Visual structure (formatting a few hundred lines costs nothing next to the quantization work).
    void blank() { put("\n"); }
    // "[   t] ━━━ title ━━━━━━…" to a fixed width; heavy for sections, light for sub-sections.
    void rule(const std::string &title, bool heavy = true) {
        const char *bar = heavy ? "\u2501" : "\u2500";
        std::string line = std::string(bar) + bar + bar + " " + title + " ";
        for (int64_t w = display_width(line); w < kRuleWidth; ++w) line += bar;
        (*this)("%s", line.c_str());
    }
    // An explanation, set off with an info mark and word-wrapped under it.
    void note(const std::string &text) {
        std::string cur = "  \u2139  ";
        const std::string indent = "     ";
        size_t at = 0;
        while (at < text.size()) {
            size_t sp = text.find(' ', at);
            if (sp == std::string::npos) sp = text.size();
            const std::string word = text.substr(at, sp - at);
            if (display_width(cur) + (int64_t)word.size() + 1 > kRuleWidth && display_width(cur) > (int64_t)indent.size() + 2) {
                (*this)("%s", cur.c_str());
                cur = indent;
            }
            cur += (cur.back() == ' ' ? "" : " ") + word;
            at = sp + 1;
        }
        (*this)("%s", cur.c_str());
    }

private:
    static constexpr int64_t kRuleWidth = 100;
    static int64_t display_width(const std::string &s) {  // UTF-8 characters, not bytes
        int64_t w = 0;
        for (unsigned char c : s) w += (c & 0xc0) != 0x80;
        return w;
    }
    void put(const char *line) {
        for (FILE *f : {console_, file_})
            if (f) std::fputs(line, f), std::fflush(f);
    }
    FILE *console_ = nullptr, *file_ = nullptr;
    Clock::time_point t0_;
};

// ---------------------------------------------------------------- the source checkpoint

class Source {
public:
    explicit Source(const std::string &dir) : dir_(dir) {
        STRIX_CHECK(fs::is_directory(dir), "convert: source '", dir, "' is not a directory");
        std::vector<std::string> paths;
        for (const auto &e : fs::directory_iterator(dir))
            if (e.path().extension() == ".safetensors") paths.push_back(e.path().string());
        STRIX_CHECK(!paths.empty(), "convert: no *.safetensors in '", dir, "'");
        std::sort(paths.begin(), paths.end());
        std::string fp;  // fingerprint input: index JSON + every shard's name, size and header
        const std::string index_json = dir + "/model.safetensors.index.json";
        if (fs::exists(index_json)) fp += read_small(index_json, 1 << 26);
        for (const std::string &p : paths) {
            files_.push_back(std::make_unique<SafetensorsFile>(p));
            const int fd = ::open(p.c_str(), O_RDONLY | O_CLOEXEC);
            STRIX_CHECK(fd >= 0, "convert: open '", p, "' failed: ", std::strerror(errno));
            fds_.push_back(fd);
            for (const TensorInfo &ti : files_.back()->tensors()) {
                auto [it, fresh] = where_.emplace(ti.name, std::make_pair(files_.size() - 1, &ti));
                STRIX_CHECK(fresh, "convert: tensor '", ti.name, "' is in two shards ('",
                            files_[it->second.first]->path(), "' and '", p, "')");
                bytes_ += ti.byte_length;
            }
            uint64_t hlen = 0;
            STRIX_CHECK(::pread(fd, &hlen, 8, 0) == 8 && hlen < (1ull << 30), "convert: bad header length in '", p,
                        "'");
            std::string h(hlen, '\0');
            STRIX_CHECK(::pread(fd, h.data(), hlen, 8) == (ssize_t)hlen, "convert: short header read in '", p, "'");
            fp += fs::path(p).filename().string() + ":" + std::to_string(files_.back()->file_size()) + ":" + h;
        }
        fingerprint_ = hex64(strix_hash64(fp.data(), fp.size()));
    }
    ~Source() {
        for (int fd : fds_) ::close(fd);
    }
    const std::string &dir() const { return dir_; }
    size_t shards() const { return files_.size(); }
    uint64_t bytes() const { return bytes_; }
    const std::string &fingerprint() const { return fingerprint_; }
    bool has(const std::string &name) const { return where_.count(name) != 0; }
    const TensorInfo &info(const std::string &name) const {
        auto it = where_.find(name);
        STRIX_CHECK(it != where_.end(), "convert: tensor '", name, "' not in the checkpoint (", where_.size(),
                    " tensors in ", files_.size(), " shards of '", dir_, "')");
        return *it->second.second;
    }
    std::vector<std::string> names() const {
        std::vector<std::string> v;
        for (const auto &kv : where_) v.push_back(kv.first);
        return v;
    }
    // bytes [off, off + n) of tensor name into dst; the page cache for that range is dropped afterwards
    // (the converter reads each byte once - caching 335 GiB would only evict everything else).
    void read(const std::string &name, uint64_t off, size_t n, void *dst) const {
        auto it = where_.find(name);
        STRIX_CHECK(it != where_.end(), "convert: read of unknown tensor '", name, "'");
        const TensorInfo &ti = *it->second.second;
        STRIX_CHECK(off <= ti.byte_length && n <= ti.byte_length - off, "convert: read of '", name, "' bytes [", off,
                    ", ", off + n, ") outside its ", ti.byte_length);
        const int fd = fds_[it->second.first];
        auto *p = static_cast<uint8_t *>(dst);
        uint64_t at = ti.byte_offset + off;
        size_t left = n;
        while (left > 0) {
            const ssize_t r = ::pread(fd, p, std::min<size_t>(left, 1u << 30), (off_t)at);
            if (r < 0 && errno == EINTR) continue;
            STRIX_CHECK(r > 0, "convert: pread of '", name, "' from '", files_[it->second.first]->path(), "' at ", at,
                        " failed: ", r < 0 ? std::strerror(errno) : "unexpected end of file");
            p += r, at += (uint64_t)r, left -= (size_t)r;
        }
        (void)::posix_fadvise(fd, (off_t)(ti.byte_offset + off), (off_t)n, POSIX_FADV_DONTNEED);
    }
    template <typename T> std::vector<T> read_all(const std::string &name) const {
        const TensorInfo &ti = info(name);
        STRIX_CHECK(ti.byte_length % sizeof(T) == 0, "convert: '", name, "' is ", ti.byte_length, " bytes");
        std::vector<T> v(ti.byte_length / sizeof(T));
        read(name, 0, ti.byte_length, v.data());
        return v;
    }
    // BF16 or F32 tensor as FP32 (exact).
    std::vector<float> read_f32(const std::string &name) const {
        const TensorInfo &ti = info(name);
        if (ti.dtype == Dtype::F32) return read_all<float>(name);
        STRIX_CHECK(ti.dtype == Dtype::BF16, "convert: '", name, "' is ", dtype_name(ti.dtype), ", expected BF16/F32");
        const std::vector<uint16_t> b = read_all<uint16_t>(name);
        std::vector<float> f(b.size());
        for (size_t i = 0; i < b.size(); ++i) f[i] = bf16_to_f32(b[i]);
        return f;
    }

private:
    static std::string read_small(const std::string &path, size_t limit) {
        FILE *f = std::fopen(path.c_str(), "rb");
        STRIX_CHECK(f != nullptr, "convert: can't open '", path, "'");
        std::string s;
        char buf[65536];
        for (size_t r; (r = std::fread(buf, 1, sizeof buf, f)) > 0 && s.size() < limit;) s.append(buf, r);
        std::fclose(f);
        return s;
    }
    std::string dir_, fingerprint_;
    std::vector<std::unique_ptr<SafetensorsFile>> files_;
    std::vector<int> fds_;
    std::map<std::string, std::pair<size_t, const TensorInfo *>> where_;
    uint64_t bytes_ = 0;
};

// ---------------------------------------------------------------- layout

// Per-class weight formats, the study's class names (reference/quant_study_qwen4exp.py) so a measured variant
// converts as written: "default=q4g64,gdn_in=q8g64,...". Plus mtp_fc (the MTP head's two input projections).
// Only formats a kernel can run are accepted: Q4 everywhere; Q8 for the dense classes linear_q8 covers and the
// HC mixes (hc_mix_down / hc_mix_up have Q8 overloads) - not the routed experts (Q4-only gather/combine).
struct QFmt {
    int bits = 4;
    int64_t G = 64;
    bool operator==(const QFmt &o) const { return bits == o.bits && G == o.G; }
    std::string str() const {
        if (bits == 16) return "bf16";
        return "q" + std::to_string(bits) + "g" + std::to_string(G);
    }
};

const char *const kLayoutClasses[] = {"exp_gu", "exp_down", "sh_gu", "sh_down", "gdn_in", "gdn_out", "attn_qkv",
                                      "attn_o", "hc_down", "hc_up", "ple_kv", "lm_head", "mtp_fc", "embed"};

std::map<std::string, QFmt> parse_layout(const std::string &spec) {
    std::map<std::string, QFmt> by;
    std::optional<QFmt> def;
    size_t at = 0;
    while (at <= spec.size()) {
        const size_t comma = spec.find(',', at);
        const std::string part = spec.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        at = comma == std::string::npos ? spec.size() + 1 : comma + 1;
        const size_t eq = part.find('=');
        STRIX_CHECK(eq != std::string::npos, "convert --layout: '", part, "' is not class=format (in '", spec, "')");
        const std::string cls = part.substr(0, eq), fmt = part.substr(eq + 1);
        STRIX_CHECK(fmt.empty() || (fmt[0] != 'w' && fmt[0] != 'g'), "convert --layout: '", part,
                    "' - calibrated formats (w/g) need calibration statistics; the converter doesn't take them yet");
        QFmt q;
        if (cls == "embed" && fmt == "bf16") {
            q.bits = 16;
            q.G = 0;
        } else {
            const bool ok = fmt.size() >= 4 && fmt[0] == 'q' &&
                            (fmt[1] == '4' || fmt[1] == '5' || fmt[1] == '6' || fmt[1] == '8') && fmt[2] == 'g';
            STRIX_CHECK(ok, "convert --layout: format '", fmt, "' in '", part, "', expected q<4|5|6|8>g<32|64|128>");
            q.bits = fmt[1] - '0';
            q.G = std::stoll(fmt.substr(3));
            STRIX_CHECK(q4_group_size_supported(q.G), "convert --layout: group size ", q.G, " in '", part,
                        "' unsupported (32, 64 or 128)");
        }
        if (cls == "default") {
            def = q;
            continue;
        }
        STRIX_CHECK(std::find_if(std::begin(kLayoutClasses), std::end(kLayoutClasses),
                                 [&](const char *c) { return cls == c; }) != std::end(kLayoutClasses),
                    "convert --layout: unknown class '", cls, "' (router stays BF16; classes: exp_gu exp_down sh_gu ",
                    "sh_down gdn_in gdn_out attn_qkv attn_o hc_down hc_up ple_kv lm_head mtp_fc embed)");
        STRIX_CHECK(by.emplace(cls, q).second, "convert --layout: class '", cls, "' given twice");
    }
    STRIX_CHECK(def.has_value(), "convert --layout '", spec, "' has no default=");
    for (const char *c : kLayoutClasses) {
        if (std::string(c) == "embed") {
            by.emplace(c, QFmt{16, 0});  // embed defaults to BF16
        } else {
            by.emplace(c, *def);
        }
    }
    for (const char *c : {"exp_gu", "exp_down"})
        STRIX_CHECK(by[c].bits == 4 || (by[c].bits == 5 && by[c].G >= 32), "convert --layout: ", c, "=", by[c].str(),
                    " - the experts gather/combine kernels take Q4 or Q5; Q6 / Q8 are available for the dense classes ",
                    "(Q8 also for the HC mixes)");
    for (const char *c : {"hc_down", "hc_up"})
        STRIX_CHECK(by[c].bits != 6 && by[c].bits != 5, "convert --layout: ", c, "=", by[c].str(),
                    " - no Q5 / Q6 kernel for the HC mixes yet (Q5 / Q6 are available for the dense projections: ",
                    "gdn_in gdn_out attn_qkv attn_o sh_gu sh_down ple_kv lm_head mtp_fc)");
    STRIX_CHECK(by["embed"].bits == 16 || by["embed"].bits == 8,
                "convert --layout: embed=", by["embed"].str(), " unsupported (expected bf16 or q8g<32|64|128>)");
    return by;
}

std::string layout_str(const std::map<std::string, QFmt> &by) {
    std::string r;
    for (const char *c : kLayoutClasses) r += (r.empty() ? "" : ",") + std::string(c) + "=" + by.at(c).str();
    return r;
}

// ---------------------------------------------------------------- planning

class Planner {
public:
    Planner(const Source &src, std::map<std::string, QFmt> layout) : src_(src), layout_(std::move(layout)) {}
    ConvertPlan plan;
    std::set<std::string> used;

    const TensorInfo &need(const std::string &name, Dtype dt) {
        const TensorInfo &ti = src_.info(name);
        STRIX_CHECK(ti.dtype == dt, "convert: '", name, "' is ", dtype_name(ti.dtype), ", expected ", dtype_name(dt));
        STRIX_CHECK(used.insert(name).second, "convert: '", name, "' planned twice");
        return ti;
    }
    void add(ConvertStep s, int64_t unit) {
        s.unit = unit;
        plan.steps.push_back(std::move(s));
    }

    bool shared_split = false;  // the shared expert written separately (its format differs from the experts')

    void set_q(ConvertStep &s, const std::string &cls) {
        const QFmt &f = layout_.at(cls);
        s.out.encoding = f.bits == 8 ? StrixwEncoding::Q8RowMajor
                         : f.bits == 6 ? StrixwEncoding::Q6RowMajor
                         : f.bits == 5 ? StrixwEncoding::Q5RowMajor
                                       : StrixwEncoding::Q4RowMajor;
        s.out.group_size = f.G;
    }
    void quant_single(const std::string &name, const std::string &cls, int64_t unit) {
        const TensorInfo &ti = need(name, Dtype::BF16);
        STRIX_CHECK(ti.shape.size() == 2, "convert: '", name, "' has shape ", shape_str(ti.shape), ", expected [N, K]");
        ConvertStep s;
        s.out.name = name, s.out.shape = ti.shape;
        set_q(s, cls);
        s.op = ConvertOp::Q4Merge, s.sources = {name};
        add(std::move(s), unit);
    }
    // The routed experts alone, stacked [E*n, K] (the shared expert goes separately).
    void experts_only(const std::string &name, const std::string &part, const std::string &cls, int64_t unit) {
        const TensorInfo &ti = need(part, Dtype::BF16);
        STRIX_CHECK(ti.shape.size() == 3, "convert: '", part, "' has shape ", shape_str(ti.shape), ", expected [E, n, K]");
        ConvertStep s;
        s.out.name = name, s.out.shape = {ti.shape[0] * ti.shape[1], ti.shape[2]}, s.out.experts = ti.shape[0];
        s.out.parts.push_back({part, 0, ti.shape[0] * ti.shape[1]});
        set_q(s, cls);
        s.op = ConvertOp::Q4Merge, s.sources = {part};
        add(std::move(s), unit);
    }
    void merge(const MergeGroup &g, bool bf16, int64_t unit, const std::string &cls = "") {
        const MergedLayout m = resolve_merge(g, [&](const std::string &n) -> const TensorInfo * {
            return src_.has(n) ? &src_.info(n) : nullptr;
        });
        ConvertStep s;
        s.out.name = g.name;
        s.out.encoding = StrixwEncoding::BF16;
        if (!bf16) set_q(s, cls);
        s.out.shape = {m.rows, m.K};
        s.out.experts = m.experts;
        for (size_t i = 0; i < g.parts.size(); ++i) {
            need(g.parts[i], Dtype::BF16);
            s.out.parts.push_back({g.parts[i], m.offset[i], m.part_rows[i]});
        }
        s.op = bf16 ? ConvertOp::BF16Rows : ConvertOp::Q4Merge;
        s.sources = g.parts;
        add(std::move(s), unit);
    }
    void copy_bf16(const std::string &name, int64_t unit) {
        const TensorInfo &ti = need(name, Dtype::BF16);
        ConvertStep s;
        s.out.name = name, s.out.encoding = StrixwEncoding::BF16, s.out.shape = ti.shape;
        s.op = ConvertOp::BF16Rows, s.sources = {name};
        add(std::move(s), unit);
    }
    void embed(const std::string &name, int64_t unit) {
        if (layout_.at("embed").bits == 8) {
            quant_single(name, "embed", unit);
        } else {
            copy_bf16(name, unit);
        }
    }
    void f32(const std::string &name, int64_t unit) {
        const TensorInfo &ti = need(name, Dtype::BF16);
        ConvertStep s;
        s.out.name = name, s.out.encoding = StrixwEncoding::F32, s.out.shape = ti.shape;
        s.op = ConvertOp::F32Widen, s.sources = {name};
        add(std::move(s), unit);
    }
    void i64(const std::string &name, int64_t unit) {
        const TensorInfo &ti = need(name, Dtype::I64);
        ConvertStep s;
        s.out.name = name, s.out.encoding = StrixwEncoding::I64, s.out.shape = ti.shape;
        s.op = ConvertOp::I64Copy, s.sources = {name};
        add(std::move(s), unit);
    }
    // One hyper-connection mixer at prefix p ("...attn_hyper_connection."): mix_down = hc_norm-folded
    // [W_down; W_inj] (no W_inj for the final mixers), mix_up = W_up chunk-major, hc_norm as F32.
    void hc(const std::string &p, bool inject, int64_t unit) {
        const std::string down = p + "input_mix_weight_down.weight", up = p + "input_mix_weight_up.weight",
                          inj = p + "block_inject_weight.weight", norm = p + "hc_norm.weight";
        const TensorInfo &td = need(down, Dtype::BF16), &tu = need(up, Dtype::BF16);
        const TensorInfo &tn = src_.info(norm);
        const int64_t r = td.shape.at(0), n4 = td.shape.at(1);
        STRIX_CHECK(td.shape.size() == 2 && tn.shape == std::vector<int64_t>{n4} &&
                        tu.shape == (std::vector<int64_t>{n4, r}),
                    "convert: hyper-connection '", p, "': down ", shape_str(td.shape), ", up ", shape_str(tu.shape),
                    ", hc_norm ", shape_str(tn.shape), " don't fit together");
        ConvertStep s;
        s.out.name = p + "mix_down";
        set_q(s, "hc_down");
        s.out.parts.push_back({down, 0, r});
        s.sources = {down, "", norm};
        int64_t rows = r;
        if (inject) {
            const TensorInfo &ti = need(inj, Dtype::BF16);
            STRIX_CHECK(ti.shape.size() == 2 && ti.shape[1] == n4, "convert: '", inj, "' has shape ",
                        shape_str(ti.shape), ", expected [H, ", n4, "]");
            s.out.parts.push_back({inj, r, ti.shape[0]});
            s.sources[1] = inj;
            rows += ti.shape[0];
        }
        s.out.shape = {rows, n4};
        s.op = ConvertOp::Q4FoldHC;
        add(std::move(s), unit);
        ConvertStep u;
        u.out.name = p + "mix_up", u.out.shape = tu.shape;
        u.out.encoding = layout_.at("hc_up").bits == 8 ? StrixwEncoding::Q8ChunkMajor : StrixwEncoding::Q4ChunkMajor;
        u.out.group_size = layout_.at("hc_up").G;
        u.op = ConvertOp::Q4ChunkMajor, u.sources = {up};
        add(std::move(u), unit);
        f32(norm, unit);
    }
    void layer(const std::string &P, int64_t unit) {
        const bool full = src_.has(P + "self_attn.q_proj.weight");
        STRIX_CHECK(full || src_.has(P + "linear_attn.in_proj_qkv.weight"), "convert: layer '", P,
                    "' has neither self_attn.q_proj nor linear_attn.in_proj_qkv");
        const std::string M = P + "mlp.";
        for (const MergeGroup &g :
             merge_plan(Arch::Qwen4Exp, P, full ? LayerKind::FullAttention : LayerKind::LinearAttention)) {
            if (g.name == M + "router") {
                merge(g, true, unit);  // BF16 (router logits are kept FP32)
            } else if (g.name == M + "experts_gate_up" || g.name == M + "experts_down") {
                // Shared expert as #512 of the stack only when it has the experts' format (one merged weight).
                const bool gu = g.name == M + "experts_gate_up";
                const std::string ecls = gu ? "exp_gu" : "exp_down", scls = gu ? "sh_gu" : "sh_down";
                if (layout_.at(ecls) == layout_.at(scls)) {
                    merge(g, false, unit, ecls);
                } else {
                    shared_split = true;
                    experts_only(g.name, g.parts[0], ecls, unit);
                    if (gu)
                        merge({M + "shared_expert.gate_up", {g.parts[1], g.parts[2]}}, false, unit, scls);
                    else
                        quant_single(g.parts[1], scls, unit);
                }
            } else {
                merge(g, false, unit, full ? "attn_qkv" : "gdn_in");
            }
        }
        if (full) {
            quant_single(P + "self_attn.o_proj.weight", "attn_o", unit);
            for (const char *n : {"self_attn.q_norm.weight", "self_attn.k_norm.weight",
                                  "self_attn.indexer.q_layernorm.weight", "self_attn.indexer.k_layernorm.weight"})
                f32(P + n, unit);
        } else {
            quant_single(P + "linear_attn.out_proj.weight", "gdn_out", unit);
            for (const char *n : {"linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.conv1d.weight",
                                  "linear_attn.norm.weight"})
                f32(P + n, unit);
        }
        hc(P + "attn_hyper_connection.", true, unit);
        hc(P + "mlp_hyper_connection.", true, unit);
        if (src_.has(P + "ple.key_proj.weight")) {
            merge({P + "ple.kv_proj", {P + "ple.key_proj.weight", P + "ple.value_proj.weight"}}, false, unit, "ple_kv");
            for (const char *n : {"ple.conv1d.weight", "ple.norm_conv.weight", "ple.norm_key.weight",
                                  "ple.norm_query.weight"})
                f32(P + n, unit);
            for (const char *n : {"ple.ple_embedding.layer_multipliers", "ple.ple_embedding.ngram_heads_offsets",
                                  "ple.ple_embedding.ngram_heads_vocab_sizes"})
                i64(P + n, unit);
            const std::string shard = P + "ple.ple_embedding.ngram_embedding.shard_";
            for (const std::string &n : src_.names())
                if (n.rfind(shard, 0) == 0)
                    plan.skipped.emplace_back(n, "n-gram table: its own file (convert_ngram_table)");
        }
    }

private:
    const Source &src_;
    std::map<std::string, QFmt> layout_;
};

int64_t count_layers(const Source &src) {
    const std::string pre = "model.language_model.layers.";
    int64_t n = 0;
    for (const std::string &name : src.names())
        if (name.rfind(pre, 0) == 0) n = std::max<int64_t>(n, std::stoll(name.substr(pre.size())) + 1);
    STRIX_CHECK(n >= 1, "convert: no decoder layers ('", pre, "N.') in the checkpoint");
    for (int64_t i = 0; i < n; ++i)
        STRIX_CHECK(src.has(pre + std::to_string(i) + ".attn_hyper_connection.hc_norm.weight"), "convert: layer ", i,
                    " of ", n, " is missing (no attn_hyper_connection.hc_norm)");
    return n;
}

ConvertPlan make_plan(const Source &src, const ConvertOptions &opt) {
    Planner p(src, parse_layout(opt.layout));
    p.plan.layers_total = count_layers(src);
    std::vector<int64_t> layers = opt.layers;
    if (layers.empty())
        for (int64_t i = 0; i < p.plan.layers_total; ++i) layers.push_back(i);
    for (int64_t l : layers)
        STRIX_CHECK(l >= 0 && l < p.plan.layers_total, "convert: layer ", l, " requested, the checkpoint has ",
                    p.plan.layers_total);
    const std::string lm = "model.language_model.";
    if (opt.globals) {
        p.embed(lm + "embed_tokens.weight", -1);
        p.hc(lm + "hyper_connection_mixer.", false, -1);
        p.quant_single("lm_head.weight", "lm_head", -1);
    }
    for (int64_t l : layers) p.layer(lm + "layers." + std::to_string(l) + ".", l);
    if (opt.mtp) {
        for (const char *n : {"mtp.fc_embedding.weight", "mtp.fc_hidden.weight"}) p.quant_single(n, "mtp_fc", -2);
        for (const char *n : {"mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight"}) p.f32(n, -2);
        p.hc("mtp.hyper_connection_mixer.", false, -2);
        p.layer("mtp.layers.0.", -2);
    }
    for (const std::string &n : src.names())
        if (n.rfind("model.visual.", 0) == 0) p.plan.skipped.emplace_back(n, "vision tower: not loaded (text only)");
    // A full conversion accounts for every source tensor; anything unplanned is a spec gap, not a skip.
    const bool full = opt.layers.empty() && opt.globals && opt.mtp;
    if (full) {
        std::set<std::string> skipped;
        for (const auto &s : p.plan.skipped) skipped.insert(s.first);
        std::vector<std::string> unknown;
        for (const std::string &n : src.names())
            if (!p.used.count(n) && !skipped.count(n)) unknown.push_back(n);
        std::string list;
        for (size_t i = 0; i < unknown.size() && i < 10; ++i) list += "\n  " + unknown[i];
        STRIX_CHECK(unknown.empty(), "convert: ", unknown.size(), " checkpoint tensors aren't in the conversion plan",
                    " (plan_conversion lists what goes in):", list);
    }
    p.plan.shared_split = p.shared_split;
    return p.plan;
}

// ---------------------------------------------------------------- execution

// Quantizes rows [0, n) of src (BF16 bits or FP32, K per row) into rows [row0, row0 + n) of out, which is
// already sized for its N rows. Split over threads; each thread converts sub-blocks to FP32 and quantizes
// them with formats/q4's quantize_q4, so the bytes are exactly the reference quantizer's.
inline Q4Weight quantize_w(const float *f, int64_t m, int64_t K, int64_t G, const Q4Weight *) {
    return quantize_q4(f, m, K, G);
}
inline Q8Weight quantize_w(const float *f, int64_t m, int64_t K, int64_t G, const Q8Weight *) {
    return quantize_q8(f, m, K, G);
}
inline Q6Weight quantize_w(const float *f, int64_t m, int64_t K, int64_t G, const Q6Weight *) {
    return quantize_q6(f, m, K, G);
}
inline Q5Weight quantize_w(const float *f, int64_t m, int64_t K, int64_t G, const Q5Weight *) {
    return quantize_q5(f, m, K, G);
}
inline std::vector<float> dequantize_w(const Q4Weight &q) { return dequantize_q4(q); }
inline std::vector<float> dequantize_w(const Q5Weight &q) { return dequantize_q5(q); }
inline std::vector<float> dequantize_w(const Q6Weight &q) { return dequantize_q6(q); }
inline std::vector<float> dequantize_w(const Q8Weight &q) { return dequantize_q8(q); }
inline void check_w(const Q4Weight &q, const char *what) { check_q4(q, what); }
inline void check_w(const Q8Weight &q, const char *what) { check_q8(q, what); }
inline void check_w(const Q6Weight &q, const char *what) { check_q6(q, what); }
inline void check_w(const Q5Weight &q, const char *what) { check_q5(q, what); }
template <typename W> constexpr int64_t code_bytes(int64_t K) {
    return std::is_same_v<W, Q8Weight>   ? K
           : std::is_same_v<W, Q6Weight> ? q6_row_bytes(K)
           : std::is_same_v<W, Q5Weight> ? q5_row_bytes(K)
                                         : K / 2;
}

template <typename W, typename In>
void quantize_rows(const In *src, int64_t n, W &out, int64_t row0, int threads) {
    const int64_t K = out.K, G = out.G, gpr = K / G;
    STRIX_CHECK(row0 >= 0 && n >= 0 && row0 + n <= out.N, "quantize_rows: rows [", row0, ", ", row0 + n,
                ") outside ", out.N);
    const int nt = (int)std::max<int64_t>(1, std::min<int64_t>(threads, n));
    std::vector<std::thread> pool;
    std::vector<std::exception_ptr> errs((size_t)nt);
    for (int t = 0; t < nt; ++t)
        pool.emplace_back([&, t] {
            try {
                const int64_t a = n * t / nt, b = n * (t + 1) / nt;
                const int64_t sub = std::max<int64_t>(1, (8ll << 20) / (K * 4));  // ~8 MB of FP32 per call
                std::vector<float> f;
                for (int64_t r = a; r < b; r += sub) {
                    const int64_t m = std::min(sub, b - r);
                    f.resize((size_t)(m * K));
                    for (size_t i = 0; i < f.size(); ++i) {
                        if constexpr (std::is_same_v<In, uint16_t>) f[i] = bf16_to_f32(src[(size_t)(r * K) + i]);
                        else f[i] = src[(size_t)(r * K) + i];
                    }
                    const W q = quantize_w(f.data(), m, K, G, (const W *)nullptr);
                    const int64_t at = row0 + r;
                    std::memcpy(out.q.data() + at * code_bytes<W>(K), q.q.data(), q.q.size());
                    std::memcpy(out.scale.data() + at * gpr, q.scale.data(), q.scale.size() * 2);
                    std::memcpy(out.minv.data() + at * gpr, q.minv.data(), q.minv.size() * 2);
                }
            } catch (...) {
                errs[(size_t)t] = std::current_exception();
            }
        });
    for (std::thread &th : pool) th.join();
    for (auto &e : errs)
        if (e) std::rethrow_exception(e);
}

template <typename W> W empty_w(int64_t N, int64_t K, int64_t G) {
    W q;
    q.N = N, q.K = K, q.G = G;
    q.q.resize((size_t)(N * code_bytes<W>(K)));
    q.scale.resize((size_t)(N * (K / G)));
    q.minv.resize((size_t)(N * (K / G)));
    return q;
}

// Relative RMS error sqrt(sum (dq - w)^2 / sum w^2) over kErrorSampleRows rows spread over the tensor;
// row_src(r, dst) fills the source values of output row r.
template <typename W, typename RowSrc> double sampled_error(const W &q, RowSrc row_src) {
    const int64_t K = q.K, gpr = K / q.G, rows = std::min<int64_t>(kErrorSampleRows, q.N);
    double se = 0, sw = 0;
    std::vector<float> w((size_t)K);
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t r = rows == 1 ? 0 : i * (q.N - 1) / (rows - 1);
        W one;
        one.N = 1, one.K = K, one.G = q.G;
        const int64_t cb = code_bytes<W>(K);
        one.q.assign(q.q.begin() + r * cb, q.q.begin() + (r + 1) * cb);
        one.scale.assign(q.scale.begin() + r * gpr, q.scale.begin() + (r + 1) * gpr);
        one.minv.assign(q.minv.begin() + r * gpr, q.minv.begin() + (r + 1) * gpr);
        const std::vector<float> dq = dequantize_w(one);
        row_src(r, w.data());
        for (int64_t k = 0; k < K; ++k) {
            const double e = (double)dq[(size_t)k] - w[(size_t)k];
            se += e * e, sw += (double)w[(size_t)k] * w[(size_t)k];
        }
    }
    return sw > 0 ? std::sqrt(se / sw) : 0.0;
}

// The log's op column: the op, with q8 / q6 shown for Q8 / Q6 steps ("q4" / "q8" / "q6" / "q4-fold" / ...).
std::string step_tag(const ConvertStep &s) {
    std::string t = convert_op_name(s.op);
    if (s.out.encoding == StrixwEncoding::Q8RowMajor || s.out.encoding == StrixwEncoding::Q8ChunkMajor)
        t = "q8" + t.substr(2);
    if (s.out.encoding == StrixwEncoding::Q6RowMajor) t = "q6" + t.substr(2);
    if (s.out.encoding == StrixwEncoding::Q5RowMajor) t = "q5" + t.substr(2);
    return t;
}

std::string short_name(const std::string &n) {
    for (const char *pre : {"model.language_model.layers.", "model.language_model.", "mtp.layers."}) {
        const std::string p = pre;
        if (n.rfind(p, 0) == 0) {
            std::string rest = n.substr(p.size());
            if (p.back() == '.' && std::isdigit((unsigned char)rest[0])) rest = rest.substr(rest.find('.') + 1);
            return rest;
        }
    }
    return n;
}

const char *explain(ConvertOp op, bool merged, int64_t experts, int bits) {
    switch (op) {
    case ConvertOp::Q4Merge:
        if (experts == 513)
            return "experts: the 512 routed experts and the shared expert (as #512) are stacked into one "
                   "[513*n, K] matrix; the expert kernels pick an expert's rows by its id.";
        if (experts > 0)
            return "experts: the 512 routed experts stacked into one [512*n, K] matrix; the shared expert has a "
                   "different format in this layout, so it's written separately (its own gate|up and down).";
        if (merged)
            return "merge: projections that read the same input are stacked row-wise into one matrix, so the "
                   "GPU runs one launch instead of several; each part's rows quantize exactly as they would alone.";
        if (bits == 5)
            return "q5: 5-bit codes (per row a plane of low nibbles, then one of the top bit) with the same BF16 "
                   "scale + min per group - 5.25 bits per weight at G128, 2x finer steps than q4.";
        if (bits == 6)
            return "q6: 6-bit codes (per row a plane of low nibbles, then one of the top 2 bits) with the same BF16 "
                   "scale + min per group - 6.25 bits per weight at G128, 4x finer steps than q4.";
        if (bits == 8)
            return "q8: 8-bit codes, one per byte, with the same BF16 scale + min per group - 8.5 bits per weight "
                   "at G64, ~16x finer steps than q4; for the classes the quantization study found sensitive.";
        return "q4: 4-bit codes (two per byte) plus a BF16 scale and min per 64 weights along K = 4.5 bits per "
               "weight; per group s = (max - min) / 15, code = round((w - min) / s), w' = code * s + min.";
    case ConvertOp::Q4FoldHC:
        return "hc fold: the hc_norm weight (1 + w) multiplies every column of W_down (and W_inject) before "
               "quantizing, so the mix kernel never has to write the normalized streams to memory.";
    case ConvertOp::Q4ChunkMajor:
        return "chunk-major: the same codes (Q4 or Q8) reordered so each 32-code chunk of consecutive rows sits side by "
               "side - hc_mix_up gives every lane one row, and this makes their loads one contiguous run.";
    case ConvertOp::BF16Rows:
        return "bf16: kept at 16 bits - the router (int4 moved the chosen experts on 46% of token-layers) and "
               "the embedding (a lookup: one row per token, so its size costs memory, not speed).";
    case ConvertOp::F32Widen:
        return "f32: small vectors (norm weights, conv taps, decay parameters) widened exactly from BF16 - the "
               "kernels read FP32.";
    case ConvertOp::I64Copy: return "i64: the PLE n-gram hashing constants, copied as they are.";
    }
    return "";
}

void set_no_compression(const std::string &dir, Log &log) {
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    int flags = 0;
    if (fd >= 0 && ::ioctl(fd, FS_IOC_GETFLAGS, &flags) == 0) {
        flags |= FS_NOCOMP_FL;
        if (::ioctl(fd, FS_IOC_SETFLAGS, &flags) == 0) {
            log("output dir: btrfs compression off (chattr +m) - 4-bit data doesn't compress, and reading it back "
                "shouldn't pay for zstd");
            ::close(fd);
            return;
        }
    }
    log("output dir: couldn't turn compression off (%s) - fine on a filesystem without compression",
        std::strerror(errno));
    if (fd >= 0) ::close(fd);
}

std::string utc_now() {
    char ts[32];
    std::time_t t = std::time(nullptr);
    std::strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return ts;
}

std::string unit_name(int64_t unit, int64_t layers_total, const ConvertPlan &plan) {
    if (unit == -1) return "globals (embedding, final mixer, LM head)";
    if (unit == -2) return "MTP head";
    bool full = false;
    for (const ConvertStep &s : plan.steps)
        if (s.unit == unit && s.out.name.find(".self_attn.qkv") != std::string::npos) full = true;
    return "layer " + std::to_string(unit) + "/" + std::to_string(layers_total - 1) +
           (full ? " - full attention (QSA)" : " - linear attention (GDN)");
}

}  // namespace

const char *convert_op_name(ConvertOp op) {
    switch (op) {
    case ConvertOp::Q4Merge: return "q4";
    case ConvertOp::Q4FoldHC: return "q4-fold";
    case ConvertOp::Q4ChunkMajor: return "q4-chunk";
    case ConvertOp::BF16Rows: return "bf16";
    case ConvertOp::F32Widen: return "f32";
    case ConvertOp::I64Copy: return "i64";
    }
    return "?";
}

std::string checkpoint_fingerprint(const std::string &dir) { return Source(dir).fingerprint(); }

ConvertResult convert_qwen4exp(const ConvertOptions &opt) {
    const auto t0 = Clock::now();
    STRIX_CHECK(!opt.src_dir.empty() && (!opt.out_dir.empty() || opt.plan_only), "convert: src_dir and out_dir needed");
    const std::string layout = layout_str(parse_layout(opt.layout));  // validates before any work
    STRIX_CHECK(!opt.git.empty(), "convert: the converter's git hash is required (recorded in the file)");
    const int threads = opt.threads > 0 ? opt.threads : (int)std::max(1u, std::thread::hardware_concurrency());
    Log log(opt.console, t0);
    ConvertResult res;
    const std::string final_path = opt.out_dir + "/weights.strixw", partial = final_path + ".partial";
    if (!opt.plan_only) {
        STRIX_CHECK(!fs::exists(final_path), "convert: '", final_path,
                    "' already exists - remove it first (the converter never overwrites a finished file)");
        const bool fresh = !fs::exists(opt.out_dir);
        fs::create_directories(opt.out_dir);
        log.open_file(opt.out_dir + "/convert.log");
        if (fresh) set_no_compression(opt.out_dir, log);
    }
    log.rule("strix-infer converter \u00b7 Qwen3.8-Flash-Next (qwen4exp) -> strixw v" + std::to_string(kStrixwVersion));
    log("layout: %s", layout.c_str());
    log("converter git %s, %d threads", opt.git.c_str(), threads);

    log("reading the checkpoint's headers from %s ...", opt.src_dir.c_str());
    const Source src(opt.src_dir);
    log("  %zu shards, %s of tensor data, fingerprint %s", src.shards(), fmt_bytes((double)src.bytes()).c_str(),
        src.fingerprint().c_str());

    const ConvertPlan plan = make_plan(src, opt);
    uint64_t planned_src = 0, planned_out = 0, skipped_bytes = 0;
    std::map<std::string, std::pair<int, uint64_t>> by_op;
    std::vector<StrixwTensor> outs;
    for (const ConvertStep &s : plan.steps) {
        StrixwTensor t = s.out;
        t.components = strixw_components(t);
        for (const std::string &n : s.sources)
            if (!n.empty()) planned_src += src.info(n).byte_length;
        planned_out += t.bytes();
        auto &e = by_op[convert_op_name(s.op)];
        e.first += 1, e.second += t.bytes();
        outs.push_back(std::move(t));
    }
    std::map<std::string, std::pair<int, uint64_t>> skip_why;
    for (const auto &[n, why] : plan.skipped) {
        auto &e = skip_why[why];
        e.first += 1, e.second += src.info(n).byte_length;
        skipped_bytes += src.info(n).byte_length;
    }
    log.blank();
    log.rule("plan");
    log("%zu output tensors from %s of source -> %s (%.1f%% of the source size)", plan.steps.size(),
        fmt_bytes((double)planned_src).c_str(), fmt_bytes((double)planned_out).c_str(),
        100.0 * (double)planned_out / (double)std::max<uint64_t>(1, planned_src));
    for (const auto &[op, e] : by_op)
        log("  %-9s %4d tensors  %12s", op.c_str(), e.first, fmt_bytes((double)e.second).c_str());
    if (skipped_bytes) log("  skipped in total: %s (not in this file)", fmt_bytes((double)skipped_bytes).c_str());
    for (const auto &[why, e] : skip_why)
        log("  skipped %4d source tensors (%s): %s", e.first, fmt_bytes((double)e.second).c_str(), why.c_str());
    if (!opt.layers.empty() || !opt.globals || !opt.mtp)
        log("  PARTIAL conversion (%s%s%s) - the file is marked complete=no and the full-coverage check is off",
            opt.layers.empty() ? "all layers" : "some layers", opt.globals ? "" : ", no globals",
            opt.mtp ? "" : ", no MTP");
    res.source_bytes = planned_src, res.output_bytes = planned_out, res.tensors = plan.steps.size();
    if (opt.plan_only) {
        for (const ConvertStep &s : plan.steps)
            log("  %-9s %-62s %s", convert_op_name(s.op), s.out.name.c_str(), shape_str(s.out.shape).c_str());
        log("plan only: nothing written");
        res.seconds = log.elapsed();
        return res;
    }

    std::string layer_list = "all";
    if (!opt.layers.empty()) {
        layer_list.clear();
        for (int64_t l : opt.layers) layer_list += (layer_list.empty() ? "" : ",") + std::to_string(l);
    }
    const bool complete = opt.layers.empty() && opt.globals && opt.mtp;
    std::map<std::string, std::string> meta = {
        {"arch", "qwen4exp"},
        {"layout", layout},
        {"shared_expert", plan.shared_split ? "separate" : "stacked"},
        {"source_dir", opt.src_dir},
        {"source_fingerprint", src.fingerprint()},
        {"converter_git", opt.git},
        {"created", utc_now()},
        {"layers", layer_list},
        {"globals", opt.globals ? "yes" : "no"},
        {"mtp", opt.mtp ? "yes" : "no"},
        {"complete", complete ? "yes" : "no"},
        {"format_note", "embedding BF16 (not layout A's int8); n-gram table not included (OPEN)"}};
    StrixwWriter writer(partial, meta, outs);
    log.blank();
    log.rule("convert");
    log("writing %s (%s, %zu tensors; header + index %s)", partial.c_str(),
        fmt_bytes((double)writer.file_bytes()).c_str(), outs.size(),
        fmt_bytes((double)writer.tensors().front().components.front().offset).c_str());

    std::set<std::string> explained_kinds;
    uint64_t done_src = 0, done_out = 0;
    size_t unit_first = 0;
    int units_done = 0, units_total = 0;
    {
        std::set<int64_t> u;
        for (const ConvertStep &s : plan.steps) u.insert(s.unit);
        units_total = (int)u.size();
    }
    auto ta = Clock::now();
    for (size_t si = 0; si < plan.steps.size(); ++si) {
        const ConvertStep &s = plan.steps[si];
        if (si == unit_first) {
            uint64_t unit_src = 0;
            for (size_t j = si; j < plan.steps.size() && plan.steps[j].unit == s.unit; ++j)
                for (const std::string &n : plan.steps[j].sources)
                    if (!n.empty()) unit_src += src.info(n).byte_length;
            log.blank();
            log.rule(unit_name(s.unit, plan.layers_total, plan) + " \u00b7 " + fmt_bytes((double)unit_src) + " of source",
                     false);
            ta = Clock::now();
        }
        const bool merged = s.out.parts.size() > 1;
        // Each explanation once (keyed on its text), the first time a step needs it.
        const bool q8 = s.out.encoding == StrixwEncoding::Q8RowMajor || s.out.encoding == StrixwEncoding::Q8ChunkMajor;
        const int bits = q8                                              ? 8
                         : s.out.encoding == StrixwEncoding::Q6RowMajor ? 6
                         : s.out.encoding == StrixwEncoding::Q5RowMajor ? 5
                                                                        : 4;
        const std::string note = explain(s.op, merged, s.out.experts, bits);
        if (explained_kinds.insert(note).second) log.note(note);
        if (bits != 4 && merged && explained_kinds.insert(explain(s.op, false, 0, bits)).second)
            log.note(explain(s.op, false, 0, bits));
        const auto ts = Clock::now();
        uint64_t step_src = 0;
        for (const std::string &n : s.sources)
            if (!n.empty()) step_src += src.info(n).byte_length;
        double err = -1;
        const StrixwTensor &planned = writer.tensors()[si];

        if (s.op == ConvertOp::Q4Merge || s.op == ConvertOp::Q4ChunkMajor || s.op == ConvertOp::Q4FoldHC) {
          // Generic over the weight type: Q4 or Q8, every op (merged rows, the HC fold, chunk-major).
          auto quantize_step = [&](auto proto) {
            using W = decltype(proto);
            const int64_t N = s.out.shape[0], K = s.out.shape[1], G = s.out.group_size;
            W q = empty_w<W>(N, K, G);
            if (s.op == ConvertOp::Q4FoldHC) {
                // Fold in FP32: w' = w * (1 + hc_norm[k]), exactly as test_hc_kernel builds the reference.
                const std::vector<float> norm = src.read_f32(s.sources[2]);
                std::vector<float> di((size_t)(N * K));
                int64_t row = 0;
                for (const StrixwPart &p : s.out.parts) {
                    const std::vector<float> w = src.read_f32(p.source);
                    for (int64_t i = 0; i < p.rows; ++i, ++row)
                        for (int64_t k = 0; k < K; ++k)
                            di[(size_t)(row * K + k)] = w[(size_t)(i * K + k)] * (1.0f + norm[(size_t)k]);
                }
                quantize_rows(di.data(), N, q, 0, threads);
                err = sampled_error(q, [&](int64_t r, float *dst) { std::memcpy(dst, &di[(size_t)(r * K)], K * 4); });
            } else {
                // Stream each part in ~256 MB chunks: the next chunk is read while this one quantizes.
                std::vector<StrixwPart> parts = s.out.parts;
                if (parts.empty()) parts.push_back({s.sources[0], 0, N});
                const int64_t chunk_rows = std::max<int64_t>(1, (int64_t)(kChunkBytes / (size_t)(K * 2)));
                std::vector<uint16_t> buf[2];
                for (const StrixwPart &p : parts) {
                    auto read = [&, name = p.source](int64_t r0, int64_t n, std::vector<uint16_t> &b) {
                        b.resize((size_t)(n * K));
                        src.read(name, (uint64_t)(r0 * K * 2), (size_t)(n * K * 2), b.data());
                    };
                    int cur = 0;
                    std::future<void> next = std::async(std::launch::async, read, 0,
                                                        std::min(chunk_rows, p.rows), std::ref(buf[0]));
                    for (int64_t r0 = 0; r0 < p.rows; r0 += chunk_rows) {
                        next.get();
                        const int64_t n = std::min(chunk_rows, p.rows - r0), r1 = r0 + n;
                        if (r1 < p.rows)
                            next = std::async(std::launch::async, read, r1, std::min(chunk_rows, p.rows - r1),
                                              std::ref(buf[1 - cur]));
                        quantize_rows(buf[cur].data(), n, q, p.row_offset + r0, threads);
                        cur = 1 - cur;
                    }
                }
                err = sampled_error(q, [&](int64_t r, float *dst) {
                    for (const StrixwPart &p : parts)
                        if (r >= p.row_offset && r < p.row_offset + p.rows) {
                            std::vector<uint16_t> b((size_t)K);
                            src.read(p.source, (uint64_t)((r - p.row_offset) * K * 2), (size_t)(K * 2), b.data());
                            for (int64_t k = 0; k < K; ++k) dst[k] = bf16_to_f32(b[(size_t)k]);
                            return;
                        }
                });
            }
            check_w(q, s.out.name.c_str());
            bool chunk_major_done = false;
            if constexpr (!std::is_same_v<W, Q6Weight> && !std::is_same_v<W, Q5Weight>) {  // Q5 / Q6: no chunk-major form
                if (s.op == ConvertOp::Q4ChunkMajor) {
                    const auto cm = to_chunk_major(q, s.out.name.c_str());  // Q4ChunkMajor or Q8ChunkMajor
                    writer.write(si, StrixwRole::Q, cm.q.data(), cm.q.size());
                    writer.write(si, StrixwRole::Scale, cm.scale.data(), cm.scale.size() * 2);
                    writer.write(si, StrixwRole::Min, cm.minv.data(), cm.minv.size() * 2);
                    chunk_major_done = true;
                }
            }
            if (!chunk_major_done) {
                writer.write(si, StrixwRole::Q, q.q.data(), q.q.size());
                writer.write(si, StrixwRole::Scale, q.scale.data(), q.scale.size() * 2);
                writer.write(si, StrixwRole::Min, q.minv.data(), q.minv.size() * 2);
            }
          };
          if (planned.encoding == StrixwEncoding::Q8RowMajor || planned.encoding == StrixwEncoding::Q8ChunkMajor) {
              STRIX_CHECK((planned.encoding == StrixwEncoding::Q8ChunkMajor) == (s.op == ConvertOp::Q4ChunkMajor),
                          "convert: '", s.out.name, "': chunk-major encoding ", strixw_encoding_name(planned.encoding),
                          " for op ", convert_op_name(s.op));
              quantize_step(Q8Weight{});
          } else if (planned.encoding == StrixwEncoding::Q6RowMajor) {
              STRIX_CHECK(s.op != ConvertOp::Q4ChunkMajor, "convert: '", s.out.name, "': Q6 has no chunk-major form (op ",
                          convert_op_name(s.op), ")");
              quantize_step(Q6Weight{});
          } else if (planned.encoding == StrixwEncoding::Q5RowMajor) {
              STRIX_CHECK(s.op != ConvertOp::Q4ChunkMajor, "convert: '", s.out.name, "': Q5 has no chunk-major form (op ",
                          convert_op_name(s.op), ")");
              quantize_step(Q5Weight{});
          } else {
              quantize_step(Q4Weight{});
          }
        } else if (s.op == ConvertOp::BF16Rows) {
            std::vector<uint8_t> b(planned.bytes());
            size_t at = 0;
            for (const std::string &n : s.sources) {
                const TensorInfo &ti = src.info(n);
                src.read(n, 0, ti.byte_length, b.data() + at);
                at += ti.byte_length;
            }
            STRIX_CHECK(at == b.size(), "convert: '", s.out.name, "' got ", at, " bytes, planned ", b.size());
            writer.write(si, StrixwRole::Data, b.data(), b.size());
        } else if (s.op == ConvertOp::F32Widen) {
            const std::vector<float> f = src.read_f32(s.sources[0]);
            writer.write(si, StrixwRole::Data, f.data(), f.size() * 4);
        } else {
            const std::vector<int64_t> v = src.read_all<int64_t>(s.sources[0]);
            writer.write(si, StrixwRole::Data, v.data(), v.size() * 8);
        }

        const double dt = std::chrono::duration<double>(Clock::now() - ts).count();
        done_src += step_src, done_out += planned.bytes();
        std::string from;
        if (merged || s.op == ConvertOp::Q4FoldHC) {
            for (const StrixwPart &p : s.out.parts)
                from += (from.empty() ? "" : " + ") + short_name(p.source) + shape_str(src.info(p.source).shape);
            if (s.op == ConvertOp::Q4FoldHC) from += " * (1 + hc_norm)";
            log("  %-8s %s <- %s", step_tag(s).c_str(), short_name(s.out.name).c_str(), from.c_str());
        }
        char q4info[96] = "";
        if (planned.encoding == StrixwEncoding::Q4RowMajor || planned.encoding == StrixwEncoding::Q4ChunkMajor ||
            planned.encoding == StrixwEncoding::Q8RowMajor || planned.encoding == StrixwEncoding::Q8ChunkMajor ||
            planned.encoding == StrixwEncoding::Q6RowMajor || planned.encoding == StrixwEncoding::Q5RowMajor)
            std::snprintf(q4info, sizeof q4info, "  %.2f bpw  err %.2f%%", 8.0 * (double)planned.bytes() / (double)planned.numel(),
                          100 * err);
        log("  %-8s %-44s %-16s %10s%s  %6.2fs  %s/s", step_tag(s).c_str(),
            (merged || s.op == ConvertOp::Q4FoldHC) ? "" : short_name(s.out.name).c_str(),
            shape_str(s.out.shape).c_str(), fmt_bytes((double)planned.bytes()).c_str(), q4info, dt,
            fmt_bytes(dt > 0 ? (double)step_src / dt : 0).c_str());

        const bool unit_end = si + 1 == plan.steps.size() || plan.steps[si + 1].unit != s.unit;
        if (unit_end) {
            ++units_done;
            unit_first = si + 1;
            const double el = log.elapsed(), rate = (double)done_src / std::max(1e-9, el);
            const double eta = (double)(planned_src - done_src) / std::max(1.0, rate);
            log("  \u2713 done in %s \u2502 %d/%d parts \u2502 read %s of %s (%s/s avg) \u2502 written %s of %s \u2502 ETA %s",
                fmt_dur(std::chrono::duration<double>(Clock::now() - ta).count()).c_str(), units_done, units_total,
                fmt_bytes((double)done_src).c_str(), fmt_bytes((double)planned_src).c_str(),
                fmt_bytes(rate).c_str(), fmt_bytes((double)done_out).c_str(), fmt_bytes((double)planned_out).c_str(),
                fmt_dur(eta).c_str());
        }
    }

    log.blank();
    log.rule("finish");
    log("all tensors written; writing the final index (component hashes) and syncing to disk ...");
    writer.finish();
    log("verifying: re-opening the file through the reader and re-hashing every component (%d threads) ...",
        threads);
    const auto tv = Clock::now();
    {
        const StrixwFile check(partial);
        STRIX_CHECK(check.tensors().size() == plan.steps.size(), "convert: the written file has ",
                    check.tensors().size(), " tensors, planned ", plan.steps.size());
        check.verify_hashes(threads);
    }
    const double vt = std::chrono::duration<double>(Clock::now() - tv).count();
    if (const int fd = ::open(partial.c_str(), O_RDONLY | O_CLOEXEC); fd >= 0) {  // don't leave 67 GiB cached
        (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        ::close(fd);
    }
    log("  verified %s in %s (%s/s)", fmt_bytes((double)planned_out).c_str(), fmt_dur(vt).c_str(),
        fmt_bytes((double)planned_out / std::max(1e-9, vt)).c_str());
    fs::rename(partial, final_path);
    res.weights_path = final_path;
    res.seconds = log.elapsed();
    log("\u2714 done: %s -> %s, %zu tensors, %s in %s", fmt_bytes((double)planned_src).c_str(), final_path.c_str(),
        plan.steps.size(), fmt_bytes((double)planned_out).c_str(), fmt_dur(res.seconds).c_str());
    return res;
}

}  // namespace strix
