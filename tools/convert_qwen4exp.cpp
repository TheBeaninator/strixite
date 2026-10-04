// Offline converter: Qwen3.8-Flash-Next HF safetensors -> <out>/weights.strixw (formats/strixw.hpp).
// Verbose by default: every step is logged to stderr and to <out>/convert.log.
//
// Usage: convert_qwen4exp --src DIR --out DIR --git HASH [--threads N] [--layout default=q4g64,...]
//                         [--layers 0,3,7] [--no-globals] [--no-mtp] [--plan]
//   --plan        read the headers, log the full plan (every output tensor), write nothing
//   --layers ...  convert only these decoder layers (a partial file, marked complete=no)

#include "common/args.hpp"
#include "formats/convert_qwen4exp.hpp"

#include <cstdio>
#include <exception>
#include <sstream>
#include <string>

using namespace strix;

namespace {

int run(int argc, char **argv) {
    ConvertOptions o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](const char *flag) {
            STRIX_CHECK(i + 1 < argc, flag, " needs a value");
            return std::string(argv[++i]);
        };
        if (a == "--src") o.src_dir = value("--src");
        else if (a == "--out") o.out_dir = value("--out");
        else if (a == "--git") o.git = value("--git");
        else if (a == "--threads") o.threads = (int)parse_int(value("--threads"), "--threads", 1, 1024);
        else if (a == "--layout") o.layout = value("--layout");
        else if (a == "--layers") {
            std::stringstream ss(value("--layers"));
            for (std::string t; std::getline(ss, t, ',');) o.layers.push_back(parse_int(t, "--layers entry", 0, 1 << 20));
        } else if (a == "--no-globals") o.globals = false;
        else if (a == "--no-mtp") o.mtp = false;
        else if (a == "--plan") o.plan_only = true;
        else {
            std::fprintf(stderr,
                         "usage: %s --src DIR --out DIR --git HASH [--threads N] [--layout default=q4g64,...] "
                         "[--layers 0,3] [--no-globals] [--no-mtp] [--plan]\n",
                         argv[0]);
            return 2;
        }
    }
    STRIX_CHECK(!o.src_dir.empty() && !o.git.empty() && (o.plan_only || !o.out_dir.empty()),
                "--src, --git and (unless --plan) --out are required");
    convert_qwen4exp(o);
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "convert_qwen4exp: %s\n", e.what());
        return 1;
    }
}
