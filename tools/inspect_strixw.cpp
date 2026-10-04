// Prints a strixw file's metadata and tensor index (formats/strixw.hpp); --verify re-hashes every
// component. Usage: inspect_strixw FILE [--verify] [--threads N]

#include "common/args.hpp"
#include "formats/strixw.hpp"

#include <cstdio>
#include <exception>
#include <map>
#include <string>
#include <thread>

using namespace strix;

namespace {

int run(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s FILE [--verify] [--threads N]\n", argv[0]);
        return 2;
    }
    bool verify = false;
    int threads = (int)std::max(1u, std::thread::hardware_concurrency());
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--verify") verify = true;
        else if (a == "--threads" && i + 1 < argc) threads = (int)parse_int(argv[++i], "--threads", 1, 1024);
        else STRIX_CHECK(false, "unknown argument '", a, "'");
    }
    const StrixwFile f(argv[1]);
    std::printf("%s: strixw v%u, %llu bytes, data at %llu\n", f.path().c_str(), kStrixwVersion,
                (unsigned long long)f.file_bytes(), (unsigned long long)f.data_offset());
    for (const auto &[k, v] : f.meta()) std::printf("  %-20s %s\n", k.c_str(), v.c_str());
    std::map<std::string, std::pair<int, uint64_t>> by_enc;
    for (const StrixwTensor &t : f.tensors()) {
        std::string shape;
        for (int64_t d : t.shape) shape += (shape.empty() ? "" : " x ") + std::to_string(d);
        std::printf("  %-14s %-24s %12llu B  %s%s\n", strixw_encoding_name(t.encoding), ("[" + shape + "]").c_str(),
                    (unsigned long long)t.bytes(), t.name.c_str(),
                    t.parts.empty() ? "" : ("  (" + std::to_string(t.parts.size()) + " parts)").c_str());
        auto &e = by_enc[strixw_encoding_name(t.encoding)];
        e.first += 1, e.second += t.bytes();
    }
    for (const auto &[enc, e] : by_enc)
        std::printf("%-14s %5d tensors %8.2f GiB\n", enc.c_str(), e.first, (double)e.second / (1ull << 30));
    if (verify) {
        f.verify_hashes(threads);
        std::printf("verify: all %zu tensors' component hashes match\n", f.tensors().size());
    }
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "inspect_strixw: %s\n", e.what());
        return 1;
    }
}
