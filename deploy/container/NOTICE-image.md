# Third-party software in this image

strixite itself is under the AGPL-3.0 (`/opt/strixite/LICENSE`); its own third-party code is listed in
`/opt/strixite/THIRD_PARTY.md`. The image also carries runtime libraries from AMD's ROCm, built by TheRock
(`therock-dist-linux-gfx1151`, the version in the image's labels), in `/opt/strixite/lib`. Each keeps its own
license; the texts are in `/opt/strixite/licenses/`.

| library in /opt/strixite/lib | project | license text in /opt/strixite/licenses/ |
|---|---|---|
| libamdhip64 | HIP runtime (ROCm) | `therock/hip/` |
| libhsa-runtime64 | ROCr runtime (ROCm) | `therock/rocr/` |
| libamd_comgr | Code Object Manager (ROCm) | `therock/amd_comgr/` |
| librocprofiler-register | rocprofiler-register (ROCm) | `therock/rocprofiler-register/` |
| librocm_kpack | rocm-kpack, part of TheRock | TheRock's license: https://github.com/ROCm/TheRock |
| libLLVM, libclang-cpp | LLVM / Clang (as built by ROCm) | `llvm-libs/` |
| librocm_sysdeps_z | zlib (the text from Fedora's zlib-ng, under the same zlib license) | `zlib/` |
| librocm_sysdeps_zstd | zstd | `zstd/` |
| librocm_sysdeps_elf | elfutils libelf (LGPL) | `elfutils-libelf/` |
| librocm_sysdeps_drm, _drm_amdgpu | libdrm | `libdrm/` |
| librocm_sysdeps_numa | numactl libnuma (LGPL) | `numactl-libs/` |
| librocm_sysdeps_liblzma | xz liblzma | `xz-libs/` |
| librocm_sysdeps_bz2 | bzip2 | `bzip2-libs/` |

The upstream libraries (LLVM and the `rocm_sysdeps` ones) are TheRock's builds of those projects; their license
texts here come from Fedora's packages of the same projects. Their source is available from each upstream project
and from https://github.com/ROCm/TheRock (the sources TheRock builds from). The LGPL libraries are shared libraries
loaded at run time, so they can be replaced: put a build of your own in `/opt/strixite/lib` (or earlier on
`LD_LIBRARY_PATH`).

The image's base is Fedora's `fedora-minimal`; its packages carry their own licenses under `/usr/share/licenses/`.
